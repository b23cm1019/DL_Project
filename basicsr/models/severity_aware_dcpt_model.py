"""Three-stage severity-aware extension of paper-faithful DCPT.

Stage 1 retains DCPT's clean-image identity objective and replaces only the
flat degradation classifier with a feature-based ordinal extractor.  Stage 2
adapts that extractor to CDD-11 through type-only supervision plus synthetic
severity replay.  Stage 3 freezes the analysis path and trains downstream
conditioning/restoration modules on paired CDD-11 restoration.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


class _SeverityDCPTBase(BaseModel):
    def _build_analysis_networks(self, opt):
        self.net_g = self.model_to_device(build_network(opt["network_g"]))
        self.net_extractor = self.model_to_device(build_network(opt["network_extractor"]))

    def _load_analysis_networks(self, opt):
        path = opt["path"]
        if path.get("pretrain_network_g"):
            self.load_network(
                self.net_g, path["pretrain_network_g"], path.get("strict_load_g", True),
                path.get("param_key_g", "params"),
            )
        if path.get("pretrain_network_extractor"):
            self.load_network(
                self.net_extractor, path["pretrain_network_extractor"],
                path.get("strict_load_extractor", True), path.get("param_key_extractor", "params"),
            )

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.gt = data["gt"].to(self.device, non_blocking=True)
        self.severity_bins = data["severity_bins"].to(self.device, non_blocking=True)
        self.type_labels = data["type_labels"].to(self.device, non_blocking=True)
        self.has_severity = data["has_severity"].to(self.device, non_blocking=True).bool()

    def _severity_logits(self, image):
        encoder_features, middle_feature = self.net_g(image, return_degradation_features=True)
        return self.net_extractor(encoder_features, middle_feature)

    @staticmethod
    def _type_probabilities(logits):
        return 1.0 - logits.softmax(dim=-1)[..., 0]

    def _per_type_balanced_type_loss(self, logits):
        """Average balanced positive/negative BCE equally across primitives."""
        probabilities = self._type_probabilities(logits).clamp(1e-6, 1 - 1e-6)
        losses = []
        for type_index in range(probabilities.shape[1]):
            prediction = probabilities[:, type_index]
            target = self.type_labels[:, type_index]
            positive = target >= 0.5
            negative = ~positive
            parts = []
            if positive.any():
                parts.append(F.binary_cross_entropy(prediction[positive], target[positive]))
            if negative.any():
                parts.append(F.binary_cross_entropy(prediction[negative], target[negative]))
            if parts:
                losses.append(torch.stack(parts).mean())
        return torch.stack(losses).mean() if losses else logits.new_zeros(())

    @staticmethod
    def _per_type_ordinal_severity_loss(logits, bins, has_severity):
        """CDF/EMD ordinal loss over active synthetic severity bins 1--3.

        Bin zero represents *absence*, which belongs to type-presence BCE,
        rather than an ordinal severity level. Each primitive contributes one
        equally weighted term, preventing low/haze composition frequency from
        dominating rain/snow supervision.
        """
        losses = []
        for type_index in range(logits.shape[1]):
            active = has_severity & (bins[:, type_index] > 0)
            if not active.any():
                continue
            conditional_logits = logits[active, type_index, 1:]
            conditional_target = bins[active, type_index] - 1
            predicted_cdf = conditional_logits.softmax(dim=-1).cumsum(dim=-1)[:, :-1]
            target_cdf = (
                torch.arange(conditional_logits.shape[-1] - 1, device=logits.device)[None, :]
                >= conditional_target[:, None]
            ).to(predicted_cdf.dtype)
            losses.append((predicted_cdf - target_cdf).square().mean())
        return torch.stack(losses).mean() if losses else logits.new_zeros(())

    def _validation_metrics(self, dataloader, current_iter, tb_logger):
        self.net_g.eval()
        self.net_extractor.eval()
        severity_correct = severity_total = 0
        severity_absolute_error = 0.0
        type_tp = type_fp = type_fn = 0
        exact = type_total = 0
        with torch.inference_mode():
            for data in dataloader:
                self.feed_data(data)
                logits = self._severity_logits(self.lq)
                # Absence-bin accuracy is a type metric, not a severity metric.
                severity_prediction = logits[..., 1:].argmax(dim=-1) + 1
                active_severity = self.has_severity[:, None] & (self.severity_bins > 0)
                if active_severity.any():
                    severity_correct += (
                        severity_prediction[active_severity]
                        == self.severity_bins[active_severity]
                    ).sum().item()
                    severity_total += active_severity.sum().item()
                    severity_absolute_error += (
                        severity_prediction[active_severity].float()
                        - self.severity_bins[active_severity].float()
                    ).abs().sum().item()
                types = self._type_probabilities(logits) >= 0.5
                target = self.type_labels >= 0.5
                type_tp += (types & target).sum().item()
                type_fp += (types & ~target).sum().item()
                type_fn += (~types & target).sum().item()
                exact += (types == target).all(dim=1).sum().item()
                type_total += target.shape[0]
        precision = type_tp / max(type_tp + type_fp, 1)
        recall = type_tp / max(type_tp + type_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics = {
            "severity_accuracy": severity_correct / max(severity_total, 1),
            "severity_mae": severity_absolute_error / max(severity_total, 1),
            "type_micro_f1": f1,
            "type_exact_match": exact / max(type_total, 1),
        }
        get_root_logger().info("Severity-aware validation: " + ", ".join(
            f"{key}={value:.4f}" for key, value in metrics.items()))
        if tb_logger:
            for key, value in metrics.items():
                tb_logger.add_scalar(f"metrics/{key}", value, current_iter)
        self.net_g.train()
        self.net_extractor.train()

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        self._validation_metrics(dataloader, current_iter, tb_logger)

    def dist_validation(self, *args, **kwargs):
        if self.opt.get("rank", 0) == 0:
            self.nondist_validation(*args, **kwargs)

    def save_analysis(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_network(self.net_extractor, "net_extractor", current_iter)
        self.save_training_state(epoch, current_iter)


@MODEL_REGISTRY.register()
class SeverityAwareDCPTPretrainModel(_SeverityDCPTBase):
    """Stage 1: DCPT clean identity plus synthetic type/severity supervision."""

    def __init__(self, opt):
        super().__init__(opt)
        self._build_analysis_networks(opt)
        self._load_analysis_networks(opt)
        if self.is_train:
            train = opt["train"]
            self.cri_pix = build_loss(train["pixel_opt"]).to(self.device)
            self._setup_optimizers(train)
            self.setup_schedulers()

    def _setup_optimizers(self, train):
        generator = train["optim_g"].copy()
        extractor = train["optim_extractor"].copy()
        self.optimizer_g = self.get_optimizer(generator.pop("type"), self.net_g.parameters(), **generator)
        self.optimizer_extractor = self.get_optimizer(
            extractor.pop("type"), self.net_extractor.parameters(), **extractor
        )
        self.optimizers.extend([self.optimizer_g, self.optimizer_extractor])

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.optimizer_extractor.zero_grad()
        # Paper-faithful DCPT identity path: clean image -> clean image.
        identity_loss = self.cri_pix(self.net_g(self.gt), self.gt)
        logits = self._severity_logits(self.lq)
        train = self.opt["train"]
        severity_loss = self._per_type_ordinal_severity_loss(
            logits, self.severity_bins, self.has_severity
        )
        type_loss = self._per_type_balanced_type_loss(logits)
        total = (
            train.get("identity_loss_weight", 1.0) * identity_loss
            + train.get("severity_loss_weight", 1.0) * severity_loss
            + train.get("type_loss_weight", 0.0) * type_loss
        )
        total.backward()
        self.optimizer_g.step()
        self.optimizer_extractor.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(
            l_identity=identity_loss, l_severity=severity_loss, l_type=type_loss, l_total=total
        ))

    def save(self, epoch, current_iter):
        self.save_analysis(epoch, current_iter)


@MODEL_REGISTRY.register()
class SeverityExtractorAdaptationModel(_SeverityDCPTBase):
    """Stage 2: CDD type adaptation with synthetic severity replay."""

    _ANALYSIS_PREFIXES = ("intro", "encoders", "downs", "middle_blks")

    def __init__(self, opt):
        super().__init__(opt)
        self._build_analysis_networks(opt)
        self._load_analysis_networks(opt)
        analysis_prefixes = tuple(opt.get("analysis_parameter_prefixes", self._ANALYSIS_PREFIXES))
        for name, parameter in self.net_g.named_parameters():
            # `model_to_device` wraps even a one-GPU launcher in DDP, adding
            # `module.` to names returned by `named_parameters()`.  Classify
            # the underlying NAFNet name so Stage 2 has the same freeze policy
            # in distributed and non-distributed execution.
            bare_name = name.removeprefix("module.")
            parameter.requires_grad = bare_name.startswith(analysis_prefixes)
        if self.is_train:
            train = opt["train"]
            self.cri_pix = build_loss(train["pixel_opt"]).to(self.device)
            self._setup_optimizers(train)
            self.setup_schedulers()

    def _setup_optimizers(self, train):
        generator = train["optim_g"].copy()
        extractor = train["optim_extractor"].copy()
        self.optimizer_g = self.get_optimizer(
            generator.pop("type"), [p for p in self.net_g.parameters() if p.requires_grad], **generator
        )
        self.optimizer_extractor = self.get_optimizer(
            extractor.pop("type"), self.net_extractor.parameters(), **extractor
        )
        self.optimizers.extend([self.optimizer_g, self.optimizer_extractor])

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.optimizer_extractor.zero_grad()
        train = self.opt["train"]
        # Decoder is frozen, but identity gradients preserve the transferable
        # analysis representation while adapting it to CDD-11 appearance.
        identity_loss = self.cri_pix(self.net_g(self.gt), self.gt)
        logits = self._severity_logits(self.lq)
        replay_loss = self._per_type_ordinal_severity_loss(
            logits, self.severity_bins, self.has_severity
        )
        type_loss = self._per_type_balanced_type_loss(logits)
        total = (
            train.get("identity_loss_weight", 1.0) * identity_loss
            + train.get("replay_severity_loss_weight", 1.0) * replay_loss
            + train.get("type_loss_weight", 1.0) * type_loss
        )
        total.backward()
        self.optimizer_g.step()
        self.optimizer_extractor.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(
            l_identity=identity_loss, l_replay_severity=replay_loss, l_type=type_loss, l_total=total
        ))

    def save(self, epoch, current_iter):
        self.save_analysis(epoch, current_iter)


@MODEL_REGISTRY.register()
class FrozenFeatureConditionedRestorationModel(BaseModel):
    """Stage 3: restore CDD-11 images without moving extractor inputs."""

    _TRAINABLE_GENERATOR_PREFIXES = (
        "condition_projector", "middle_film", "decoder_film", "ups", "decoder", "ending"
    )

    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = self.model_to_device(build_network(opt["network_g"]))
        self.net_extractor = self.model_to_device(build_network(opt["network_extractor"]))
        path = opt["path"]
        self.load_network(
            self.net_g, path["pretrain_network_g"], path.get("strict_load_g", False),
            path.get("param_key_g", "params"),
        )
        self.load_network(
            self.net_extractor, path["pretrain_network_extractor"],
            path.get("strict_load_extractor", True), path.get("param_key_extractor", "params"),
        )
        for parameter in self.net_extractor.parameters():
            parameter.requires_grad = False
        self.net_extractor.eval()
        trainable_prefixes = tuple(
            opt.get("stage3_trainable_parameter_prefixes", self._TRAINABLE_GENERATOR_PREFIXES)
        )
        for name, parameter in self.net_g.named_parameters():
            # The one-GPU ``pytorch`` launcher still wraps the model in DDP,
            # which exposes parameter names as ``module.<NAFNet name>``.
            # Select on the underlying name, otherwise Stage 3 finds no
            # trainable projector/FiLM/decoder parameters and AdamW receives
            # an empty parameter list.
            bare_name = name.removeprefix("module.")
            parameter.requires_grad = bare_name.startswith(trainable_prefixes)
        if self.is_train:
            self.cri_pix = build_loss(opt["train"]["pixel_opt"]).to(self.device)
            optim = opt["train"]["optim_g"].copy()
            self.optimizer_g = self.get_optimizer(
                optim.pop("type"), [p for p in self.net_g.parameters() if p.requires_grad], **optim
            )
            self.optimizers.append(self.optimizer_g)
            self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.gt = data["gt"].to(self.device, non_blocking=True)

    @torch.no_grad()
    def _condition(self):
        # Stable, pre-adapter feature tap: neither restoration loss nor FiLM can
        # move the frozen extractor's input distribution.
        encoder_features, middle_feature = self.net_g(
            self.lq, return_degradation_features=True
        )
        logits = self.net_extractor(encoder_features, middle_feature)
        return logits.softmax(dim=-1).flatten(1)

    def optimize_parameters(self, current_iter):
        self.net_g.train()
        self.net_extractor.eval()
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self._condition())
        loss = self.cri_pix(self.output, self.gt)
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_restoration=loss))

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        self.net_g.eval()
        total = count = 0.0
        with torch.inference_mode():
            for data in dataloader:
                self.feed_data(data)
                output = self.net_g(self.lq, self._condition()).clamp(0, 1)
                total += self.cri_pix(output, self.gt).item() * output.shape[0]
                count += output.shape[0]
        l1 = total / max(count, 1)
        get_root_logger().info("Stage 3 validation: l1=%.6f", l1)
        if tb_logger:
            tb_logger.add_scalar("metrics/l1", l1, current_iter)
        self.net_g.train()

    def dist_validation(self, *args, **kwargs):
        if self.opt.get("rank", 0) == 0:
            self.nondist_validation(*args, **kwargs)
