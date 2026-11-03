"""Stage 3b: full fine-tuning of the severity-conditioned restoration copy.

The frozen analysis copy (Stage-2 trunk + extractor) is *not* part of this
model: its full-image logits are cached by scripts/cache_severity_full_conditions.py,
so restoration gradients can never reach it.

Loss:  L = L1(y_hat, y) + fft_weight * L1(|FFT(y_hat)|, |FFT(y)|) + kl_weight * KL(p_hat || q)
       (FFT uses norm="ortho"; default weights 1.0 and 0.01 in the Stage 3b YAML)
Training protocol matches the DCPT reproduction: AdamW, grad clip 0.01, EMA 0.999.
"""
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

_CONDITION_PREFIXES = ("calibrator", "null_logits", "gate_logit", "condition_projector",
                       "middle_film", "decoder_film", "skip_film")
_ENCODER_PREFIXES = ("intro", "encoders", "downs", "middle_blks")


def fft_amplitude_l1(output, target):
    return F.l1_loss(torch.fft.rfft2(output.float(), norm="ortho").abs(),
                     torch.fft.rfft2(target.float(), norm="ortho").abs())


@MODEL_REGISTRY.register()
class SeverityFullFinetuneModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        network = build_network(opt["network_g"])
        # Stage 3a: freeze the trunk BEFORE the DDP wrap (DDP must not expect its grads).
        # Stage 3b (default): empty list -> everything trainable.
        frozen = tuple(opt.get("train", {}).get("frozen_prefixes", []) or []) if self.is_train else ()
        for name, parameter in network.named_parameters():
            parameter.requires_grad = not (frozen and name.startswith(frozen))
        self.net_g = self.model_to_device(network)
        path = opt["path"]
        if path.get("pretrain_network_g"):
            self.load_network(self.net_g, path["pretrain_network_g"],
                              path.get("strict_load_g", False), path.get("param_key_g", "params"))
        if self.is_train:
            self._init_training_settings()

    # ---------------------------------------------------------------- setup
    def _init_training_settings(self):
        train = self.opt["train"]
        self.net_g.train()
        self.grad_clip = self.opt.get("grad_clip", 0)
        self.fft_weight = train.get("fft_loss_weight", 0.0)
        self.kl_weight = train.get("kl_loss_weight", 0.0)
        self.condition_dropout = train.get("condition_dropout", 0.0)
        self.condition_dropout_warmup = train.get("condition_dropout_warmup", 0)

        self.ema_decay = train.get("ema_decay", 0)
        if self.ema_decay > 0:
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            load_path = self.opt["path"].get("pretrain_network_g")
            if load_path:
                # Falls back to "params" when the checkpoint has no EMA copy (Stage 3a).
                self.load_network(self.net_g_ema, load_path,
                                  self.opt["path"].get("strict_load_g", False), "params_ema")
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        optim = train["optim_g"].copy()
        optim_type, base_lr = optim.pop("type"), optim.pop("lr")
        encoder_prefixes = tuple(train.get("encoder_prefixes", _ENCODER_PREFIXES))
        groups = {"condition": [], "encoder": [], "decoder": []}
        frozen_count = 0
        for name, parameter in self.get_bare_model(self.net_g).named_parameters():
            if not parameter.requires_grad:  # frozen in __init__ (Stage 3a)
                frozen_count += parameter.numel()
                continue
            if name.startswith(_CONDITION_PREFIXES):
                groups["condition"].append(parameter)
            elif name.startswith(encoder_prefixes):
                groups["encoder"].append(parameter)
            else:
                groups["decoder"].append(parameter)
        multipliers = {"condition": train.get("condition_lr_mult", 1.0),
                       "encoder": train.get("encoder_lr_mult", 1.0), "decoder": 1.0}
        param_groups = [{"params": params, "lr": base_lr * multipliers[key], "name": key}
                        for key, params in groups.items() if params]
        logger = get_root_logger()
        logger.info(f"Stage 3 frozen parameters: {frozen_count / 1e6:.2f}M")
        for group in param_groups:
            count = sum(p.numel() for p in group["params"])
            logger.info(f"Stage 3b param group {group['name']}: {count / 1e6:.2f}M params, lr {group['lr']:.2e}")
        self.optimizer_g = self.get_optimizer(optim_type, param_groups, base_lr, **optim)
        self.optimizers.append(self.optimizer_g)
        self.setup_schedulers()

    # ---------------------------------------------------------------- training
    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.gt = data["gt"].to(self.device, non_blocking=True)
        self.logits = data["logits"].to(self.device, non_blocking=True)
        self.settings = data.get("setting")

    def optimize_parameters(self, current_iter, accumulation_index=0, accumulation_steps=1):
        if accumulation_index == 0:
            self.optimizer_g.zero_grad(set_to_none=True)
        with self.gradient_accumulation_context(
            [self.net_g], accumulation_index, accumulation_steps
        ):
            keep = None
            if self.condition_dropout > 0:
                rate = self.condition_dropout
                if self.condition_dropout_warmup > 0:  # linear ramp from 0
                    rate *= min(1.0, current_iter / self.condition_dropout_warmup)
                keep = (torch.rand(self.lq.shape[0], device=self.device) >= rate).float()
            self.output, condition = self.net_g(
                self.lq, self.logits, condition_keep=keep, return_condition=True
            )

            l_pix = F.l1_loss(self.output, self.gt)
            loss_dict = OrderedDict(l_pix=l_pix)
            total = l_pix
            if self.fft_weight > 0:
                l_fft = fft_amplitude_l1(self.output, self.gt)
                total = total + self.fft_weight * l_fft
                loss_dict["l_fft"] = l_fft
            if condition["p_hat"] is not None:
                p_hat, q = condition["p_hat"], condition["q"]
                l_kl = (
                    p_hat
                    * (p_hat.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())
                ).sum(-1).sum(-1).mean()
                if self.kl_weight > 0:
                    total = total + self.kl_weight * l_kl
                loss_dict["l_kl"] = l_kl
            if condition["gate"] is not None:
                loss_dict["gate"] = condition["gate"].mean().detach()
            loss_dict["l_total"] = total
            (total / accumulation_steps).backward()
        if accumulation_index + 1 == accumulation_steps:
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)
            self.optimizer_g.step()
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)
        self.accumulate_loss_dict(loss_dict, accumulation_index, accumulation_steps)

    # ---------------------------------------------------------------- evaluation
    def _eval_net(self):
        return self.net_g_ema if hasattr(self, "net_g_ema") else self.get_bare_model(self.net_g)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        net = self._eval_net()
        was_training = net.training
        net.eval()
        psnr, ssim = defaultdict(list), defaultdict(list)
        with torch.inference_mode():
            for data in dataloader:
                self.feed_data(data)
                output = net(self.lq, self.logits).clamp(0, 1).float().cpu().numpy()
                gt = self.gt.clamp(0, 1).float().cpu().numpy()
                for i, setting in enumerate(self.settings):
                    psnr[setting].append(calculate_psnr(output[i:i + 1], gt[i:i + 1], 0))
                    ssim[setting].append(calculate_ssim(output[i:i + 1], gt[i:i + 1], 0))
        logger = get_root_logger()
        rows = []
        for setting in sorted(psnr):
            p, s = float(np.mean(psnr[setting])), float(np.mean(ssim[setting]))
            rows.append(f"{setting}={p:.2f}/{s:.4f}")
            if tb_logger:
                tb_logger.add_scalar(f"psnr/{setting}", p, current_iter)
                tb_logger.add_scalar(f"ssim/{setting}", s, current_iter)
        mean_p = float(np.mean([np.mean(v) for v in psnr.values()]))
        mean_s = float(np.mean([np.mean(v) for v in ssim.values()]))
        logger.info(f"Stage 3b validation @ {current_iter}: avg PSNR {mean_p:.3f} SSIM {mean_s:.4f} | "
                    + " ".join(rows))
        if tb_logger:
            tb_logger.add_scalar("psnr/avg", mean_p, current_iter)
            tb_logger.add_scalar("ssim/avg", mean_s, current_iter)
        if was_training:
            net.train()

    def dist_validation(self, *args, **kwargs):
        if self.opt.get("rank", 0) == 0:
            self.nondist_validation(*args, **kwargs)

    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network([self.net_g, self.net_g_ema], "net_g", current_iter,
                              param_key=["params", "params_ema"])
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
