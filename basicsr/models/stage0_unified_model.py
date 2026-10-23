"""Unified Stage 0 model for synthetic severity and CDD-11 type supervision."""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from basicsr.archs import build_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Stage0UnifiedModel(BaseModel):
    """One [B, type, severity] head with masked multi-source objectives."""

    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = self.model_to_device(build_network(opt["network_g"]))
        load_path = opt["path"].get("pretrain_network_g")
        if load_path:
            self.load_network(
                self.net_g,
                load_path,
                opt["path"].get("strict_load_g", False),
                opt["path"].get("param_key_g", "params"),
            )
        if self.is_train:
            self.setup_optimizers()
            self.setup_schedulers()

    def setup_optimizers(self):
        config = self.opt["train"]["optim_g"].copy()
        optimizer_type = config.pop("type")
        self.optimizer_g = self.get_optimizer(optimizer_type, self.net_g.parameters(), **config)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.severity_bins = data["severity_bins"].to(self.device, non_blocking=True)
        self.severity_soft = data["severity_soft"].to(self.device, non_blocking=True)
        self.type_labels = data["type_labels"].to(self.device, non_blocking=True)
        self.severity_mask = data["severity_mask"].to(self.device, non_blocking=True).bool()
        self.type_mask = data["type_mask"].to(self.device, non_blocking=True).bool()

    def _losses(self, logits):
        severity_loss = logits.new_zeros(())
        type_loss = logits.new_zeros(())
        if self.severity_mask.any():
            selected = logits[self.severity_mask]
            targets = self.severity_soft[self.severity_mask]
            severity_loss = -(targets * F.log_softmax(selected, dim=-1)).sum(dim=-1).mean()
        if self.type_mask.any():
            selected = logits[self.type_mask]
            probabilities = selected.softmax(dim=-1)[..., 1:].sum(dim=-1)
            targets = self.type_labels[self.type_mask]
            type_loss = F.binary_cross_entropy(probabilities.clamp(1e-6, 1 - 1e-6), targets)
        return severity_loss, type_loss

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        severity_loss, type_loss = self._losses(self.net_g(self.lq))
        train_opt = self.opt["train"]
        total = train_opt.get("severity_loss_weight", 1.0) * severity_loss
        total = total + train_opt.get("type_loss_weight", 1.0) * type_loss
        total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(
            l_severity=severity_loss, l_type=type_loss, l_total=total))

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        self.net_g.eval()
        severity_correct = torch.zeros(self.opt["network_g"]["num_types"], device=self.device)
        severity_adjacent = torch.zeros_like(severity_correct)
        severity_abs_error = torch.zeros_like(severity_correct)
        severity_total = torch.zeros_like(severity_correct)
        tp = torch.zeros_like(severity_correct)
        fp = torch.zeros_like(severity_correct)
        fn = torch.zeros_like(severity_correct)
        exact = type_total = 0
        for data in dataloader:
            self.feed_data(data)
            logits = self.net_g(self.lq)
            if self.severity_mask.any():
                prediction = logits[self.severity_mask].argmax(dim=-1)
                target = self.severity_bins[self.severity_mask]
                severity_correct += (prediction == target).sum(0)
                severity_adjacent += ((prediction - target).abs() <= 1).sum(0)
                severity_abs_error += (prediction - target).abs().sum(0)
                severity_total += target.shape[0]
            if self.type_mask.any():
                prediction = logits[self.type_mask].softmax(-1)[..., 1:].sum(-1) >= 0.5
                target = self.type_labels[self.type_mask] >= 0.5
                tp += (prediction & target).sum(0)
                fp += (prediction & ~target).sum(0)
                fn += (~prediction & target).sum(0)
                exact += (prediction == target).all(1).sum().item()
                type_total += target.shape[0]
        severity_accuracy = (severity_correct / severity_total.clamp_min(1)).mean().item()
        adjacent_accuracy = (severity_adjacent / severity_total.clamp_min(1)).mean().item()
        severity_mae = (severity_abs_error / severity_total.clamp_min(1)).mean().item()
        f1 = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
        macro_f1 = f1.mean().item()
        exact_match = exact / max(type_total, 1)
        get_root_logger().info(
            "Stage 0 validation: severity_accuracy=%.4f severity_mae=%.4f adjacent=%.4f type_macro_f1=%.4f type_exact=%.4f",
            severity_accuracy, severity_mae, adjacent_accuracy, macro_f1, exact_match)
        if tb_logger:
            tb_logger.add_scalar("metrics/severity_accuracy", severity_accuracy, current_iter)
            tb_logger.add_scalar("metrics/severity_mae", severity_mae, current_iter)
            tb_logger.add_scalar("metrics/severity_adjacent_accuracy", adjacent_accuracy, current_iter)
            tb_logger.add_scalar("metrics/type_macro_f1", macro_f1, current_iter)
            tb_logger.add_scalar("metrics/type_exact_match", exact_match, current_iter)
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        """Run validation on rank zero for the single-GPU distributed launcher."""
        if self.opt.get("rank", 0) == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, clamp)

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
