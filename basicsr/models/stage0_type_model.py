"""Training model for Stage 0a CDD-11 type recognition."""
from collections import OrderedDict

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class Stage0TypeModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = self.model_to_device(build_network(opt["network_g"]))
        if self.is_train:
            self.cri_type = build_loss(opt["train"]["type_opt"]).to(self.device)
            self.setup_optimizers()
            self.setup_schedulers()

    def setup_optimizers(self):
        config = self.opt["train"]["optim_g"].copy()
        optimizer_type = config.pop("type")
        self.optimizer_g = self.get_optimizer(optimizer_type, self.net_g.parameters(), **config)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.type_labels = data["type_labels"].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        logits = self.net_g(self.lq)
        loss = self.cri_type(logits, self.type_labels)
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_type=loss))

    @torch.no_grad()
    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        self.net_g.eval()
        tp = torch.zeros(self.opt["network_g"]["num_types"], device=self.device)
        fp = torch.zeros_like(tp)
        fn = torch.zeros_like(tp)
        exact = total = 0
        for data in dataloader:
            self.feed_data(data)
            predictions = self.net_g(self.lq).sigmoid() >= 0.5
            targets = self.type_labels >= 0.5
            tp += (predictions & targets).sum(0)
            fp += (predictions & ~targets).sum(0)
            fn += (~predictions & targets).sum(0)
            exact += (predictions == targets).all(dim=1).sum().item()
            total += targets.shape[0]
        precision = tp / (tp + fp).clamp_min(1)
        recall = tp / (tp + fn).clamp_min(1)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
        macro_f1 = f1.mean().item()
        exact_accuracy = exact / max(total, 1)
        get_root_logger().info(
            "Stage 0a validation: macro_f1=%.4f exact_match=%.4f per_type_f1=%s",
            macro_f1, exact_accuracy, ", ".join(f"{x:.4f}" for x in f1.tolist()))
        if tb_logger:
            tb_logger.add_scalar("metrics/macro_f1", macro_f1, current_iter)
            tb_logger.add_scalar("metrics/exact_match", exact_accuracy, current_iter)
        self.net_g.train()

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
