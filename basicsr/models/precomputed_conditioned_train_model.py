"""Train a Row-C-initialized restorer using cached frozen conditions."""
from collections import OrderedDict

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class PrecomputedConditionedTrainModel(BaseModel):
    """Train a Row-C initialized conditioned restorer.

    ``network_g.freeze_backbone`` keeps the historical adapter-only pilot
    behavior.  For the official R4 experiment it is set to ``false`` so the
    restoration backbone and the conditioning adapter are optimized together.
    The extractor itself is not part of this model: its predictions are
    precomputed and remain frozen, which prevents restoration supervision from
    silently changing the degradation labels.
    """

    def __init__(self, opt):
        super().__init__(opt)
        network_opt = opt["network_g"].copy()
        freeze_backbone = network_opt.pop("freeze_backbone", True)
        network = build_network(network_opt)
        for name, parameter in network.named_parameters():
            parameter.requires_grad = (
                not freeze_backbone
                or "feedback_projector" in name
                or "_film" in name
            )
        self.net_g = self.model_to_device(network)
        self.load_network(
            self.net_g,
            opt["path"]["pretrain_network_g"],
            False,
            opt["path"].get("param_key_g", "params"),
        )
        if self.is_train:
            self.cri_pix = build_loss(opt["train"]["pixel_opt"]).to(self.device)
            optim = opt["train"]["optim_g"].copy()
            optim_type = optim.pop("type")
            parameters = [p for p in self.net_g.parameters() if p.requires_grad]
            self.optimizer_g = self.get_optimizer(optim_type, parameters, **optim)
            self.optimizers.append(self.optimizer_g)
            self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.gt = data["gt"].to(self.device, non_blocking=True)
        self.condition = data["condition"].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        self.net_g.train()
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.condition)
        loss = self.cri_pix(self.output, self.gt)
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_pilot=loss))

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        self.net_g.eval()
        total = 0.0
        count = 0
        with torch.inference_mode():
            for data in dataloader:
                self.feed_data(data)
                output = self.net_g(self.lq, self.condition).clamp(0, 1)
                total += self.cri_pix(output, self.gt).item() * output.shape[0]
                count += output.shape[0]
        if tb_logger:
            tb_logger.add_scalar("metrics/l1", total / max(count, 1), current_iter)
        self.net_g.train()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def dist_validation(self, *args, **kwargs):
        if self.opt["rank"] == 0:
            self.nondist_validation(*args, **kwargs)
