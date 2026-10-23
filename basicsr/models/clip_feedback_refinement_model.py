from collections import OrderedDict

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class CLIPFeedbackRefinementModel(SRModel):
    """Trains only feedback projector/FiLM while preserving Row-C weights."""
    def __init__(self, opt):
        super().__init__(opt)
        for name, parameter in self.net_g.named_parameters():
            parameter.requires_grad = ("feedback_projector" in name or "_film" in name)

    def setup_optimizers(self):
        config = self.opt["train"]["optim_g"].copy()
        optimizer_type = config.pop("type")
        parameters = [p for p in self.net_g.parameters() if p.requires_grad]
        self.optimizer_g = self.get_optimizer(optimizer_type, parameters, **config)
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        self.gt = data["gt"].to(self.device, non_blocking=True)
        self.feedback = data["feedback"].to(self.device, non_blocking=True)

    def optimize_parameters(self, current_iter):
        self.net_g.train()
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq, self.feedback)
        loss = self.cri_pix(self.output, self.gt)
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_refine=loss))

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq, self.feedback)
        self.net_g.train()
