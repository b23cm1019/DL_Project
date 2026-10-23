from collections import OrderedDict

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class ConditionedSRModel(SRModel):
    """Stage 1 restorer: frozen Stage 0 extractor plus trainable FiLM NAFNet."""
    def __init__(self, opt):
        super().__init__(opt)
        self.net_extractor = self.model_to_device(build_network(opt["network_extractor"]), dist=False)
        self.load_network(self.net_extractor, opt["path"]["pretrain_network_extractor"], True, opt["path"].get("param_key_extractor", "params"))
        self.net_extractor.eval()
        for parameter in self.net_extractor.parameters():
            parameter.requires_grad = False

    def _condition(self, image):
        with torch.no_grad():
            return torch.softmax(self.net_extractor(image), dim=-1)

    def optimize_parameters(self, current_iter):
        self.net_g.train()
        self.optimizer_g.zero_grad()
        output = self.net_g(self.lq, self._condition(self.lq))
        loss = self.cri_pix(output, self.gt)
        logs = OrderedDict(l_pix=loss)
        idempotence_weight = self.opt["train"].get("idempotence_weight", 0.0)
        if idempotence_weight:
            clean_output = self.net_g(self.gt, self._condition(self.gt))
            idem_loss = self.cri_pix(clean_output, self.gt)
            loss = loss + idempotence_weight * idem_loss
            logs["l_idem"] = idem_loss
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(logs)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq, self._condition(self.lq))
        self.net_g.train()
