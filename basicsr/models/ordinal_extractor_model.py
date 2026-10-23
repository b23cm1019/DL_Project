from collections import OrderedDict

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class OrdinalExtractorModel(BaseModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.net_g = self.model_to_device(build_network(opt["network_g"]))
        if self.is_train:
            self.cri_classify = build_loss(opt["train"]["classify_opt"]).to(self.device)
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

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)
        loss = self.cri_classify(self.output, self.severity_bins)
        loss.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(OrderedDict(l_ordinal=loss))

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        self.net_g.eval()
        correct = torch.zeros(self.opt["network_g"]["num_types"], device=self.device)
        total = 0
        with torch.no_grad():
            for data in dataloader:
                self.feed_data(data)
                prediction = self.net_g(self.lq).argmax(dim=-1)
                correct += (prediction == self.severity_bins).sum(dim=0)
                total += prediction.shape[0]
        values = correct / max(total, 1)
        logger = get_root_logger()
        logger.info("Ordinal extractor validation: " + ", ".join(f"type_{i}={value:.4f}" for i, value in enumerate(values.tolist())))
        if tb_logger:
            for index, value in enumerate(values.tolist()):
                tb_logger.add_scalar(f"metrics/type_{index}_accuracy", value, current_iter)
        self.net_g.train()
