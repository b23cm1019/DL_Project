from collections import OrderedDict
from os import path as osp
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel

@MODEL_REGISTRY.register()
class PromptSRModel(SRModel):
    """SR model with Prompt Injection from frozen Degradation Classifier."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # Build frozen classifier net_dc
        self.net_dc = build_network(opt["network_dc"])
        self.net_dc = self.model_to_device(self.net_dc)
        self.net_dc.eval()
        for param in self.net_dc.parameters():
            param.requires_grad = False
            
        load_path_dc = self.opt["path"].get("pretrain_network_dc", None)
        if load_path_dc is not None:
            self.load_network(
                self.net_dc,
                load_path_dc,
                True,
                "params",
                False,
            )
            
        # Hook logic for net_dc
        self.hook_outputs = list()
        self.hooks = list()
        hook_names = self.opt.get("hook_names", "decoder")
        for name, module in self.net_g.named_modules():
            if hook_names in name and name.count(".") == 1:
                hook = module.register_forward_hook(self.hook_forward_fn)
                self.hooks.append(hook)
                
        if hasattr(self, "net_g_ema"):
            for name, module in self.net_g_ema.named_modules():
                if hook_names in name and name.count(".") == 1:
                    hook = module.register_forward_hook(self.hook_forward_fn)
                    self.hooks.append(hook)

    def hook_forward_fn(self, module, input, output):
        if isinstance(output, tuple):
            output = output[-1]
        self.hook_outputs.append(output)

    def optimize_parameters(self, current_iter):
        self.net_g.train()
        self.optimizer_g.zero_grad()
        
        # Pass 1: Get features (no prompt)
        self.hook_outputs = []
        with torch.no_grad():
            _ = self.net_g(self.lq)
            
        cls_output = self.net_dc(self.lq, self.hook_outputs[::-1])
        # Use sigmoid if BCE was used, or softmax/logits
        prompt = torch.sigmoid(cls_output)
        self.hook_outputs = []

        # Pass 2: Generation with prompt
        self.output = self.net_g(self.lq, prompt=prompt)
        self.hook_outputs = []

        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
            
        if self.cri_perceptual:
             l_percep, l_style = self.cri_perceptual(self.output, self.gt)
             if l_percep is not None:
                 l_total += l_percep
                 loss_dict["l_percep"] = l_percep
             if l_style is not None:
                 l_total += l_style
                 loss_dict["l_style"] = l_style

        l_total.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), self.grad_clip)
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.hook_outputs = []
                _ = self.net_g_ema(self.lq)
                cls_output = self.net_dc(self.lq, self.hook_outputs[::-1])
                prompt = torch.sigmoid(cls_output)
                self.hook_outputs = []
                self.output = self.net_g_ema(self.lq, prompt=prompt)
                self.hook_outputs = []
        else:
            self.net_g.eval()
            with torch.no_grad():
                 self.hook_outputs = []
                 _ = self.net_g(self.lq)
                 cls_output = self.net_dc(self.lq, self.hook_outputs[::-1])
                 prompt = torch.sigmoid(cls_output)
                 self.hook_outputs = []
                 self.output = self.net_g(self.lq, prompt=prompt)
                 self.hook_outputs = []
            self.net_g.train()

