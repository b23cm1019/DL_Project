from collections import OrderedDict
import torch
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class CLIPFeedbackRefinementTrainModel(BaseModel):
    """Adapter-only trainer; avoids constructing an optimizer for Row-C weights."""
    def __init__(self,opt):
        super().__init__(opt)
        net_g = build_network(opt['network_g'])
        # Freeze Row-C before DistributedDataParallel sees the module. Otherwise
        # DDP expects gradients for the frozen backbone and fails its reduction.
        for name, parameter in net_g.named_parameters():
            parameter.requires_grad = "feedback_projector" in name or "_film" in name
        self.net_g = self.model_to_device(net_g)
        self.load_network(self.net_g, opt['path']['pretrain_network_g'], False, opt['path'].get('param_key_g', 'params'))
        if self.is_train:
            self.cri_pix = build_loss(opt['train']['pixel_opt']).to(self.device)
            c = opt['train']['optim_g'].copy()
            typ = c.pop('type')
            self.optimizer_g = self.get_optimizer(
                typ, [p for p in self.net_g.parameters() if p.requires_grad], **c
            )
            self.optimizers.append(self.optimizer_g)
            self.setup_schedulers()
    def feed_data(self,data): self.lq=data['lq'].to(self.device); self.gt=data['gt'].to(self.device); self.feedback=data['feedback'].to(self.device)
    def optimize_parameters(self,current_iter):
        self.optimizer_g.zero_grad(); self.output=self.net_g(self.lq,self.feedback); loss=self.cri_pix(self.output,self.gt); loss.backward(); self.optimizer_g.step(); self.log_dict=self.reduce_loss_dict(OrderedDict(l_refine=loss))
    def save(self,epoch,current_iter): self.save_network(self.net_g,'net_g',current_iter); self.save_training_state(epoch,current_iter)
    def nondist_validation(self,dataloader,current_iter,tb_logger,save_img=False,clamp=True):
        self.net_g.eval(); mse=0.; n=0
        with torch.no_grad():
            for data in dataloader: self.feed_data(data); out=self.net_g(self.lq,self.feedback).clamp(0,1); mse += ((out-self.gt).square().mean(dim=(1,2,3))).sum().item(); n+=out.shape[0]
        psnr=10*torch.log10(torch.tensor(1/(mse/max(n,1)+1e-12))).item(); self.net_g.train()
        if tb_logger: tb_logger.add_scalar('metrics/psnr',psnr,current_iter)
    def dist_validation(self,*args,**kwargs):
        if self.opt['rank']==0:self.nondist_validation(*args,**kwargs)
