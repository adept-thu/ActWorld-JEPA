"""Outcome-JEPA projection is consumed directly by planning heads."""
import torch
from torch.nn import functional as F
from .actworld_refine import RefineOutcomeJEPA

class ProjectedOutcomeJEPA(RefineOutcomeJEPA):
    def forward(self,tokens,candidates,base_logits,zero_actions=False):
        out=super().forward(tokens,candidates,base_logits,zero_actions)
        projected=self.projector(out['latent'])
        predicted_l2=F.softplus(self.cost(projected).squeeze(-1))
        cost=(predicted_l2*self.horizon_weights).sum(-1)
        centered=cost-cost[:,:1]
        correction=(centered-centered.mean(1,keepdim=True)).clamp(-3,3)
        raw=self.refine(projected).tanh()
        b,m,t,d=raw.shape
        smooth=F.avg_pool1d(F.pad(raw.reshape(b*m,t,d).transpose(1,2),(1,1),mode='replicate'),3,stride=1).transpose(1,2).reshape(b,m,t,d)
        ramp=torch.arange(1,t+1,device=raw.device,dtype=raw.dtype)/t
        offset=.5*smooth*ramp[None,None,:,None]
        out.update(predicted_l2=predicted_l2,cost=cost,logits=base_logits.detach().float()-self.score_scale*correction,refinement=offset,refined_candidates=candidates.detach()+offset,planning_latent=projected)
        return out

    def loss(self,prediction,candidates,gt_delta,valid):
        losses=super().loss(prediction,candidates,gt_delta,valid)
        gt=gt_delta.float().cumsum(-2)
        error=candidates.detach().float()-gt[:,None]
        distance=error.square().sum(-1).sqrt()
        velocity_error=torch.diff(error,dim=2,prepend=torch.zeros_like(error[:,:,:1]))
        # Absolute GT-dependent outcome: subtracting baseline error would cancel
        # the GT from the signed position channels. No labels enter forward().
        outcome=torch.cat([error/2,distance[...,None]/2,velocity_error.square().sum(-1).sqrt()[...,None]],-1)
        with torch.no_grad():
            phase=outcome.clamp(-5,5)@self.outcome_projection
            target=F.normalize(torch.cat([phase.sin(),phase.cos()],-1),dim=-1)
        z=F.normalize(prediction['planning_latent'],dim=-1)
        mask=valid.float().reshape(-1,1,12).expand_as(distance)
        alignment=((1-(z*target).sum(-1))*mask).sum()/mask.sum().clamp_min(1)
        losses['loss_aw_jepa']=self.jepa_weight*alignment
        losses['aw_alignment']=alignment.detach()
        return losses
