import torch
from .actworld_projected import ProjectedOutcomeJEPA
class CandidateOutcomeJEPA(ProjectedOutcomeJEPA):
 def loss(self,prediction,candidates,gt_delta,valid):
  losses=super().loss(prediction,candidates,gt_delta,valid)
  gt=gt_delta.float().cumsum(-2)
  base=prediction['base_logits'];eligible=base>=base.amax(-1,keepdim=True)-100
  selected=base.argmax(-1);rows=torch.arange(len(selected),device=base.device)
  other=eligible.clone();other[rows,selected]=False
  weights=valid.float().reshape(-1,1,12)*self.horizon_weights[None,None,:]*other[:,:,None]
  error=((prediction['refined_candidates']-gt[:,None]).square().sum(-1)+1e-6).sqrt()
  losses['loss_aw_other_candidate_refinement']=.25*(error*weights).sum()/weights.sum().clamp_min(1)
  return losses
