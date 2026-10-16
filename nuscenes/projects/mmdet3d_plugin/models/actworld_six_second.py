import torch
from mmdet.models import DETECTORS
from .actworld_l2_focus import L2FocusOutcomeJEPA
from .actworld_temporal_safety import ActWorldMomADv2TemporalSafety
class SixSecondOutcomeJEPA(L2FocusOutcomeJEPA):
 def loss(self,prediction,candidates,gt_delta,valid):
  losses=super().loss(prediction,candidates,gt_delta,valid)
  gt=gt_delta.float().cumsum(-2)
  idx=prediction['base_logits'].argmax(-1);rows=torch.arange(len(idx),device=idx.device)
  error=((prediction['refined_candidates'][rows,idx]-gt).square().sum(-1)+1e-6).sqrt()
  weights=valid.float().reshape(-1,12)
  losses['loss_aw_six_second']=3.0*(error*weights).sum()/weights.sum().clamp_min(1)
  return losses
@DETECTORS.register_module()
class ActWorldMomADv2SixSecond(ActWorldMomADv2TemporalSafety):
 def __init__(self,actworld=None,**kwargs):
  super().__init__(actworld=actworld,**kwargs)
  self.actworld.__class__=SixSecondOutcomeJEPA
