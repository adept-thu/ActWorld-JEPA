from mmdet.models import DETECTORS
from .actworld_aligned_planner import AlignedPlannerOutcomeJEPA, ActWorldMomADv2AlignedPlanner
class L2FocusOutcomeJEPA(AlignedPlannerOutcomeJEPA):
 def loss(self,prediction,candidates,gt_delta,valid):
  losses=super().loss(prediction,candidates,gt_delta,valid)
  losses['loss_aw_refinement']=3.0*losses['loss_aw_refinement']
  losses['loss_aw_other_candidate_refinement']=0.2*losses['loss_aw_other_candidate_refinement']
  return losses
@DETECTORS.register_module()
class ActWorldMomADv2L2Focus(ActWorldMomADv2AlignedPlanner):
 def __init__(self,actworld=None,**kwargs):
  super().__init__(actworld=actworld,**kwargs)
  self.actworld.__class__=L2FocusOutcomeJEPA
