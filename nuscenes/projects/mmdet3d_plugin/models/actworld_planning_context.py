import torch
from mmdet.models import DETECTORS
from .actworld_official_frozen import ActWorldMomADv2OfficialFrozen
@DETECTORS.register_module()
class ActWorldMomADv2PlanningContext(ActWorldMomADv2OfficialFrozen):
 def _candidate_inputs(self,model_outs,data):
  tokens,candidates,base=super()._candidate_inputs(model_outs,data)
  b=candidates.shape[0]
  # Head updates this cache from CURRENT frame before returning outputs.
  # Detached planner representation: no future labels or baseline gradients.
  query=self.head.motion_plan_head.last_plan_query.detach().reshape(b,3,6,256)
  command=data['gt_ego_fut_cmd'].argmax(-1)
  rows=torch.arange(b,device=query.device)
  tokens=torch.cat([tokens,query[rows,command]],dim=1)
  return tokens,candidates,base
