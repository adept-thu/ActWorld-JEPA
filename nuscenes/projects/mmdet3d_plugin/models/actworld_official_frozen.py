import torch
from mmdet.models import DETECTORS
from .actworld_momadv2_candidate import ActWorldMomADv2Candidate
from .actworld_momadv2_matched import ActWorldMomADv2Matched
@DETECTORS.register_module()
class ActWorldMomADv2OfficialFrozen(ActWorldMomADv2Candidate):
 _branch_results=ActWorldMomADv2Matched._branch_results
 def simple_test(self,img,**data):
  with torch.no_grad():
   outputs=self.head(self.extract_feat(img),data)
   original=self.head.post_process(outputs,data)
   full=self._branch_results(outputs,data,original,self.actworld)
   return [dict(img_bbox=r) for r in full]
