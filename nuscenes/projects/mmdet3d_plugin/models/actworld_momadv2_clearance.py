from mmdet.models import DETECTORS
from .actworld_momadv2_conservative import ActWorldMomADv2RefineConservative
from .actworld_clearance_gate import clearance_decreased
@DETECTORS.register_module()
class ActWorldMomADv2Clearance(ActWorldMomADv2RefineConservative):
    def simple_test(self,img,**data):
        rows=super().simple_test(img,**data)
        for row in rows:
            r=row['img_bbox']
            if 'actworld_baseline_planning' not in r:continue
            rejected=clearance_decreased(r)
            r['actworld_clearance_rejected']=rejected
            if rejected:r['final_planning']=r['actworld_baseline_planning'].clone()
        return rows
