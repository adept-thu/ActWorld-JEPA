from mmdet.models import DETECTORS
from .actworld_momadv2_refine import ActWorldMomADv2Refine
@DETECTORS.register_module()
class ActWorldMomADv2RefineConservative(ActWorldMomADv2Refine):
    def simple_test(self,img,**data):
        rows=super().simple_test(img,**data)
        for row in rows:
            r=row['img_bbox']
            if 'actworld_selected_mode' not in r:continue
            rejected=r['actworld_selected_mode']!=r['actworld_baseline_mode']
            r['actworld_switch_rejected']=rejected
            if rejected:
                r['final_planning']=r['actworld_baseline_planning'].clone()
                r['actworld_selected_mode']=r['actworld_baseline_mode']
        return rows
