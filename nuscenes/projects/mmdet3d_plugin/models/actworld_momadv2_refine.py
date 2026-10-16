import os
import torch
from mmdet.models import DETECTORS
from .actworld_momadv2 import ActWorldMomADv2
from .actworld_refine import RefineOutcomeJEPA

@DETECTORS.register_module()
class ActWorldMomADv2Refine(ActWorldMomADv2):
    def __init__(self,actworld=None,**kwargs):
        super().__init__(actworld=actworld,**kwargs)
        self.actworld=RefineOutcomeJEPA(**(actworld or {}))

    def simple_test(self,img,**data):
        with torch.no_grad():
            features=self.extract_feat(img)
            outputs=self.head(features,data)
            results=self.head.post_process(outputs,data)
            if os.environ.get('ACTWORLD_DISABLE','0')=='1':
                return [dict(img_bbox=r) for r in results]
            tokens,candidates,base=self._candidate_inputs(outputs,data)
            prediction=self.actworld(tokens,candidates,base,zero_actions=os.environ.get('ACTWORLD_ZERO_ACTIONS','0')=='1')
            refined=prediction['refined_candidates']
            det,maps,motion,plan=outputs
            b=candidates.shape[0];rows=torch.arange(b,device=candidates.device)
            cmd=data['gt_ego_fut_cmd'].argmax(-1)
            full=plan['prediction'][-1].reshape(b,3,6,12,2).cumsum(-2).clone()
            full[rows,cmd]=refined
            cls=plan['classification'][-1].reshape(b,3,6)
            rescored,_=self.head.motion_plan_head.planning_decoder.select(det,motion,cls,full,data)
            revised=rescored[rows,cmd]
            baseline_mode=base.argmax(-1)
            cost=prediction['cost'];correction=(cost-cost.mean(-1,keepdim=True)).clamp(-3,3)
            chosen=(revised-self.actworld.score_scale*correction).argmax(-1)
            gain=cost[rows,baseline_mode]-cost[rows,chosen]
            chosen=torch.where(gain>=0.25,chosen,baseline_mode)
            # If the existing collision veto rejects a refined candidate,
            # fall back to the unmodified baseline output for this sample.
            rejected=revised[rows,chosen]<revised.amax(-1)-100
            for i,r in enumerate(results):
                r['actworld_baseline_planning']=r['final_planning'].clone()
                if not rejected[i]:r['final_planning']=refined[i,chosen[i]].cpu()
                r['actworld_selected_mode']=int(baseline_mode[i] if rejected[i] else chosen[i])
                r['actworld_baseline_mode']=int(baseline_mode[i])
                r['actworld_predicted_cost']=cost[i].cpu()
                r['actworld_refinement_rejected']=bool(rejected[i])
            return [dict(img_bbox=r) for r in results]
