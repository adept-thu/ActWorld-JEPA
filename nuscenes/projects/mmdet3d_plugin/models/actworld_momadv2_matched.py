import os
import torch
from mmdet.models import DETECTORS
from .actworld_momadv2_refine import ActWorldMomADv2Refine
from .actworld_refine import RefineOutcomeJEPA
from .actworld_clearance_gate import clearance_decreased

@DETECTORS.register_module()
class ActWorldMomADv2Matched(ActWorldMomADv2Refine):
    def __init__(self, control_checkpoint, actworld=None, **kwargs):
        super().__init__(actworld=actworld, **kwargs)
        self.control = RefineOutcomeJEPA(**(actworld or {}))
        state=torch.load(control_checkpoint,map_location='cpu')['state_dict']
        self.control.load_state_dict({k[len('actworld.'):]:v for k,v in state.items() if k.startswith('actworld.')},strict=True)
        self.control.requires_grad_(False)

    def _branch_results(self,outputs,data,original,branch,zero_actions=False):
        results=[dict(r) for r in original]
        tokens,candidates,base=self._candidate_inputs(outputs,data)
        prediction=branch(tokens,candidates,base,zero_actions=zero_actions)
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
        chosen=(revised-branch.score_scale*correction).argmax(-1)
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
        for r in results:
            switch=r['actworld_selected_mode']!=r['actworld_baseline_mode']
            if switch:r['final_planning']=r['actworld_baseline_planning'].clone()
            if clearance_decreased(r):r['final_planning']=r['actworld_baseline_planning'].clone()
        return results

    def simple_test(self,img,**data):
        with torch.no_grad():
            features=self.extract_feat(img)
            outputs=self.head(features,data)
            original=self.head.post_process(outputs,data)
            full=self._branch_results(outputs,data,original,self.actworld)
            control=self._branch_results(outputs,data,original,self.control)
            zero=self._branch_results(outputs,data,original,self.actworld,True)
            for i,r in enumerate(full):
                r['matched_no_jepa_planning']=control[i]['final_planning']
                r['matched_zero_actions_planning']=zero[i]['final_planning']
            return [dict(img_bbox=r) for r in full]
