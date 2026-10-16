import torch
from torch.nn import functional as F
from mmdet.models import DETECTORS
from .actworld_candidate import CandidateOutcomeJEPA
from .actworld_planning_context import ActWorldMomADv2PlanningContext
class AlignedPlannerOutcomeJEPA(CandidateOutcomeJEPA):
    def forward(self,tokens,candidates,base_logits,zero_actions=False):
        batch,modes,horizon,_=candidates.shape
        assert tokens.shape[1]==66, tokens.shape
        scene=self.scene(tokens.detach().float())
        aligned=scene[:,-6:]
        pooled=self.attention(self.query.expand(batch,-1,-1),scene,scene)[0][:,0]
        state=(pooled[:,None]+aligned).reshape(batch*modes,-1)
        position=candidates.detach().float()
        delta=torch.diff(position,dim=2,prepend=torch.zeros_like(position[:,:,:1]))
        acceleration=torch.diff(delta,dim=2,prepend=torch.zeros_like(delta[:,:,:1]))
        actions=torch.cat([position/30,delta/5,acceleration/5],dim=-1)
        if zero_actions:actions=torch.zeros_like(actions)
        rollout=[]
        for t in range(horizon):
            action=self.action(actions[:,:,t].reshape(batch*modes,6))
            conditioned=action+aligned.reshape(batch*modes,-1)
            context=self.attention(conditioned.reshape(batch,modes,-1),scene,scene)[0].reshape(batch*modes,-1)
            state=self.transition(conditioned+context,state)
            rollout.append(state.reshape(batch,modes,-1))
        out={'latent':torch.stack(rollout,dim=2),'base_logits':base_logits.detach().float()}
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


@DETECTORS.register_module()
class ActWorldMomADv2AlignedPlanner(ActWorldMomADv2PlanningContext):
 def __init__(self,actworld=None,**kwargs):
  super().__init__(actworld=actworld,**kwargs)
  self.actworld.__class__=AlignedPlannerOutcomeJEPA
