"""Bounded trajectory refinement from action-conditioned JEPA rollout states."""
import torch
from torch import nn
from torch.nn import functional as F
from .actworld_relative import RelativeOutcomeJEPA

class RefineOutcomeJEPA(RelativeOutcomeJEPA):
    def __init__(self, width=128, jepa_weight=0.2, score_scale=0.1):
        super().__init__(width=width, jepa_weight=jepa_weight, score_scale=score_scale)
        self.refine = nn.Linear(width, 2)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def forward(self, tokens, candidates, base_logits, zero_actions=False):
        out = super().forward(tokens, candidates, base_logits, zero_actions)
        raw = self.refine(out['latent']).tanh()
        b, m, t, d = raw.shape
        smooth = F.avg_pool1d(F.pad(raw.reshape(b*m,t,d).transpose(1,2), (1,1), mode='replicate'), 3, stride=1).transpose(1,2).reshape(b,m,t,d)
        ramp = torch.arange(1,t+1,device=raw.device,dtype=raw.dtype)/t
        offset = 0.5 * smooth * ramp[None,None,:,None]
        out['refined_candidates'] = candidates.detach() + offset
        out['refinement'] = offset
        return out

    def loss(self, prediction, candidates, gt_delta, valid):
        losses = super().loss(prediction,candidates,gt_delta,valid)
        gt = gt_delta.float().cumsum(-2)
        base_index = prediction['base_logits'].argmax(-1)
        rows = torch.arange(len(base_index),device=base_index.device)
        refined = prediction['refined_candidates'][rows,base_index]
        weights = valid.float().reshape(-1,12)*self.horizon_weights
        error = ((refined-gt).square().sum(-1)+1e-6).sqrt()
        losses['loss_aw_refinement'] = (error*weights).sum()/weights.sum().clamp_min(1)
        losses['loss_aw_refine_regularization'] = 0.05*prediction['refinement'].square().mean()
        return losses
