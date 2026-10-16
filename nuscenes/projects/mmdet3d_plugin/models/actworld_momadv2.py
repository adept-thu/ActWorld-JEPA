"""Frozen MomADv2 + training-only outcome targets + deployed JEPA reranker."""
import os
import torch
from mmdet.models import DETECTORS
from .sparsedrive import SparseDrive
from .actworld_outcome import OutcomeJEPA


@DETECTORS.register_module()
class ActWorldMomADv2(SparseDrive):
    def __init__(self, actworld=None, **kwargs):
        super().__init__(**kwargs)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.actworld = OutcomeJEPA(**(actworld or {}))

    def train(self, mode=True):
        # Keep dropout, normalization and temporal baseline behavior identical
        # to inference while exposing only the new branch to the optimizer.
        super().train(False)
        self.training = mode
        self.actworld.train(mode)
        return self

    def _candidate_inputs(self, model_outs, data):
        det, maps, motion, plan = model_outs
        b = plan['prediction'][-1].shape[0]
        full = plan['prediction'][-1].reshape(b, 3, 6, 12, 2).cumsum(-2)
        cls = plan['classification'][-1].reshape(b, 3, 6)
        # Retain MomAD's inference-time collision veto; use no future labels.
        rescored, _ = self.head.motion_plan_head.planning_decoder.select(det, motion, cls, full, data)
        rows = torch.arange(b, device=full.device)
        command = data['gt_ego_fut_cmd'].argmax(-1)
        tokens = []
        for out, count in [(det, 50), (maps, 10)]:
            confidence = out['classification'][-1].sigmoid().amax(-1)
            index = confidence.topk(min(count, confidence.shape[1]), dim=1).indices
            tokens.append(out['instance_feature'].gather(1, index[..., None].expand(-1, -1, 256)))
        return torch.cat(tokens, 1), full[rows, command], rescored[rows, command]

    def forward_train(self, img, **data):
        with torch.no_grad():
            features = self.extract_feat(img, metas=data)
            outputs = self.head(features, data)
            tokens, candidates, base = self._candidate_inputs(outputs, data)
        prediction = self.actworld(tokens, candidates, base)
        losses = self.actworld.loss(prediction, candidates, data['gt_ego_fut_trajs'], data['gt_ego_fut_masks'])
        losses['aw_switch_rate'] = (prediction['logits'].argmax(-1) != base.argmax(-1)).float().mean().detach()
        return losses

    def simple_test(self, img, **data):
        with torch.no_grad():
            features = self.extract_feat(img)
            outputs = self.head(features, data)
            results = self.head.post_process(outputs, data)
            if os.environ.get('ACTWORLD_DISABLE', '0') == '1' or self.actworld.score_scale == 0:
                return [dict(img_bbox=result) for result in results]
            tokens, candidates, base = self._candidate_inputs(outputs, data)
            prediction = self.actworld(tokens, candidates, base,
                                       zero_actions=os.environ.get('ACTWORLD_ZERO_ACTIONS', '0') == '1')
            chosen = prediction['logits'].argmax(-1)
            for i, result in enumerate(results):
                result['actworld_baseline_planning'] = result['final_planning'].clone()
                result['final_planning'] = candidates[i, chosen[i]].cpu()
                result['actworld_selected_mode'] = int(chosen[i])
                result['actworld_baseline_mode'] = int(base[i].argmax())
                result['actworld_predicted_cost'] = prediction['cost'][i].cpu()
            return [dict(img_bbox=result) for result in results]
