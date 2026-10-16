"""Action-conditioned Outcome-JEPA for the frozen MomADv2 candidate set."""
import torch
from torch import nn
from torch.nn import functional as F


class RelativeOutcomeJEPA(nn.Module):
    def __init__(self, width=128, jepa_weight=0.2, score_scale=0.5):
        super().__init__()
        self.jepa_weight = float(jepa_weight)
        self.score_scale = float(score_scale)
        self.scene = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, width), nn.GELU())
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.action = nn.Sequential(nn.Linear(6, width), nn.GELU(), nn.Linear(width, width))
        self.transition = nn.GRUCell(width, width)
        self.projector = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
        self.cost = nn.Linear(width, 1)
        nn.init.zeros_(self.cost.weight)
        nn.init.zeros_(self.cost.bias)
        generator = torch.Generator().manual_seed(20261008)
        # Fixed outcome embedding: no trainable target encoder can collapse.
        self.register_buffer('outcome_projection', torch.randn(4, width // 2, generator=generator))
        weights = torch.zeros(12)
        for end in (2, 4, 6, 8, 10, 12):
            weights[:end] += 1.0 / (6 * end)
        self.register_buffer('horizon_weights', weights)

    def forward(self, tokens, candidates, base_logits, zero_actions=False):
        assert candidates.shape[1:] == (6, 12, 2)
        batch, modes, horizon, _ = candidates.shape
        scene = self.scene(tokens.detach().float())
        state = self.attention(self.query.expand(batch, -1, -1), scene, scene)[0][:, 0]
        state = state[:, None].expand(-1, modes, -1).reshape(batch * modes, -1)
        position = candidates.detach().float()
        delta = torch.diff(position, dim=2, prepend=torch.zeros_like(position[:, :, :1]))
        acceleration = torch.diff(delta, dim=2, prepend=torch.zeros_like(delta[:, :, :1]))
        actions = torch.cat([position / 30, delta / 5, acceleration / 5], dim=-1)
        if zero_actions:
            actions = torch.zeros_like(actions)
        rollout = []
        for t in range(horizon):
            action_t = self.action(actions[:, :, t].reshape(batch * modes, 6))
            context_t = self.attention(action_t.reshape(batch, modes, -1), scene, scene)[0].reshape(batch * modes, -1)
            state = self.transition(action_t + context_t, state)
            rollout.append(state.reshape(batch, modes, -1))
        latent = torch.stack(rollout, dim=2)
        predicted_l2 = F.softplus(self.cost(latent).squeeze(-1))
        cost = (predicted_l2 * self.horizon_weights).sum(-1)
        # Centering retains the original classification scale; initialized cost
        # is constant across candidates, hence initial ranking is unchanged.
        centered = cost - cost[:, :1]
        correction = (centered - centered.mean(1, keepdim=True)).clamp(-3, 3)
        logits = base_logits.detach().float() - self.score_scale * correction
        return dict(latent=latent, predicted_l2=predicted_l2, logits=logits,
                    cost=cost, base_logits=base_logits.detach().float())

    def loss(self, prediction, candidates, gt_delta, valid):
        gt = gt_delta.float().cumsum(-2)
        valid = valid.float().reshape(gt.shape[0], 12)
        error = candidates.detach().float() - gt[:, None]
        l2 = error.square().sum(-1).sqrt()
        delta_error = torch.diff(error, dim=2, prepend=torch.zeros_like(error[:, :, :1]))
        outcome = torch.cat([error / 10, l2[..., None] / 10,
                             delta_error.square().sum(-1).sqrt()[..., None] / 5], -1)
        # Predict relative outcomes against the baseline-selected action.
        base_index = prediction['base_logits'].argmax(-1)
        batch_rows = torch.arange(len(base_index), device=base_index.device)
        relative_outcome = outcome - outcome[batch_rows, base_index][:, None]
        with torch.no_grad():
            phase = relative_outcome.clamp(-5, 5) @ self.outcome_projection
            target = F.normalize(torch.cat([phase.sin(), phase.cos()], -1), dim=-1)
        mask = valid[:, None].expand_as(l2)
        z = F.normalize(self.projector(prediction['latent']), dim=-1)
        alignment = ((1 - (z * target).sum(-1)) * mask).sum() / mask.sum().clamp_min(1)
        regression = (F.smooth_l1_loss(prediction['predicted_l2'], l2.clamp_max(50),
                                     reduction='none') * mask).sum() / mask.sum().clamp_min(1)
        weights = valid * self.horizon_weights
        true_cost = (l2 * weights[:, None]).sum(-1) / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        # A candidate rejected by MomAD's collision veto cannot be rescued by
        # our bounded score correction. Do not teach the ranker to undo it.
        base = prediction['base_logits']
        eligible = base >= base.amax(-1, keepdim=True) - 100
        teacher = F.softmax((-true_cost / 0.5).masked_fill(~eligible, -1e4), dim=-1).detach()
        rank = -(teacher * F.log_softmax(prediction['logits'], dim=-1)).sum(-1)
        has_future = valid.sum(-1) > 0
        ranking = (rank * has_future).sum() / has_future.sum().clamp_min(1)
        true_difference = true_cost[:, :, None] - true_cost[:, None, :]
        predicted_difference = prediction['cost'][:, :, None] - prediction['cost'][:, None, :]
        pair_mask = eligible[:, :, None] & eligible[:, None, :] & has_future[:, None, None]
        pair_mask = pair_mask & ~torch.eye(6, device=base.device, dtype=torch.bool)[None]
        pairwise = (F.smooth_l1_loss(predicted_difference, true_difference.detach(), reduction='none') * pair_mask).sum() / pair_mask.sum().clamp_min(1)
        chosen = prediction['logits'].argmax(-1)
        rows = torch.arange(len(chosen), device=chosen.device)
        return dict(loss_aw_regression=0.5 * regression, loss_aw_ranking=0.5 * ranking,
                    loss_aw_pairwise=pairwise,
                    loss_aw_jepa=self.jepa_weight * alignment,
                    aw_selected_l2=true_cost[rows, chosen].mean().detach(),
                    aw_oracle_l2=true_cost.min(-1).values.mean().detach(),
                    aw_alignment=alignment.detach())

