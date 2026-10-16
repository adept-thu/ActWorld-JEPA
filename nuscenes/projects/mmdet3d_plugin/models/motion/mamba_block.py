import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS


@PLUGIN_LAYERS.register_module()
class PureTorchSelectiveSSMQueryEncoder(BaseModule):
    """
    Pure PyTorch selective SSM block for planning query modeling.

    This follows the key Mamba idea:
    input-dependent delta/B/C parameters + recurrent selective scan.

    Input:
        x: [B, L, C]
    Output:
        y: [B, L, C]
    """

    def __init__(
        self,
        d_model=256,
        d_state=16,
        expand=2,
        d_conv=4,
        num_layers=2,
        dropout=0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            PureTorchSelectiveSSMLayer(
                d_model=d_model,
                d_state=d_state,
                d_inner=self.d_inner,
                d_conv=d_conv,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # 外层使用 residual，不要在这里改变完整 query 的数值空间
        self.out_norm = nn.Identity()


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.out_norm(x)


class PureTorchSelectiveSSMLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_state=16,
        d_inner=512,
        d_conv=4,
        dropout=0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner

        self.norm = nn.LayerNorm(d_model)

        self.in_proj = nn.Linear(d_model, d_inner * 2)

        self.d_conv = d_conv

        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=0,
            groups=d_inner,
        )


        # produce input-dependent delta, B, C
        self.x_proj = nn.Linear(d_inner, 1 + 2 * d_state)

        # diagonal state transition A
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # skip connection D
        self.D = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

        # 外层还有 global gate 和 mode gate，因此这里可以稍大一些
        self.layer_scale = nn.Parameter(torch.ones(d_model) * 0.01)


        # out_proj 不要零初始化，否则 SSM 前期几乎没贡献
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.3)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)


    def forward(self, x):
        """
        x: [B, L, C]
        """
        orig_dtype = x.dtype

        with torch.cuda.amp.autocast(enabled=False):
            residual = x.float()
            x_f = x.float()
            x_f = torch.nan_to_num(x_f, nan=0.0, posinf=1e4, neginf=-1e4)
            x_f = x_f.clamp(min=-1e4, max=1e4)

            # fp32 LayerNorm
            x_f = F.layer_norm(
                x_f,
                normalized_shape=(self.d_model,),
                weight=self.norm.weight.float(),
                bias=self.norm.bias.float() if self.norm.bias is not None else None,
                eps=self.norm.eps,
            )

            # fp32 in_proj
            xz = F.linear(
                x_f,
                self.in_proj.weight.float(),
                self.in_proj.bias.float() if self.in_proj.bias is not None else None,
            )
            xz = torch.nan_to_num(xz, nan=0.0, posinf=1e4, neginf=-1e4)
            xz = xz.clamp(min=-1e4, max=1e4)

            x_inner, z = xz.chunk(2, dim=-1)

            # fp32 causal depthwise conv
            x_conv = x_inner.transpose(1, 2)  # [B, d_inner, L]
            x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
            x_conv = F.conv1d(
                x_conv,
                self.conv1d.weight.float(),
                self.conv1d.bias.float() if self.conv1d.bias is not None else None,
                stride=self.conv1d.stride,
                padding=0,
                dilation=self.conv1d.dilation,
                groups=self.conv1d.groups,
            )
            x_conv = x_conv.transpose(1, 2)
            x_conv = F.silu(x_conv)

            y = self.selective_scan(x_conv)

            z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
            z = z.clamp(min=-1e4, max=1e4)

            y = y * F.silu(z)

            # fp32 out_proj
            y = F.linear(
                y,
                self.out_proj.weight.float(),
                self.out_proj.bias.float() if self.out_proj.bias is not None else None,
            )
            y = self.dropout(y)

            y = residual + self.layer_scale.float()[None, None, :] * y
            y = torch.nan_to_num(y, nan=0.0, posinf=1e4, neginf=-1e4)
            y = y.clamp(min=-1e4, max=1e4)

        return y.to(orig_dtype)



    def selective_scan(self, x):
        orig_dtype = x.dtype

        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)
            x = x.clamp(min=-100.0, max=100.0)

            B, L, D = x.shape
            N = self.d_state

            params = F.linear(
                x,
                self.x_proj.weight.float(),
                self.x_proj.bias.float() if self.x_proj.bias is not None else None,
            )

            params = torch.nan_to_num(params, nan=0.0, posinf=20.0, neginf=-20.0)
            params = params.clamp(min=-20.0, max=20.0)

            delta = F.softplus(params[..., 0]).clamp(min=1e-4, max=0.3)

            B_param = torch.tanh(params[..., 1:1 + N])
            C_param = torch.tanh(params[..., 1 + N:1 + 2 * N])

            A_log = self.A_log.float().clamp(min=-6.0, max=2.0)
            A = -torch.exp(A_log)

            D_skip = self.D.float().clamp(min=-3.0, max=3.0)

            h = x.new_zeros(B, D, N)
            outputs = []

            for t in range(L):
                xt = x[:, t, :]
                dt = delta[:, t]
                Bt = B_param[:, t, :]
                Ct = C_param[:, t, :]

                exp_arg = dt[:, None, None] * A[None, :, :]
                exp_arg = exp_arg.clamp(min=-20.0, max=0.0)

                dA = torch.exp(exp_arg)
                dB_x = dt[:, None, None] * Bt[:, None, :] * xt[:, :, None]

                h = dA * h + dB_x
                h = torch.nan_to_num(h, nan=0.0, posinf=100.0, neginf=-100.0)
                h = h.clamp(min=-100.0, max=100.0)

                yt = torch.sum(h * Ct[:, None, :], dim=-1)
                yt = yt + D_skip[None, :] * xt

                yt = torch.nan_to_num(yt, nan=0.0, posinf=100.0, neginf=-100.0)
                yt = yt.clamp(min=-100.0, max=100.0)

                outputs.append(yt)

            y = torch.stack(outputs, dim=1)
            y = torch.nan_to_num(y, nan=0.0, posinf=100.0, neginf=-100.0)
            y = y.clamp(min=-100.0, max=100.0)

        return y.to(orig_dtype)





class PlanningMambaBlock(PureTorchSelectiveSSMQueryEncoder):
    def __init__(
        self,
        embed_dims=256,
        d_model=None,
        d_state=16,
        expand=2,
        d_conv=4,
        num_layers=2,
        dropout=0.0,
        init_cfg=None,
        **kwargs,
    ):
        if d_model is None:
            d_model = embed_dims

        super().__init__(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            num_layers=num_layers,
            dropout=dropout,
            init_cfg=init_cfg,
        )


def timestep_embedding(t, dim, max_period=10000):
    """
    t: [B]
    return: [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

    return emb


@PLUGIN_LAYERS.register_module()
class FlowMatchingPlanningHead(BaseModule):
    """
    Flow Matching head for ego planning trajectory.

    cond_query: [B, 1, 18, 256]
    x_t:        [B, 1, 18, T, 2]
    t:          [B]
    output:     [B, 1, 18, T, 2]
    """

    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        num_modes=18,
        num_layers=2,
        dropout=0.1,
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.num_modes = num_modes

        self.traj_encoder = nn.Linear(fut_ts * 2, embed_dims)

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.SiLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self.input_mlp = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.SiLU(),
            nn.LayerNorm(embed_dims),
            nn.Dropout(dropout),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=8,
            dim_feedforward=embed_dims * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.SiLU(),
            nn.Linear(embed_dims, fut_ts * 2),
        )
        # Flow head 最后一层零初始化，避免一开始输出过大导致梯度不稳定
        nn.init.zeros_(self.out_mlp[-1].weight)
        nn.init.zeros_(self.out_mlp[-1].bias)

    def forward(self, cond_query, x_t, t):
        """
        cond_query: [B, 1, M, C]
        x_t:        [B, 1, M, T, 2]
        t:          [B]
        """
        B, Q, M, C = cond_query.shape
        T = x_t.shape[-2]

        cond = cond_query.reshape(B * Q, M, C)
        x_flat = x_t.reshape(B * Q, M, T * 2)

        traj_feat = self.traj_encoder(x_flat)

        t = t.reshape(B * Q).to(device=cond.device, dtype=cond.dtype)
        t_feat = timestep_embedding(t, C)
        t_feat = self.time_mlp(t_feat)
        t_feat = t_feat.unsqueeze(1).expand(-1, M, -1)

        h = torch.cat([cond, traj_feat, t_feat], dim=-1)
        h = self.input_mlp(h)
        h = self.encoder(h)

        v = self.out_mlp(h)
        v = v.reshape(B, Q, M, T, 2)

        return v