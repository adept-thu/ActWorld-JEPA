import torch
import torch.nn as nn
import torch.nn.functional as F




class GraphWorldModel(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Ego plan_query 查询邻居
        self.plan_cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.plan_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim)
        )

        # Anchor / Ego 投影
        self.W_anchor = nn.Linear(embed_dim, embed_dim)
        self.W_ego = nn.Linear(embed_dim, embed_dim)

        # 动态节点重要性 (importance weights)
        self.node_importance = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # world_state → 条件生成器
        self.cond_proj = nn.Linear(embed_dim, embed_dim)
        self.cond_ln   = nn.LayerNorm(embed_dim)

    def forward(self, instance_feature, close_ids, motion_query, plan_query, ego_feat):
        """
        instance_feature: [B, total_nodes, C]
        close_ids:        [K] K=9 邻居 id
        motion_query:     [B, N, M, C]
        plan_query:       [B, 1, 18, C] ego 的 18 条多模态轨迹
        ego_feat:         [B, 1, C]
        """
        B, total_nodes, C = instance_feature.shape
        N, M = motion_query.shape[1], motion_query.shape[2]
        device = instance_feature.device

        # -------------------------------
        # Step1: 取邻居节点特征
        # -------------------------------
        close_ids = close_ids.to(device)
        neigh_nodes = instance_feature[:, close_ids, :]      # [B, 9, C]

        # -------------------------------
        # Step2: Ego plan_query 查询邻居
        # -------------------------------
        plan_q = plan_query.squeeze(1)  # [B, 18, C]

        attn_out_plan, _ = self.plan_cross_attn(
            plan_q,        # Query
            neigh_nodes,   # Key
            neigh_nodes    # Value
        )

        plan_out = self.plan_proj(attn_out_plan).unsqueeze(1)  # [B,1,18,C]

        # -------------------------------
        # Step3: 基础 world_state
        # -------------------------------
        world_state = instance_feature.clone()                    # [B, total_nodes, C]
        world_state[:, :N, :] += self.W_anchor(motion_query.mean(dim=2))
        world_state[:, N:, :] += self.W_ego(plan_out.mean(dim=2))

        # -------------------------------
        # Step4: 动态节点重要性（motion importance）
        # -------------------------------
        motion_flat = motion_query.reshape(B, N*M, C)             # [B, 5400, C]

        anchor_world_state = world_state[:, :N, :].unsqueeze(2).repeat(1,1,M,1)
        anchor_world_state = anchor_world_state.reshape(B, N*M, C)

        # ego 节点扩展，不用 repeat（避免显存浪费）
        ego_expand = ego_feat.expand(B, N*M, C)                  # [B, 5400, C]

        node_feat = torch.cat([motion_flat + anchor_world_state, ego_expand], dim=-1)
        motion_weight = self.node_importance(node_feat).reshape(B, N, M, 1)

        # -------------------------------
        # Step5: world_state → motion_query 的条件生成器
        # -------------------------------
        world_cond = self.cond_proj(world_state[:, :N, :]).unsqueeze(2)  # [B, N, 1, C]
        motion_query_cond = self.cond_ln(motion_query + world_cond)

        # 结合节点重要性
        motion_query_weighted = motion_query_cond * motion_weight

        return world_state, motion_query_weighted
