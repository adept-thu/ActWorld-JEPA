from typing import List, Optional, Tuple, Union
import warnings
import copy

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.motion.motion_blocks import *
from projects.mmdet3d_plugin.models.motion.mamba_block import PlanningMambaBlock


from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk
from nuscenes.nuscenes import NuScenes
from ....configs.mambaAD_small_stage2_6s import batch_size
from .next_token_prediction import NextTokenPredictor
@HEADS.register_module()
class MotionPlanningHeadmambaADNoFlow_6s(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=12,
        ego_fut_mode=6,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        mamba_model=None,
        use_ssm=True,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        use_rescore= True
        
    ):
        super(MotionPlanningHeadmambaADNoFlow_6s, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        self.batch_size =batch_size
        self.last_planning_classification=torch.zeros([batch_size, 1, 18])
        self.last_planning_prediction=torch.zeros([batch_size, 1, 18, 12, 2])
        self.last_final_planning_prediction=torch.zeros([batch_size, 12, 2])
        self.last_plan_query = torch.zeros([batch_size, 1, 18, 256])
        self.last_ego_cmd = torch.zeros([batch_size, 3])
        self.nusc = NuScenes(version='v1.0-trainval', dataroot="data/nuscenes/", verbose=True)
        self.use_rescore = use_rescore

        self.num_plan_cmd = 3


        self.ssm_res_gate = nn.Parameter(torch.tensor(-3.0))
        self.ssm_reg_mix = 0.75

        # SSM selected trajectory auxiliary loss
        self.ssm_selected_reg_weight = 0.01
        self.ssm_improve_weight = 0.08

        # 重点压 4~6s
        self.ssm_tail_reg_weight = 0.015
        self.ssm_final_reg_weight = 0.025
        self.ssm_tail_improve_weight = 0.04

        # 不对微小 residual 再做 LayerNorm 放大
        self.ssm_out_norm = nn.Identity()


        # mode-wise gate，让每个 mode 自己判断历史是否有用
        self.ssm_mode_gate = nn.Sequential(
            nn.Linear(embed_dims * 2, embed_dims),
            nn.SiLU(),
            nn.Linear(embed_dims, 1),
        )

        nn.init.zeros_(self.ssm_mode_gate[-1].weight)
        nn.init.constant_(self.ssm_mode_gate[-1].bias, -2.0)

        # ================= SSM trajectory residual head =================
        # 让 SSM 不只改 query，也直接预测一个小的轨迹 residual
        self.use_ssm_traj_residual = True

        # 输入: cur query + hist query + mamba output
        # [B, 6, 256 * 3] -> [B, 6, 12 * 2]
        self.ssm_traj_res_head = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.SiLU(),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, ego_fut_ts * 2),
        )

        # 最后一层零初始化：刚开始完全不影响原模型，训练后再慢慢学
        nn.init.zeros_(self.ssm_traj_res_head[-1].weight)
        nn.init.zeros_(self.ssm_traj_res_head[-1].bias)

        # 控制 trajectory residual 强度
        self.ssm_traj_gate = nn.Parameter(torch.tensor(-3.0))


        self.last_sample_tokens = [None for _ in range(batch_size)]
        self._ssm_debug_step = 0
        self._ssm_eval_debug_step = 0

        # ================= Multi-frame SSM memory =================
        # 真正使用多帧历史，而不是只用上一帧
        self.ssm_history_steps = 4
        self.ssm_memory = {}
        self.ssm_memory_max = 60000

        # ================= Long-horizon tail offset head =================
        # 专门修 4~6s，输出最终长时域 residual offset
        self.use_ssm_tail_offset = True

        self.ssm_tail_offset_head = nn.Sequential(
            nn.Linear(embed_dims * 3, embed_dims),
            nn.SiLU(),
            nn.LayerNorm(embed_dims),
            nn.Linear(embed_dims, 2),
        )

        nn.init.zeros_(self.ssm_tail_offset_head[-1].weight)
        nn.init.zeros_(self.ssm_tail_offset_head[-1].bias)

        # 初始不要太猛，但允许训练后打开
        self.ssm_tail_offset_gate = nn.Parameter(torch.tensor(-2.0))



        # ================= SSM residual scale control =================
        # ssm_update_gain：放大 SSM 的方向
        # ssm_max_update_ratio：限制最终修改不能超过 query norm 的多少比例
        # 先用这两个值做安全实验
        self.ssm_update_gain = 300
        self.ssm_max_update_ratio = 0.03


        # Flow 作为 trajectory residual corrector，初始小幅作用，避免炸




        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        
        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.history_mamba = build(mamba_model, PLUGIN_LAYERS)
        self.use_ssm = use_ssm
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)
        self.last_op = ""

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map
        #self.sa_atten_layer = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)

        self.refine_2th_layer = MotionPlanning2thRefinementModule(
            embed_dims=256,
            ego_fut_ts=12,
            ego_fut_mode=6
        )

        # 现在第一阶段不用 second refine，先冻结，避免 DDP unused parameter
        for p in self.refine_2th_layer.parameters():
            p.requires_grad = False



    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        
        for m in self.modules():
            #import pdb;pdb.set_trace()
            if hasattr(m, "init_weight"):
                m.init_weight()

    def get_motion_anchor(
        self, 
        classification, 
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar


    def rescore(
        self, 
        plan_cls,
        plan_reg, 
        motion_cls,
        motion_reg, 
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):
        
        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat
        
        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)
        
        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[0] += offset * torch.cos(ego_box[6])
        ego_box[1] += offset * torch.sin(ego_box[6])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls


    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )
    def sa_atten(self, plan_query, last_query):
        #import pdb;pdb.set_trace()
        plan_query = self.sa_atten_layer(torch.cat([plan_query,last_query],dim=-1).permute(0,3,1,2))
        return plan_query.permute(0,2,3,1)
    # def select_last_final_planning_prediction(self,planning_classification,planning_prediction,metas):
    #     #print("hhhhhhhhhhhhhhhhhhhhhh",len(planning_classification))
    #     #import pdb; pdb.set_trace()
    #     bs = planning_classification[0].shape[0]
    #     #bs = planning_classification_cmd.shape[0] #3
    #     planning_classification_cmd = planning_classification[-1].reshape(bs, 3, self.ego_fut_mode)#[3, 3, 6]
    #     planning_prediction_cmd = planning_prediction[-1].reshape(bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)#[3, 3, 6, 12, 2]
    #     bs_indices = torch.arange(bs, device=planning_classification_cmd.device) #tensor([0, 1, 2], device='cuda:0')
    #     gt_ego_fut_cmd = metas["gt_ego_fut_cmd"].argmax(dim=-1)#tensor([2, 2, 2], device='cuda:0')
    #     planning_classification_cmd_select = planning_classification_cmd[bs_indices, gt_ego_fut_cmd]#torch.Size([3, 6])
    #     planning_prediction_cmd_select = planning_prediction_cmd[bs_indices, gt_ego_fut_cmd]#torch.Size([3, 6, 12, 2])

    #     mode_idx = planning_classification_cmd_select.argmax(dim=-1)#tensor([5, 5, 0], device='cuda:0')
    #     last_final_planning_prediction = planning_prediction_cmd_select[bs_indices, mode_idx] #torch.Size([3, 12, 2])
    #     return last_final_planning_prediction
    def select_last_final_planning_prediction(
        self,
        planning_classification,
        planning_prediction,
        metas
    ):
        bs = planning_classification[0].shape[0]

        planning_classification_cmd = planning_classification[-1].reshape(
            bs, 3, self.ego_fut_mode
        )

        # 注意：这里不要 cumsum，统一在 euclidean_distance 里处理
        planning_prediction_cmd = planning_prediction[-1].reshape(
            bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2
        )

        bs_indices = torch.arange(
            bs,
            device=planning_classification_cmd.device
        )

        gt_ego_fut_cmd = metas["gt_ego_fut_cmd"].argmax(dim=-1)

        planning_classification_cmd_select = planning_classification_cmd[
            bs_indices,
            gt_ego_fut_cmd
        ]

        planning_prediction_cmd_select = planning_prediction_cmd[
            bs_indices,
            gt_ego_fut_cmd
        ]

        mode_idx = planning_classification_cmd_select.argmax(dim=-1)

        last_final_planning_prediction = planning_prediction_cmd_select[
            bs_indices,
            mode_idx
        ]

        return last_final_planning_prediction

    def select_currect_best_planning(self, last_final_planning_prediction, planning_prediction, current_ego_cmd, last_ego_cmd):
        # import pdb; pdb.set_trace()
        ego_mask=(current_ego_cmd.argmax(-1) == last_ego_cmd.argmax(-1))
        distances_euclidean = self.euclidean_distance(last_final_planning_prediction.unsqueeze(1).repeat(1,18,1,1), planning_prediction[0].squeeze(1))
        min_idx = torch.argmin(distances_euclidean, dim=-1).cpu().numpy().tolist()
        bs = planning_prediction[0].squeeze(1).shape[0]
        #import pdb;pdb.set_trace()
        select_currect_best_planning_prediction = torch.where(ego_mask.view(-1, 1, 1), planning_prediction[0].squeeze(1)[torch.arange(bs), min_idx], torch.full((12, 2), 999.9).repeat(bs, 1, 1).to(ego_mask.device))
        #import pdb;pdb.set_trace()
        return select_currect_best_planning_prediction, min_idx, ego_mask
    
    def select_former_best_query(self, plan_query, last_ego_cmd, planning_classification):
        last_ego_cmd = last_ego_cmd.repeat_interleave(6, dim=1)
        planning_classification = planning_classification.squeeze(1).sigmoid() * last_ego_cmd

        return planning_classification.argmax(-1)
       
    def euclidean_distance(self, TA, TB):
        TA = TA.cumsum(dim=-2)
        TB = TB.cumsum(dim=-2)
        distances = torch.sqrt(torch.sum((TA - TB)**2, dim=-1))
        distance_sums = distances.sum(dim=-1)
        return distance_sums
    def dtw_distance(self, TA, TB):
        TA = TA.cumsum(dim=-2)
        TB = TB.cumsum(dim=-2)
        TA_np = TA.detach().cpu().numpy()  # 将tensor转换为numpy数组
        TB_np = TB.detach().cpu().numpy()
        distance, _ = fastdtw(TA_np, TB_np, dist=euclidean)
        return distance


    def shifted_euclidean_distance(self, TA, TB):
        """
        TA/TB: delta traj, [..., T, 2]
        用上一帧的 t+1 对齐当前帧的 t，减少 temporal offset。
        """
        TA = TA.cumsum(dim=-2)
        TB = TB.cumsum(dim=-2)

        if TA.shape[-2] > 1:
            TA = TA[..., 1:, :]
            TB = TB[..., :-1, :]

        distances = torch.sqrt(torch.sum((TA - TB) ** 2, dim=-1) + 1e-6)
        return distances.sum(dim=-1)



    def fetch_ssm_history_from_memory(
        self,
        cur_tokens,
        cmd_idx,
        device,
        dtype,
    ):
        """
        根据 token 找前 K 帧历史。
        只使用 command 一致的历史。
        """
        B = len(cur_tokens)
        K = int(self.ssm_history_steps)
        C = int(self.embed_dims)
        T = int(self.ego_fut_ts)

        hist_cmd_traj = torch.zeros(
            B, K, self.ego_fut_mode, T, 2,
            device=device,
            dtype=dtype,
        )

        hist_q = torch.zeros(
            B, K, self.num_plan_cmd * self.ego_fut_mode, C,
            device=device,
            dtype=dtype,
        )

        hist_traj = torch.zeros(
            B, K, T, 2,
            device=device,
            dtype=dtype,
        )

        hist_valid = torch.zeros(
            B, K,
            device=device,
            dtype=torch.bool,
        )

        hist_age = torch.zeros(
            B, K,
            device=device,
            dtype=dtype,
        )

        cmd_idx_cpu = cmd_idx.detach().cpu().tolist()

        for b, token in enumerate(cur_tokens):
            prev_token = self.nusc.get("sample", token)["prev"]

            for k in range(K):
                if prev_token == "":
                    break

                mem = self.ssm_memory.get(prev_token, None)

                if mem is not None:
                    if int(mem["cmd_idx"]) == int(cmd_idx_cpu[b]):
                        hist_q[b, k] = mem["plan_query"].to(
                            device=device,
                            dtype=dtype,
                        )

                        hist_traj[b, k] = mem["final_traj"].to(
                            device=device,
                            dtype=dtype,
                        )

                        if mem.get("cmd_traj", None) is not None:
                            hist_cmd_traj[b, k] = mem["cmd_traj"].to(
                                device=device,
                                dtype=dtype,
                            )
                        else:
                            # 兼容旧 memory：没有 6-mode traj 时，用 final_traj 复制成 6 条
                            hist_cmd_traj[b, k] = mem["final_traj"].to(
                                device=device,
                                dtype=dtype,
                            )[None].expand(self.ego_fut_mode, -1, -1)

                        hist_valid[b, k] = True
                        hist_age[b, k] = float(k + 1)


                prev_token = self.nusc.get("sample", prev_token)["prev"]

        return hist_q, hist_traj, hist_cmd_traj, hist_valid, hist_age



    def update_ssm_memory(
        self,
        cur_tokens,
        plan_query,
        final_traj,
        ego_cmd,
        cmd_traj=None,
    ):
        cmd_traj_cpu = None
        if cmd_traj is not None:
            cmd_traj_cpu = cmd_traj.detach().float().cpu()
        """
        把当前帧写入 token memory。
        注意：存 CPU，避免 GPU 显存泄漏。
        """
        if plan_query is None or final_traj is None:
            return

        pq = plan_query.detach().float().cpu()

        # [B, 1, 18, C] -> [B, 18, C]
        if pq.dim() == 4:
            pq = pq[:, 0]

        traj = final_traj.detach().float().cpu()

        cmd_idx = ego_cmd.detach().argmax(dim=-1).cpu().tolist()

        for b, token in enumerate(cur_tokens):
            self.ssm_memory[token] = {
                "plan_query": pq[b].clone(),
                "final_traj": traj[b].clone(),
                "cmd_traj": (
                    cmd_traj_cpu[b].clone()
                    if cmd_traj_cpu is not None
                    else None
                ),
                "cmd_idx": int(cmd_idx[b]),
            }

        # 防止 memory 无限增长
        overflow = len(self.ssm_memory) - int(self.ssm_memory_max)
        if overflow > 0:
            old_keys = list(self.ssm_memory.keys())[:overflow]
            for key in old_keys:
                del self.ssm_memory[key]




    def apply_ssm_to_first_plan_query(
        self,
        plan_query,
        base_plan_cls,
        base_plan_reg,
        metas,
        device=None,
        dtype=None,
    ):
        device = plan_query.device if device is None else device
        dtype = plan_query.dtype if dtype is None else dtype

        bs_cur = plan_query.shape[0]
        cur_q = plan_query.squeeze(1)  # [B, 18, 256]

        def align_cache(x, bs, device, dtype=None):
            x = x.to(device=device)

            if dtype is not None and x.is_floating_point():
                x = x.to(dtype=dtype)

            if x.shape[0] > bs:
                x = x[:bs]
            elif x.shape[0] < bs:
                pad_shape = (bs - x.shape[0],) + tuple(x.shape[1:])
                pad = torch.zeros(pad_shape, device=device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)

            return x

        last_query = align_cache(
            self.last_plan_query,
            bs_cur,
            device,
            dtype,
        )

        last_ego_cmd = align_cache(
            self.last_ego_cmd,
            bs_cur,
            device,
            dtype,
        )

        last_final_traj = align_cache(
            self.last_final_planning_prediction,
            bs_cur,
            device,
            dtype,
        )

        # ================= token continuity check =================
        cur_tokens = []
        prev_tokens = []

        for b in range(bs_cur):
            token = metas["img_metas"][b]["token"]
            prev_token = self.nusc.get("sample", token)["prev"]

            cur_tokens.append(token)
            prev_tokens.append(prev_token)

        scene_start = torch.tensor(
            [p == "" for p in prev_tokens],
            device=device,
            dtype=torch.bool,
        )

        contiguous = torch.tensor(
            [
                self.last_sample_tokens[b] is not None
                and prev_tokens[b] == self.last_sample_tokens[b]
                for b in range(bs_cur)
            ],
            device=device,
            dtype=torch.bool,
        )

        current_ego_cmd = metas["gt_ego_fut_cmd"].to(
            device=device,
            dtype=dtype,
        )

        history_valid = last_query.abs().sum(dim=(1, 2, 3)) > 0

        same_cmd = (
            current_ego_cmd.argmax(dim=-1)
            == last_ego_cmd.argmax(dim=-1)
        )

        history_valid = (
            history_valid
            & same_cmd
            & (~scene_start)
            & contiguous
        )

        # ================= debug counter =================
        if self.training:
            self._ssm_debug_step += 1
            debug_step = self._ssm_debug_step
            debug_prefix = "[SSM TRAIN]"
        else:
            self._ssm_eval_debug_step += 1
            debug_step = self._ssm_eval_debug_step
            debug_prefix = "[SSM EVAL]"

        should_print = debug_step % 200 == 0

        if should_print:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                rank = 0
            else:
                rank = torch.distributed.get_rank()

            if rank == 0:
                print(
                    debug_prefix,
                    "history_valid_ratio=",
                    float(history_valid.float().mean().item()),
                    "same_cmd_ratio=",
                    float(same_cmd.float().mean().item()),
                    "contiguous_ratio=",
                    float(contiguous.float().mean().item()),
                    "scene_start_ratio=",
                    float(scene_start.float().mean().item()),
                    "raw_gate=",
                    float(torch.sigmoid(self.ssm_res_gate).item()),
                )

        # ================= DDP safety dummy =================
        ssm_dummy = cur_q.sum() * 0.0

        if self.use_ssm and self.history_mamba is not None:
            for p in self.history_mamba.parameters():
                if p.requires_grad:
                    ssm_dummy = ssm_dummy + p.float().sum() * 0.0

            for p in self.ssm_mode_gate.parameters():
                if p.requires_grad:
                    ssm_dummy = ssm_dummy + p.float().sum() * 0.0

            if self.ssm_res_gate.requires_grad:
                ssm_dummy = ssm_dummy + self.ssm_res_gate.float().sum() * 0.0

            if hasattr(self, "ssm_traj_res_head"):
                for p in self.ssm_traj_res_head.parameters():
                    if p.requires_grad:
                        ssm_dummy = ssm_dummy + p.float().sum() * 0.0

            if hasattr(self, "ssm_traj_gate"):
                if self.ssm_traj_gate.requires_grad:
                    ssm_dummy = ssm_dummy + self.ssm_traj_gate.float().sum() * 0.0

            if hasattr(self, "ssm_tail_offset_head"):
                for p in self.ssm_tail_offset_head.parameters():
                    if p.requires_grad:
                        ssm_dummy = ssm_dummy + p.float().sum() * 0.0

            if hasattr(self, "ssm_tail_offset_gate"):
                if self.ssm_tail_offset_gate.requires_grad:
                    ssm_dummy = ssm_dummy + self.ssm_tail_offset_gate.float().sum() * 0.0



        # 如果没开 SSM，直接返回原始 query
        if not self.use_ssm or self.history_mamba is None:
            enhanced_plan_query = plan_query
            return enhanced_plan_query, ssm_dummy, cur_tokens, None


        # ================= command 6-mode selection =================
        batch_idx = torch.arange(
            bs_cur,
            device=device,
        )

        M = self.ego_fut_mode
        num_cmd = self.num_plan_cmd
        K = int(self.ssm_history_steps)

        cmd_idx = current_ego_cmd.argmax(dim=-1).clamp(
            min=0,
            max=num_cmd - 1,
        )

        local_mode = torch.arange(M, device=device)

        cmd_mode_idx = (
            cmd_idx[:, None] * M
            + local_mode[None, :]
        )  # [B, 6]

        batch_idx = torch.arange(bs_cur, device=device)
        batch_idx_2d = batch_idx[:, None].expand(-1, M)

        cur_selected = cur_q[
            batch_idx_2d,
            cmd_mode_idx,
        ]  # [B, 6, 256]

        # ================= Multi-frame token memory =================
        hist_q_seq, hist_traj_seq, hist_cmd_traj_seq, hist_valid_seq, hist_age_seq = (
            self.fetch_ssm_history_from_memory(
                cur_tokens=cur_tokens,
                cmd_idx=cmd_idx,
                device=device,
                dtype=dtype,
            )
        )


        # 是否至少有一帧有效历史
        history_valid = hist_valid_seq.any(dim=1)  # [B]

        # ================= Cross-mode history alignment =================
        # 先取历史帧当前 command 下的 6 个 query:
        # hist_q_cmd_seq: [B, K, 6, C]
        hist_q_cmd_seq = hist_q_seq[
            batch_idx[:, None, None].expand(-1, K, M),
            torch.arange(K, device=device)[None, :, None].expand(bs_cur, -1, M),
            cmd_mode_idx[:, None, :].expand(-1, K, -1),
        ]

        # chronological: 远 -> 近
        hist_q_cmd_chrono = torch.flip(
            hist_q_cmd_seq,
            dims=[1],
        )  # [B, K, 6, C]

        hist_cmd_traj_chrono = torch.flip(
            hist_cmd_traj_seq,
            dims=[1],
        )  # [B, K, 6, T, 2]

        hist_valid_chrono = torch.flip(
            hist_valid_seq,
            dims=[1],
        )  # [B, K]

        hist_age_chrono = torch.flip(
            hist_age_seq,
            dims=[1],
        ).clamp_min(1.0)  # [B, K]


        # ================= history similarity gate =================
        with torch.no_grad():
            cur_pred_all = base_plan_reg.squeeze(1)  # [B, 18, T, 2]

            cur_selected_traj = cur_pred_all[
                batch_idx_2d,
                cmd_mode_idx,
            ]  # [B, 6, T, 2]

            T = cur_selected_traj.shape[-2]

            # 对每个当前 mode，和历史每一帧的 6 个 mode 都算距离
            # cur_traj_cross:  [B, cur6, K, hist6, T, 2]
            # hist_traj_cross: [B, cur6, K, hist6, T, 2]
            cur_traj_cross = cur_selected_traj[:, :, None, None, :, :].expand(
                -1,
                M,
                K,
                M,
                -1,
                -1,
            )

            hist_traj_cross = hist_cmd_traj_chrono[:, None, :, :, :, :].expand(
                -1,
                M,
                -1,
                -1,
                -1,
                -1,
            )

            cross_dist = self.shifted_euclidean_distance(
                hist_traj_cross.reshape(bs_cur * M * K * M, T, 2),
                cur_traj_cross.reshape(bs_cur * M * K * M, T, 2),
            ).reshape(bs_cur, M, K, M)  # [B, cur6, K, hist6]

            valid_cross = hist_valid_chrono[:, None, :, None].to(torch.bool)

            cross_dist = torch.where(
                valid_cross,
                cross_dist,
                torch.full_like(cross_dist, 1e6),
            )

            # 对每个当前 mode、每个历史帧，选最像的历史 mode
            hist_dist, best_hist_mode = cross_dist.min(dim=-1)  # [B, 6, K]

            # debug: 有多少时候不是 same-local-mode
            local_mode_for_debug = torch.arange(
                M,
                device=device,
            )[None, :, None].expand(bs_cur, -1, K)

            valid_for_debug = hist_valid_chrono[:, None, :].expand(
                -1,
                M,
                -1,
            ).float()  # [B, 6, K]

            cross_mode_change_rate = (
                ((best_hist_mode != local_mode_for_debug).float() * valid_for_debug).sum()
                / valid_for_debug.sum().clamp_min(1.0)
            )

            same_local_idx = local_mode_for_debug[..., None]  # [B, 6, K, 1]

            same_local_dist = torch.gather(
                cross_dist,
                dim=-1,
                index=same_local_idx,
            ).squeeze(-1)  # [B, 6, K]

            cross_mode_dist_gain = (
                ((same_local_dist - hist_dist) * valid_for_debug).sum()
                / valid_for_debug.sum().clamp_min(1.0)
            )



        # 根据 best_hist_mode 去拿对应历史 query
        # hist_q_for_all_cur: [B, cur6, K, hist6, C]
        hist_q_for_all_cur = hist_q_cmd_chrono[:, None, :, :, :].expand(
            -1,
            M,
            -1,
            -1,
            -1,
        )

        gather_idx = best_hist_mode[..., None, None].expand(
            -1,
            -1,
            -1,
            1,
            cur_q.shape[-1],
        )

        hist_selected_seq = torch.gather(
            hist_q_for_all_cur,
            dim=3,
            index=gather_idx,
        ).squeeze(3)  # [B, 6, K, C]

        valid_hist = hist_valid_chrono[:, None, :].to(dtype=dtype)  # [B, 1, K]

        # 越相似、越近的历史权重大
        age_decay = torch.exp(
            -hist_age_chrono[:, None, :].to(dtype=dtype) / 2.0
        )

        hist_score = torch.exp(-hist_dist / 80.0).to(dtype=dtype)
        hist_score = hist_score * age_decay * valid_hist

        hist_weight = hist_score / hist_score.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        hist_summary = (
            hist_selected_seq
            * hist_weight[..., None]
        ).sum(dim=2)  # [B, 6, C]

        min_hist_dist = torch.where(
            valid_hist.bool(),
            hist_dist,
            torch.full_like(hist_dist, 1e6),
        ).min(dim=-1).values

        min_hist_dist = torch.where(
            min_hist_dist > 1e5,
            torch.full_like(min_hist_dist, 60.0),
            min_hist_dist,
        )

        # ================= reliability gate =================
        valid_count = hist_valid_seq.float().sum(dim=1)  # [B]

        valid_count_gate = (
            valid_count / float(max(1, min(K, 2)))
        ).clamp(0.0, 1.0).view(bs_cur, 1, 1)  # [B, 1, 1]

        dist_gate = torch.sigmoid(
            (35.0 - min_hist_dist) / 8.0
        ).unsqueeze(-1).to(dtype=dtype)  # [B, 6, 1]

        traj_gate = dist_gate * valid_count_gate.to(dtype=dtype)
        traj_gate = traj_gate.clamp(0.0, 1.0)




        # ================= SSM input: [history K frames + current] =================
        mamba_seq = torch.cat(
            [
                hist_selected_seq,
                cur_selected[:, :, None, :],
            ],
            dim=2,
        )  # [B, 6, K + 1, 256]

        mamba_in = mamba_seq



        B_, M_, L_, C_ = mamba_in.shape

        mamba_in = mamba_in.reshape(
            B_ * M_,
            L_,
            C_,
        )

        mamba_in = torch.nan_to_num(
            mamba_in,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        mamba_out = self.history_mamba(
            mamba_in.float()
        ).to(dtype)

        mamba_out = mamba_out.reshape(
            B_,
            M_,
            L_,
            C_,
        )

        mamba_cur = mamba_out[:, :, -1, :]  # [B, 6, 256]


        raw_ssm_delta = mamba_cur - cur_selected

        raw_ssm_delta = torch.nan_to_num(
            raw_ssm_delta,
            nan=0.0,
            posinf=10.0,
            neginf=-10.0,
        )

        ssm_delta = torch.tanh(raw_ssm_delta)

        global_gate = (
            0.08
            + 0.12 * torch.sigmoid(self.ssm_res_gate)
        ).to(dtype=dtype)


        mode_gate = torch.sigmoid(
            self.ssm_mode_gate(
                torch.cat(
                    [cur_selected, hist_summary],
                    dim=-1,
                ).float()
            )
        ).to(dtype=dtype)


        valid_f = history_valid[:, None, None].to(dtype=dtype)

        raw_ssm_update = (
            valid_f
            * traj_gate
            * global_gate
            * mode_gate
            * ssm_delta
        )

        raw_ssm_update = raw_ssm_update * float(self.ssm_update_gain)

        query_norm = cur_selected.float().norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        update_norm = raw_ssm_update.float().norm(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        max_update_norm = query_norm * float(self.ssm_max_update_ratio)

        update_scale = torch.clamp(
            max_update_norm / update_norm,
            max=1.0,
        ).to(dtype=dtype)

        ssm_update = raw_ssm_update * update_scale

        enhanced_cur_q = cur_q.clone()

        enhanced_cur_q[
            batch_idx_2d,
            cmd_mode_idx,
        ] = cur_selected + ssm_update

        if should_print:
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                rank = 0
            else:
                rank = torch.distributed.get_rank()

            if rank == 0:
                with torch.no_grad():
                    raw_l2_ratio = (
                        raw_ssm_update.float().norm(dim=-1)
                        / cur_selected.float().norm(dim=-1).clamp_min(1e-6)
                    )

                    final_l2_ratio = (
                        ssm_update.float().norm(dim=-1)
                        / cur_selected.float().norm(dim=-1).clamp_min(1e-6)
                    )

                    query_diff = (
                        enhanced_cur_q.detach()
                        - cur_q.detach()
                    ).abs().mean()

                    print(
                        "[SSM CMD6 UPDATE]",
                        "query_diff=",
                        float(query_diff.item()),
                        "raw_l2_ratio_mean=",
                        float(raw_l2_ratio.mean().item()),
                        "raw_l2_ratio_max=",
                        float(raw_l2_ratio.max().item()),
                        "final_l2_ratio_mean=",
                        float(final_l2_ratio.mean().item()),
                        "final_l2_ratio_max=",
                        float(final_l2_ratio.max().item()),
                        "max_ratio=",
                        float(self.ssm_max_update_ratio),
                        "gain=",
                        float(self.ssm_update_gain),
                        "traj_gate_mean=",
                        float(traj_gate.mean().item()),
                        "mode_gate_mean=",
                        float(mode_gate.mean().item()),
                        "global_gate=",
                        float(global_gate.item()),
                        "reliability_gate_mean=",
                        float(traj_gate.mean().item()),
                        "reliability_gate_max=",
                        float(traj_gate.max().item()),
                        "cross_mode_change_rate=",
                        float(cross_mode_change_rate.item()),
                        "cross_mode_dist_gain=",
                        float(cross_mode_dist_gain.item()),
                    )

        enhanced_cur_q = torch.nan_to_num(
            enhanced_cur_q,
            nan=0.0,
            posinf=1e4,
            neginf=-1e4,
        )

        enhanced_plan_query = enhanced_cur_q.unsqueeze(1)

        ssm_info = {
            "traj_cond": torch.cat(
                [cur_selected, hist_summary, mamba_cur],
                dim=-1,
            ),  # [B, 6, 256 * 3]

            "cmd_mode_idx": cmd_mode_idx,
            "batch_idx_2d": batch_idx_2d,
            "history_valid": history_valid,
            "hist_valid_seq": hist_valid_seq,
            # 新增：给 trajectory residual / tail offset 也用同一个可靠性门控
            "reliability_gate": traj_gate.detach(),
        }


        return enhanced_plan_query, ssm_dummy, cur_tokens, ssm_info


    def apply_ssm_traj_residual_to_plan_reg(
        self,
        plan_reg,
        ssm_info,
    ):
        """
        plan_reg: [B, 1, 18, T, 2], delta trajectory
        ssm_info: apply_ssm_to_first_plan_query 返回的辅助信息

        作用：
        让 SSM 直接预测累计轨迹空间 residual，
        再转换成 delta residual 加回 plan_reg。
        """

        if (
            not getattr(self, "use_ssm_traj_residual", False)
            or ssm_info is None
        ):
            return plan_reg



        B = plan_reg.shape[0]
        T = plan_reg.shape[-2]
        M = self.ego_fut_mode

        cond = ssm_info["traj_cond"].float()  # [B, 6, 768]

        # [B, 6, T * 2] -> [B, 6, T, 2]
        res_abs = self.ssm_traj_res_head(cond).view(
            B,
            M,
            T,
            2,
        )

        # 限制 residual，避免一开始把轨迹拉飞
        res_abs = torch.tanh(res_abs)

        if T == 12:
            # 0~2s 小修，2~4s 中等，4~6s 大一点
            abs_limit = res_abs.new_tensor(
                [0.15] * 4 + [0.35] * 4 + [0.70] * 4
            ).view(1, 1, T, 1)
        else:
            t1 = T // 3
            t2 = 2 * T // 3

            abs_limit = res_abs.new_ones(T)
            abs_limit[:t1] = 0.15
            abs_limit[t1:t2] = 0.35
            abs_limit[t2:] = 0.70
            abs_limit = abs_limit.view(1, 1, T, 1)

        # gate 初始大约 0.34，不会太小，也不会太猛
        gate = (
            0.10 + 0.50 * torch.sigmoid(self.ssm_traj_gate.float())
        ).to(dtype=res_abs.dtype)

        valid = ssm_info["history_valid"].to(
            device=res_abs.device,
            dtype=res_abs.dtype,
        ).view(B, 1, 1, 1)

        reliability_gate = ssm_info.get("reliability_gate", None)

        if reliability_gate is not None:
            reliability_gate = reliability_gate.to(
                device=res_abs.device,
                dtype=res_abs.dtype,
            ).view(B, M, 1, 1)
        else:
            reliability_gate = res_abs.new_ones(B, M, 1, 1)

        valid_mode = valid * reliability_gate

        res_abs = res_abs * abs_limit * gate * valid_mode



        # ================= Long-horizon tail offset =================
        # 专门修 4~6s，不明显影响 0~2s
        if getattr(self, "use_ssm_tail_offset", False):
            tail_offset = self.ssm_tail_offset_head(cond).view(
                B,
                M,
                2,
            )

            # 单方向最大允许 1.5m，靠 gate 控制实际幅度
            tail_offset = torch.tanh(tail_offset) * 1.5

            tail_gate = (
                0.05
                + 0.95 * torch.sigmoid(self.ssm_tail_offset_gate.float())
            ).to(dtype=res_abs.dtype)

            # 0~2s 不动，2~4s 小幅，4~6s 逐渐增强
            ramp_1d = res_abs.new_zeros(T)

            if T == 12:
                ramp_1d[4:] = torch.linspace(
                    0.15,
                    1.0,
                    T - 4,
                    device=res_abs.device,
                    dtype=res_abs.dtype,
                )
            else:
                start = T // 3
                ramp_1d[start:] = torch.linspace(
                    0.15,
                    1.0,
                    T - start,
                    device=res_abs.device,
                    dtype=res_abs.dtype,
                )

            ramp_1d = ramp_1d.pow(2)
            ramp = ramp_1d.view(1, 1, T, 1)

            tail_res_abs = (
                ramp
                * tail_offset[:, :, None, :]
                * tail_gate
                * valid_mode
            )

            res_abs = res_abs + tail_res_abs


        if (
            (self.training and self._ssm_debug_step % 200 == 0)
            or ((not self.training) and self._ssm_eval_debug_step % 200 == 0)
        ):
            if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
            ):
                rank = 0
            else:
                rank = torch.distributed.get_rank()

            if rank == 0:
                with torch.no_grad():
                    print(
                        "[SSM TRAJ RES]",
                        "res_abs_mean=",
                        float(res_abs.abs().mean().item()),
                        "res_abs_max=",
                        float(res_abs.abs().max().item()),
                        "gate=",
                        float(gate.item()),
                        "valid_ratio=",
                        float(ssm_info["history_valid"].float().mean().item()),
                    )


        # 转成 delta residual
        # 因为 plan_reg 是 delta，decoder 后面会 cumsum
        res_delta_first = res_abs[:, :, :1, :]
        res_delta_rest = res_abs[:, :, 1:, :] - res_abs[:, :, :-1, :]
        res_delta = torch.cat(
            [res_delta_first, res_delta_rest],
            dim=2,
        )  # [B, 6, T, 2]

        reg_cur = plan_reg.squeeze(1).clone()  # [B, 18, T, 2]

        batch_idx_2d = ssm_info["batch_idx_2d"]
        cmd_mode_idx = ssm_info["cmd_mode_idx"]

        reg_cur[
            batch_idx_2d,
            cmd_mode_idx,
        ] = reg_cur[
            batch_idx_2d,
            cmd_mode_idx,
        ] + res_delta.to(dtype=reg_cur.dtype)

        return reg_cur.unsqueeze(1)







    def forward(
        self, 
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):   

        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        #det_output包括 ['clas sification', 'prediction', 'quality', 'instance_feature', 'anchor_embed', 'instance_id']
        #det_output['classification'][0].shape  torch.Size([6, 900, 10]) (cam,anchor,class)
        #det_output['prediction'][0].shape  torch.Size([6, 900, 11]) (cam,anchor,11) 11:{x, y, z, ln w, ln h, ln l, sin yaw, cos yaw, vx, vy, vz}
        #det_output['quality'][0].shape  torch.Size([6, 900, 2]) (cam,anchor,2) 2:centerness,yawness
        #det_output['instance_feature'][0].shape  torch.Size([6, 900, 256]) (cam,anchor,dim) 
        #det_output['anchor_embed'][0].shape  torch.Size([6, 900, 256]) (cam,anchor,dim) 
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )
        #instance_feature_selected.shape  torch.Size([6, 50, 256])
        map_instance_feature = map_output["instance_feature"]
        #map_output包括dict_keys(['classification', 'prediction', 'quality', 'instance_feature', 'anchor_embed'])
        #map_output[classification] torch.Size([6, 100, 3])
        #map_output[prediction] torch.Size([6, 100, 40])
        #map_output[quality] [None, None, None, None, None, None]
        #map_output[instance_feature] torch.Size([6, 100, 256])
        #map_output[anchor_embed] torch.Size([6, 100, 256])

        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )
       
        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,#torch.Size([6, 1, 256])
            ego_anchor,#[6, 1, 11]
            temp_instance_feature,#torch.Size([6, 901, 1, 256])
            temp_anchor,#torch.Size([6, 901, 1, 11])
            temp_mask,#torch.Size([6, 901, 1])
        ) = self.instance_queue.get(
            det_output,#dict_keys(['classification', 'prediction', 'quality', 'instance_feature', 'anchor_embed', 'instance_id'])
            feature_maps,#torch.Size([6, 89760, 256]) torch.Size([6, 4, 2]) torch.Size([6, 4])
            metas,#dict_keys(['img_metas', 'timestamp', 'projection_mat', 'image_wh', 'gt_depth', 'focal', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status'])
            bs,#6
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)#torch.Size([6, 1, 256])
        temp_anchor_embed = anchor_encoder(temp_anchor)#torch.Size([6, 901, 1, 256])
        temp_instance_feature = temp_instance_feature.flatten(0, 1)#torch.Size([5406, 1, 256])
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)#torch.Size([6, 901, 1, 256])
        temp_mask = temp_mask.flatten(0, 1)#torch.Size([6, 901, 1])

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors) #motion_anchor torch.Size([6, 900, 6, 12, 2]) #torch.Size([6, 900, 10])  torch.Size([6, 900, 11])
        plan_anchor = torch.tile( #torch.Size([6, 3, 6, 12, 2])
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :])) #torch.Size([6, 900, 6, 256])
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :]) #torch.Size([6, 3, 6, 256])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)#torch.Size([6, 1, 18, 256])

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1) #torch.Size([6, 50, 256]) torch.Size([6, 50, 256]) torch.Size([6, 1, 256])
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1) #torch.Size([6, 50, 256]) torch.Size([6, 50, 256]) torch.Size([6, 1, 256])

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1) #torch.Size([6, 900, 256]) torch.Size([6, 900, 256]) torch.Size([6, 1, 256])
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)#torch.Size([6, 900, 256]) torch.Size([6, 900, 256]) torch.Size([6, 1, 256])

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_prediction = []
        planning_status = []
        planning_classification_refined = []
        planning_prediction_refined = []
        planning_status_refined = []

        base_plan_cls_for_loss = None
        base_plan_reg_for_loss = None

        # ================= memory clean source =================
        # SSM memory 里存干净的 baseline 信息，避免 SSM 自反馈累积漂移
        raw_plan_query_for_memory = None
        base_final_traj_for_memory = None
        base_cmd_traj_for_memory = None

        for i, op in enumerate(self.operation_order):
            #import pdb;pdb.set_trace()
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                #self.last_op = "temp_gnn"
                instance_feature = self.graph_model(#[5406, 1, 256])
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,#torch.Size([5406, 1, 256])
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),#torch.Size([5406, 1, 256])
                    key_pos=temp_anchor_embed,#torch.Size([5406, 1, 256])
                    key_padding_mask=temp_mask,#torch.Size([5406, 1])
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)#torch.Size([6, 901, 256])
            elif op == "gnn":
                #self.last_op = "gnn"
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                #self.last_op = "norm"
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                #self.last_op = "cross_gnn"
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            # elif op == "refine":
            #     motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
            #     plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)
            #     #plan_query= self.sa_atten(plan_query, self.last_plan_query.to(plan_query.device))
            #     # import pdb;pdb.set_trace()
                
            #     (
            #         motion_cls,
            #         motion_reg,
            #         plan_cls,#torch.Size([6, 1, 18])
            #         plan_reg,#torch.Size([6, 1, 18, 12, 2])
            #         plan_status,
            #     ) = self.layers[i](
            #         motion_query, #([6, 900, 6, 256]
            #         plan_query, #6, 1, 18, 256]
            #         instance_feature[:, num_anchor:],
            #         anchor_embed[:, num_anchor:],
            #     )
                
            #     motion_classification.append(motion_cls)
            #     motion_prediction.append(motion_reg)
            #     planning_classification.append(plan_cls)
            #     planning_prediction.append(plan_reg)
            #     planning_status.append(plan_status)


            elif op == "refine":
                motion_query = motion_mode_query + (
                    instance_feature + anchor_embed
                )[:, :num_anchor].unsqueeze(2)

                plan_query = plan_mode_query + (
                    instance_feature + anchor_embed
                )[:, num_anchor:].unsqueeze(2)

                # 这里的 plan_query 是没有经过 SSM 的 raw query
                # 用它写入 ssm_memory，避免把增强后的 query 反复喂回历史
                raw_plan_query_for_memory = plan_query.detach()

                if self.use_ssm and self.history_mamba is not None:
                    # 1. 先跑一次原始 MomAD head，只用于拿 base prediction 做匹配
                    with torch.no_grad():
                        (
                            motion_cls_base,
                            motion_reg_base,
                            plan_cls_base,
                            plan_reg_base,
                            plan_status_base,
                        ) = self.layers[i](
                            motion_query,
                            plan_query,
                            instance_feature[:, num_anchor:],
                            anchor_embed[:, num_anchor:],
                        )
                        # 用 base head 自己的分类，选出 base 最终轨迹
                        # 注意这里存的是 delta trajectory，不做 cumsum，保持和 update_ssm_memory / euclidean_distance 一致
                        base_final_traj_for_memory = self.select_last_final_planning_prediction(
                            [plan_cls_base.detach()],
                            [plan_reg_base.detach()],
                            metas,
                        ).detach()

                        B_mem = plan_reg_base.shape[0]
                        batch_idx_mem = torch.arange(
                            B_mem,
                            device=plan_reg_base.device,
                        )

                        cmd_idx_mem = metas["gt_ego_fut_cmd"].to(
                            plan_reg_base.device
                        ).argmax(dim=-1)

                        base_cmd_traj_for_memory = plan_reg_base.detach().squeeze(1).reshape(
                            B_mem,
                            3,
                            self.ego_fut_mode,
                            self.ego_fut_ts,
                            2,
                        )[
                            batch_idx_mem,
                            cmd_idx_mem,
                        ]  # [B, 6, T, 2]


                    # 2. 用 base prediction 做 SSM 增强
                    enhanced_plan_query, ssm_dummy, cur_tokens, ssm_info = self.apply_ssm_to_first_plan_query(
                        plan_query=plan_query,
                        base_plan_cls=plan_cls_base,
                        base_plan_reg=plan_reg_base,
                        metas=metas,
                    )


                    # 3. 增强后的 query 再送回原始 MomAD head
                    (
                        motion_cls,
                        motion_reg,
                        plan_cls,
                        plan_reg,
                        plan_status,
                    ) = self.layers[i](
                        motion_query,
                        enhanced_plan_query,
                        instance_feature[:, num_anchor:],
                        anchor_embed[:, num_anchor:],
                    )

                    # eval 时锁住原始 MomAD 的 classification
                    # 只让 SSM 改 regression，避免扰乱 mode selection
                    plan_cls = plan_cls_base.detach()
                    plan_status = plan_status_base.detach()
                    # 输出层 residual mix：控制 SSM 对最终轨迹的影响强度
                    plan_reg = (
                        plan_reg_base.detach()
                        + float(self.ssm_reg_mix) * (plan_reg - plan_reg_base.detach())
                    )

                    # SSM 直接在 trajectory space 做 residual correction
                    plan_reg= self.apply_ssm_traj_residual_to_plan_reg(
                        plan_reg,
                        ssm_info,
                    )


                    base_plan_cls_for_loss = plan_cls_base.detach()
                    base_plan_reg_for_loss = plan_reg_base.detach()


                    # ================= eval debug: SSM 是否真的改变最终被选中的轨迹 =================
                    if (
                        not self.training
                        and hasattr(self, "_ssm_eval_debug_step")
                        and self._ssm_eval_debug_step % 200 == 0
                    ):
                        if (
                            not torch.distributed.is_available()
                            or not torch.distributed.is_initialized()
                        ):
                            rank = 0
                        else:
                            rank = torch.distributed.get_rank()

                        if rank == 0:
                            with torch.no_grad():
                                B = plan_reg.shape[0]
                                batch_idx_dbg = torch.arange(
                                    B,
                                    device=plan_reg.device,
                                )

                                cmd_idx_dbg = metas["gt_ego_fut_cmd"].to(
                                    plan_reg.device
                                ).argmax(dim=-1)

                                cls_base_cmd = plan_cls_base.squeeze(1).reshape(
                                    B,
                                    3,
                                    self.ego_fut_mode,
                                )[batch_idx_dbg, cmd_idx_dbg]

                                cls_ssm_cmd = plan_cls.squeeze(1).reshape(
                                    B,
                                    3,
                                    self.ego_fut_mode,
                                )[batch_idx_dbg, cmd_idx_dbg]

                                idx_base = cls_base_cmd.argmax(dim=-1)
                                idx_ssm = cls_ssm_cmd.argmax(dim=-1)

                                reg_base_cmd = plan_reg_base.squeeze(1).reshape(
                                    B,
                                    3,
                                    self.ego_fut_mode,
                                    self.ego_fut_ts,
                                    2,
                                )[batch_idx_dbg, cmd_idx_dbg]

                                reg_ssm_cmd = plan_reg.squeeze(1).reshape(
                                    B,
                                    3,
                                    self.ego_fut_mode,
                                    self.ego_fut_ts,
                                    2,
                                )[batch_idx_dbg, cmd_idx_dbg]

                                base_selected = reg_base_cmd[
                                    batch_idx_dbg,
                                    idx_base,
                                ]

                                ssm_selected = reg_ssm_cmd[
                                    batch_idx_dbg,
                                    idx_ssm,
                                ]

                                ssm_same_base_mode = reg_ssm_cmd[
                                    batch_idx_dbg,
                                    idx_base,
                                ]

                                selected_same_mode_diff = (
                                    ssm_same_base_mode.cumsum(dim=-2)
                                    - base_selected.cumsum(dim=-2)
                                ).norm(dim=-1).mean()

                                selected_actual_diff = (
                                    ssm_selected.cumsum(dim=-2)
                                    - base_selected.cumsum(dim=-2)
                                ).norm(dim=-1).mean()

                                reg_diff_all = (
                                    plan_reg.detach().cumsum(dim=-2)
                                    - plan_reg_base.detach().cumsum(dim=-2)
                                ).norm(dim=-1).mean()

                                cls_diff = (
                                    plan_cls.detach()
                                    - plan_cls_base.detach()
                                ).abs().mean()

                                query_diff = (
                                    enhanced_plan_query.detach()
                                    - plan_query.detach()
                                ).abs().mean()

                                mode_change_rate = (
                                    idx_base != idx_ssm
                                ).float().mean()

                                print(
                                    "[SSM FINAL EFFECT]",
                                    "query_diff=",
                                    float(query_diff.item()),
                                    "cls_diff=",
                                    float(cls_diff.item()),
                                    "reg_diff_all=",
                                    float(reg_diff_all.item()),
                                    "selected_same_mode_diff=",
                                    float(selected_same_mode_diff.item()),
                                    "selected_actual_diff=",
                                    float(selected_actual_diff.item()),
                                    "mode_change_rate=",
                                    float(mode_change_rate.item()),
                                )

                    final_plan_query_for_cache = enhanced_plan_query


                else:
                    (
                        motion_cls,
                        motion_reg,
                        plan_cls,
                        plan_reg,
                        plan_status,
                    ) = self.layers[i](
                        motion_query,
                        plan_query,
                        instance_feature[:, num_anchor:],
                        anchor_embed[:, num_anchor:],
                    )

                    ssm_dummy = plan_query.sum() * 0.0
                    final_plan_query_for_cache = plan_query
                    enhanced_plan_query = plan_query
                    ssm_info = None

                    cur_tokens = [
                        metas["img_metas"][b]["token"]
                        for b in range(plan_query.shape[0])
                    ]

                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)


        
        self.instance_queue.cache_motion(
            instance_feature[:, :num_anchor],
            det_output,
            metas,
        )

        self.instance_queue.cache_planning(
            instance_feature[:, num_anchor:],
            plan_status,
        )

        motion_output = {
            "classification": motion_classification,
            "prediction": motion_prediction,
            "period": self.instance_queue.period,
            "anchor_queue": self.instance_queue.anchor_queue,
        }

        # ================= cache first predict only =================
        last_final_planning_prediction = self.select_last_final_planning_prediction(
            planning_classification,
            planning_prediction,
            metas,
        ).detach()

        self.last_planning_classification = planning_classification[-1].detach().clone()
        self.last_planning_prediction = planning_prediction[-1].detach().clone()
        self.last_plan_query = final_plan_query_for_cache.detach().clone()
        self.last_final_planning_prediction = last_final_planning_prediction.detach().clone()
        self.last_ego_cmd = metas["gt_ego_fut_cmd"].detach().clone()

        # ================= update token-keyed SSM memory =================
        # memory 用 raw query + base trajectory，避免 SSM 输出进入下一帧后形成自反馈
        memory_plan_query = (
            raw_plan_query_for_memory
            if raw_plan_query_for_memory is not None
            else final_plan_query_for_cache
        )

        memory_final_traj = (
            base_final_traj_for_memory
            if base_final_traj_for_memory is not None
            else last_final_planning_prediction
        )

        self.update_ssm_memory(
            cur_tokens=cur_tokens,
            plan_query=memory_plan_query,
            final_traj=memory_final_traj,
            ego_cmd=metas["gt_ego_fut_cmd"],
            cmd_traj=base_cmd_traj_for_memory,
        )


        bs_cur = final_plan_query_for_cache.shape[0]

        for b in range(bs_cur):
            self.last_sample_tokens[b] = cur_tokens[b]

        for b in range(bs_cur, len(self.last_sample_tokens)):
            self.last_sample_tokens[b] = None

        planning_output = {
            "classification": planning_classification,
            "prediction": planning_prediction,
            "status": planning_status,

            "ssm_dummy": ssm_dummy,
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }

        if base_plan_cls_for_loss is not None and base_plan_reg_for_loss is not None:
            planning_output.update({
                "base_classification": [base_plan_cls_for_loss],
                "base_prediction": [base_plan_reg_for_loss],
            })


        # ================= Flow matching branch: 保留，但只有开关打开才运行 =================


        return motion_output, planning_output





    @force_fp32(apply_to=("model_outs",))
    def loss_ssm_selected_reg(self, model_outs, data):
        if "base_classification" not in model_outs:
            return {}

        cls_base = model_outs["base_classification"][-1].detach().float().squeeze(1)
        reg_base = model_outs["base_prediction"][-1].detach().float().squeeze(1)
        reg_ssm = model_outs["prediction"][-1].float().squeeze(1)

        device = reg_ssm.device
        B = reg_ssm.shape[0]
        batch_idx = torch.arange(B, device=device)

        cmd_idx = data["gt_ego_fut_cmd"].to(device=device).argmax(dim=-1)

        local_mode = torch.arange(self.ego_fut_mode, device=device)
        cmd_mode_idx = (
            cmd_idx[:, None] * self.ego_fut_mode
            + local_mode[None, :]
        )  # [B, 6]

        cls_cmd = torch.gather(
            cls_base,
            dim=1,
            index=cmd_mode_idx,
        )  # [B, 6]

        best_local_idx = cls_cmd.argmax(dim=-1)

        selected_mode_idx = cmd_mode_idx[
            batch_idx,
            best_local_idx,
        ]  # [B]

        reg_ssm_sel = reg_ssm[
            batch_idx,
            selected_mode_idx,
        ]  # [B, T, 2]

        reg_base_sel = reg_base[
            batch_idx,
            selected_mode_idx,
        ]  # [B, T, 2]

        gt = data["gt_ego_fut_trajs"].to(device=device).float()

        if gt.dim() == 3:
            # [B, T, 2]
            pass
        elif gt.dim() == 4:
            # 兼容 [B, 1, T, 2] 或 [B, 6, T, 2]
            if gt.shape[1] == self.ego_fut_mode:
                gt = gt[batch_idx, best_local_idx]
            else:
                gt = gt[:, 0]
        else:
            raise ValueError(f"Unexpected gt_ego_fut_trajs shape: {gt.shape}")

        mask = data["gt_ego_fut_masks"].to(device=device).float()

        if mask.dim() == 2:
            # [B, T]
            pass
        elif mask.dim() == 3:
            # [B, T, 2] 或 [B, 6, T]
            if mask.shape[1] == self.ego_fut_mode and mask.shape[2] == self.ego_fut_ts:
                mask = mask[batch_idx, best_local_idx]
            else:
                mask = mask[..., 0]
        elif mask.dim() == 4:
            # [B, 6, T, 2]
            mask = mask[batch_idx, best_local_idx]
            if mask.shape[-1] == 2:
                mask = mask[..., 0]
        else:
            raise ValueError(f"Unexpected gt_ego_fut_masks shape: {mask.shape}")

        valid = mask.clamp(min=0.0, max=1.0)

        reg_ssm_abs = reg_ssm_sel.cumsum(dim=-2)
        reg_base_abs = reg_base_sel.cumsum(dim=-2)
        gt_abs = gt.cumsum(dim=-2)

        ssm_dist = torch.sqrt(
            ((reg_ssm_abs - gt_abs) ** 2).sum(dim=-1) + 1e-6
        )

        base_dist = torch.sqrt(
            ((reg_base_abs - gt_abs) ** 2).sum(dim=-1) + 1e-6
        ).detach()


        # ================= horizon-aware weight =================
        # 12 steps = 6s，每个点 0.5s
        # 前 0~2s 少一点，中间正常，4~6s 加强
        T = valid.shape[-1]

        time_weight = valid.new_ones(T)

        if T == 12:
            time_weight[:4] = 0.5      # 0~2s
            time_weight[4:8] = 1.0     # 2~4s
            time_weight[8:] = 2.5      # 4~6s，重点优化
        else:
            t1 = T // 3
            t2 = 2 * T // 3
            time_weight[:t1] = 0.5
            time_weight[t1:t2] = 1.0
            time_weight[t2:] = 2.5

        valid_w = valid * time_weight[None, :]

        selected_reg_loss = (
            ssm_dist * valid_w
        ).sum() / valid_w.sum().clamp_min(1.0)

        improve_loss = (
            torch.relu(ssm_dist - base_dist) * valid_w
        ).sum() / valid_w.sum().clamp_min(1.0)

        # ================= 专门优化 4~6s =================
        tail_valid = valid[:, -4:]          # 最后 4 个点，对应 4~6s
        tail_dist = ssm_dist[:, -4:]
        tail_base_dist = base_dist[:, -4:]

        tail_reg_loss = (
            tail_dist * tail_valid
        ).sum() / tail_valid.sum().clamp_min(1.0)

        tail_improve_loss = (
            torch.relu(tail_dist - tail_base_dist) * tail_valid
        ).sum() / tail_valid.sum().clamp_min(1.0)

        # ================= 专门优化最终 6s 点 =================
        final_valid = valid[:, -1]

        final_reg_loss = (
            ssm_dist[:, -1] * final_valid
        ).sum() / final_valid.sum().clamp_min(1.0)

        return {
            "planning_loss_ssm_selected_reg": selected_reg_loss * self.ssm_selected_reg_weight,
            "planning_loss_ssm_improve": improve_loss * self.ssm_improve_weight,

            "planning_loss_ssm_tail_reg": tail_reg_loss * self.ssm_tail_reg_weight,
            "planning_loss_ssm_tail_improve": tail_improve_loss * self.ssm_tail_improve_weight,
            "planning_loss_ssm_final_reg": final_reg_loss * self.ssm_final_reg_weight,
        }

    # def loss(
    #     self,
    #     motion_model_outs,
    #     planning_model_outs,
    #     data,
    #     motion_loss_cache
    # ):
    #     loss = {}

    #     motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
    #     loss.update(motion_loss)

    #     planning_loss = self.loss_planning(planning_model_outs, data)
    #     loss.update(planning_loss)

    #     planning_loss_refined = self.loss_planning_refined(planning_model_outs, data)
    #     loss.update(planning_loss_refined)

    #     if self.use_flow_aux_loss:
    #         planning_loss_flow_decode = self.loss_planning_flow_decode(planning_model_outs, data)
    #         loss.update(planning_loss_flow_decode)

    #         planning_loss_flow = self.loss_planning_flow(planning_model_outs, data)
    #         loss.update(planning_loss_flow)



    #     if "ssm_dummy" in planning_model_outs:
    #         loss["ssm_dummy_loss"] = planning_model_outs["ssm_dummy"]

    #     return loss





    def loss(
        self,
        motion_model_outs,
        planning_model_outs,
        data,
        motion_loss_cache,
    ):
        loss = {}


        planning_loss = self.loss_planning(
            planning_model_outs,
            data,
        )
        loss.update(planning_loss)

        ssm_selected_loss = self.loss_ssm_selected_reg(
            planning_model_outs,
            data,
        )
        loss.update(ssm_selected_loss)

        if "ssm_dummy" in planning_model_outs:
            loss["ssm_dummy_loss"] = planning_model_outs["ssm_dummy"]


        # Flow matching 第二创新点：现在 use_flow_aux_loss=False，所以不会执行

        return loss








    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            # reg_weight = reg_weight.flatten(end_dim=1)
            # reg_pred = reg_pred.flatten(end_dim=1)
            # reg_target = reg_target.flatten(end_dim=1)
            # reg_weight = reg_weight.unsqueeze(-1)

            # reg_loss = self.plan_loss_reg(
            #     reg_pred, reg_target, weight=reg_weight
            # )

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)

            reg_weight_2d = reg_weight
            reg_weight = reg_weight.unsqueeze(-1)

            # 1. delta-space loss：保留原来的单步位移监督，保护轨迹平滑和 collision
            reg_loss_delta = self.plan_loss_reg(
                reg_pred,
                reg_target,
                weight=reg_weight,
            )

            # 2. cumulative-space loss：累计轨迹空间，更贴近最终 L2
            valid_mask = (reg_weight_2d > 0).float().unsqueeze(-1)

            reg_pred_abs = (reg_pred * valid_mask).cumsum(dim=-2)
            reg_target_abs = (reg_target * valid_mask).cumsum(dim=-2)

            reg_loss_cum = self.plan_loss_reg(
                reg_pred_abs,
                reg_target_abs,
                weight=reg_weight,
            )

            # 3. 混合 loss：先保守一点，delta 为主，cum 为辅
            reg_loss = 0.8 * reg_loss_delta + 0.2 * reg_loss_cum





            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output
    
    def loss_planning_refined(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction_refined"]
        status_preds = model_outs["status_refined"]
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target, 
                cls_weight, 
                reg_pred, 
                reg_target, 
                reg_weight, 
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_refined_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_refined_{decoder_idx}": reg_loss,
                    f"planning_loss_status_refined_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(
        apply_to=(
            "det_output",
            "motion_output",
            "planning_output",
        )
    )
    def post_process(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )

        # 默认只用第一个 predict
        decode_planning_output = planning_output

        # Flow matching 第二创新点：只有 use_flow_decode=True 才会走

        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            decode_planning_output,
            data,
        )

        return motion_result, planning_result
