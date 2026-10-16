import os

import torch.nn as nn

from ..bevformer.transformer_decoder import MLP, MyTransformeDecoder
from ..drive_jepa_config import DriveJEPAConfig
from .map_head import MapHead


class Scorer(nn.Module):
    def __init__(self, config: DriveJEPAConfig):
        super().__init__()

        self.proposal_num = config.proposal_num
        self.score_num = 6

        self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)

        self.agent_pred = config.agent_pred

        if self.agent_pred:
            self.pred_col_agent = MLP(config.tf_d_model, config.tf_d_ffn, 2 * 40 * 9)

        self.area_pred = config.area_pred

        if self.area_pred:
            self.pred_area = MLP(config.tf_d_model, config.tf_d_ffn, 5 * 2)

        self.bev_map = config.bev_map
        self.bev_agent = config.bev_agent

        if config.bev_agent:
            self._agent_head = MyTransformeDecoder(config, config.num_bounding_boxes, 6, trajenc=False)

        if config.bev_map:
            self.map_head = MapHead(config)

    def forward(self, proposals, bev_feature):
        batch_size = len(proposals)
        p_size = proposals.shape[1]
        t_size = proposals.shape[2]

        proposal_feature = bev_feature.reshape(batch_size, p_size, t_size, -1).amax(-2)
        pred_logit = self.pred_score(proposal_feature).reshape(batch_size, -1, self.score_num)

        pred_agents_states = pred_area_logit = bev_semantic_map = agent_states = agent_labels = None

        auxiliary_inference = os.environ.get(
            "ACTWORLD_INFERENCE_AUX_HEADS", "0"
        ) == "1"
        if self.training or auxiliary_inference:
            if self.area_pred:
                pred_area_logit = self.pred_area(bev_feature)

            if self.agent_pred:
                pred_agents_states = self.pred_col_agent(proposal_feature).reshape(batch_size, p_size, t_size, -1, 2, 9)

            if self.bev_map:
                bev_semantic_map = self.map_head(bev_feature)

            if self.bev_agent:
                agents = self._agent_head(None, bev_feature)
                agent_states = agents[:, :, :-1]
                agent_labels = agents[:, :, -1]

        return (
            pred_logit,
            pred_agents_states,
            pred_area_logit,
            bev_semantic_map,
            agent_states,
            agent_labels,
        )
