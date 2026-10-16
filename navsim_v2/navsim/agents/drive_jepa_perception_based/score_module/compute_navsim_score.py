from navsim.planning.simulation.planner.pdm_planner.simulation_v2.pdm_simulator import PDMSimulator
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
import lzma
import pickle
from navsim.evaluate.pdm_score import transform_trajectory, fast_transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.traffic_agents_policies.log_replay_traffic_agents import LogReplayTrafficAgents
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)
import numpy as np
from .train_pdm_scorer import PDMScorerConfig, PDMScorer
from navsim.common.utils import mean_time_every_5_calls

# metric_cache_loader = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache"))
proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
simulator = PDMSimulator(proposal_sampling)
config = PDMScorerConfig( )
scorer = PDMScorer(proposal_sampling, config)
traffic_agents_policy = LogReplayTrafficAgents(simulator.proposal_sampling)


def get_scores(args):
    return [get_sub_score(a["token"],a["poses"],a["test"]) for a in args]

def load_metric_cache(metric_cache_path) -> MetricCache:
    with lzma.open(metric_cache_path, "rb") as f:
        metric_cache: MetricCache = pickle.load(f)
    return metric_cache

def before_score_proposals(metric_cache: MetricCache, poses, initial_ego_state):
    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory

    trajectory_states = [
        get_trajectory_as_array(pdm_trajectory, simulator.proposal_sampling, initial_ego_state.time_point)[:, :3],
    ]
    
    for model_trajectory in poses:
        pred_states_2 = fast_transform_trajectory(Trajectory(model_trajectory), simulator.proposal_sampling, initial_ego_state)
        trajectory_states.append(pred_states_2)

    trajectory_states = np.stack(trajectory_states, axis=0)
    extended = np.zeros((*trajectory_states.shape[:-1], 11), dtype=trajectory_states.dtype)
    extended[:, :, :3] = trajectory_states
    
    simulated_states = simulator.simulate_proposals(extended, initial_ego_state)#32,41,11
    return simulated_states

def get_sub_score( metric_cache_path,poses,test):
    metric_cache = load_metric_cache(metric_cache_path)
    scores_index = None
    if not test:
        scores_index_path = str(metric_cache_path).replace('train_metric_cache_v2_pruned', 'anchors_scores_index_v2')
        scores_index = np.load(scores_index_path + '.npy')

    simulated_states = before_score_proposals(metric_cache, poses, metric_cache.ego_state) 
    # infer traffic agents policy and update future observation
    # We all assume using log replay mode
    # simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(
    #     simulated_states[0], metric_cache
    # )
    simulated_agent_detections_tracks = None

    final_scores = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        simulated_agent_detections_tracks,
        metric_cache.past_human_trajectory,
    )
    scores, key_agent_corners, key_agent_labels, ego_areas = after_score_proposals(scorer, final_scores, metric_cache.ego_state, test)
    return scores, key_agent_corners, key_agent_labels, ego_areas, scores_index

def after_score_proposals(scorer, final_scores, initial_ego_state, test):
    no_at_fault_collisions = scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, :]
    drivable_area_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, :]
    traffic_light_compliance = scorer._multi_metrics[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE, :]
    driving_direction_compliance = scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION, :  ]

    ego_progress = scorer._weighted_metrics[WeightedMetricIndex.PROGRESS, :]
    time_to_collision_within_bound = scorer._weighted_metrics[WeightedMetricIndex.TTC, :]
    lane_keeping = scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING, :]
    comfort = scorer._weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT, :]


    scores=np.stack([no_at_fault_collisions,drivable_area_compliance,traffic_light_compliance,driving_direction_compliance,
                     ego_progress,time_to_collision_within_bound,lane_keeping,comfort,final_scores
                     ],axis=-1)
    
    num_col=2
    key_agent_corners = np.zeros([scores.shape[0], scorer.proposal_sampling.num_poses ,num_col, 4, 2])
    key_agent_labels = np.zeros([scores.shape[0], scorer.proposal_sampling.num_poses ,num_col],dtype=bool)
    ego_areas = scorer._ego_areas[:,1:,1:]

    
    if not test:
        for i in range(len(scores)):
            # proposal_collided_track_ids=scorer.proposal_collided_track_ids[i]
            proposal_fault_collided_track_ids = scorer.proposal_fault_collided_track_ids[i]
            # temp_collided_track_ids=scorer.temp_collided_track_ids[i]

            if len(proposal_fault_collided_track_ids):
                col_token=proposal_fault_collided_track_ids[0]
                collision_time_idcs = int(scorer._collision_time_idcs[i])+1

                for time_idx in range(1,collision_time_idcs):
                    if  col_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,0] = True
                        key_agent_corners[i][time_idx-1,0]=np.array(scorer._observation[time_idx][col_token].boundary.xy).T[:4]

            ttc_collided_track_ids = scorer.ttc_collided_track_ids[i]

            if len(ttc_collided_track_ids):
                ttc_token=ttc_collided_track_ids[0]
                ttc_time_idcs = int(scorer._ttc_time_idcs[i])+1

                for time_idx in range(1,ttc_time_idcs):
                    if  ttc_token in scorer._observation[time_idx].tokens:
                        key_agent_labels[i][time_idx-1,1] = True
                        key_agent_corners[i][time_idx-1,1]=np.array(scorer._observation[time_idx][ttc_token].boundary.xy).T[:4]

        theta = initial_ego_state.rear_axle.heading
        origin_x = initial_ego_state.rear_axle.x
        origin_y = initial_ego_state.rear_axle.y
        
        c, s = np.cos(theta), np.sin(theta)
        mat = np.array([[c, -s],
                        [s, c]])

        key_agent_corners[...,0]-=origin_x
        key_agent_corners[...,1]-=origin_y

        key_agent_corners=key_agent_corners.dot(mat)
    
    return scores[1:],key_agent_corners[1:],key_agent_labels[1:],ego_areas[1:]
