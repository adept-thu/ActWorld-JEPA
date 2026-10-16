from typing import List

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_states_to_state_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    WeightedMetricIndex,
)
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import (
    AbstractTrafficAgentsPolicy,
)
from navsim.common.utils import mean_time_every_5_calls


def transform_trajectory(
    pred_trajectory: Trajectory, initial_ego_state: EgoState
) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(
        initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length
    )

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(
        initial_ego_state.rear_axle, relative_states
    )

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)

def fast_transform_trajectory(
    pred_trajectory: Trajectory, 
    tgt_sampling: TrajectorySampling,
    initial_ego_state: EgoState,
) -> npt.NDArray[np.float64]:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling

    # Source and target time grids
    t_src = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    t_tgt = np.arange(
        0.0,
        tgt_sampling.time_horizon + tgt_sampling.interval_length,
        tgt_sampling.interval_length,
    )

    # Relative poses: [N, 3] (x_rel, y_rel, heading_rel)
    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)

    # Origin (rear axle) pose in global frame
    origin = initial_ego_state.rear_axle
    x0, y0, h0 = origin.x, origin.y, origin.heading

    # Rotation matrix for origin heading
    cos_h0 = np.cos(h0)
    sin_h0 = np.sin(h0)
    R = np.array([[cos_h0, -sin_h0], [sin_h0, cos_h0]])

    # Transform relative positions to global positions
    rel_xy = relative_poses[:, :2]  # (N, 2)
    abs_xy = rel_xy @ R.T + np.array([x0, y0])  # (N, 2)

    # Transform headings to global headings
    abs_heading = relative_poses[:, 2] + h0  # (N,)

    # Stack into (N+1, 3) pose array and prepend the initial pose at t=0
    poses_arr = np.empty((relative_poses.shape[0] + 1, 3), dtype=np.float64)
    poses_arr[0] = np.array([x0, y0, h0], dtype=np.float64)
    poses_arr[1:, :2] = abs_xy
    poses_arr[1:, 2] = abs_heading
    
    poses_arr[:, 2] = np.unwrap(poses_arr[:, 2])

    intp = interp1d(t_src, poses_arr, axis=0)
    pred_states = intp(t_tgt)
    
    min_ = -np.pi
    pred_states[:, 2] = (pred_states[:, 2] - min_) % (2 * np.pi) + min_
    return pred_states


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(
        times_us, trajectory.start_time.time_us, trajectory.end_time.time_us
    )
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
) -> pd.DataFrame:
    """
    Runs PDM-Score and saves results in dataclass.
    :param metric_cache: Metric cache dataclass
    :param model_trajectory: Predicted trajectory in ego frame.
    :return: Dataclass of PDM-Subscores.
    """

    pred_trajectory = transform_trajectory(model_trajectory, metric_cache.ego_state)

    return pdm_score_from_interpolated_trajectory(
        metric_cache=metric_cache,
        pred_trajectory=pred_trajectory,
        future_sampling=future_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )


def pdm_score_from_interpolated_trajectory(
    metric_cache: MetricCache,
    pred_trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
):
    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory

    pdm_states, pred_states = (
        get_trajectory_as_array(
            pdm_trajectory, future_sampling, initial_ego_state.time_point
        ),
        get_trajectory_as_array(
            pred_trajectory, future_sampling, initial_ego_state.time_point
        ),
    )
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], pred_states[None, ...]], axis=0
    )

    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state
    )

    # infer traffic agents policy and update future observation
    simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(
        simulated_states[1], metric_cache
    )

    assert (
        len(simulated_agent_detections_tracks) == trajectory_states.shape[1]
    ), f"""
            Traffic agents policy returned trajectories of invalid length:
            Traffic agents trajectories must be of length ego_trajectory_length = {trajectory_states.shape[1]},
            but got {len(simulated_agent_detections_tracks)}
        """

    pred_idx = (
        1  # index of predicted trajectory in trajectory_states and simulated_states
    )
    pdm_result = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        simulated_agent_detections_tracks,
        metric_cache.past_human_trajectory,
    )[pred_idx]

    if (
        scorer._config.human_penalty_filter
        and metric_cache.scene_type == SceneFrameType.ORIGINAL
    ):
        # human_penalty_filter

        human_trajectory = transform_trajectory(
            metric_cache.human_trajectory, initial_ego_state
        )

        human_states = get_trajectory_as_array(
            human_trajectory, future_sampling, initial_ego_state.time_point
        )

        human_simulated_states = simulator.simulate_proposals(
            human_states[None, ...], initial_ego_state
        )

        human_simulated_agent_detections_tracks = (
            traffic_agents_policy.simulate_environment(
                human_simulated_states[0], metric_cache
            )
        )

        human_pdm_result = scorer.score_proposals(
            human_simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_simulated_agent_detections_tracks,
        )[0]

        skip_columns = {
            "multiplicative_metrics_prod",
            "weighted_metrics",
            "weighted_metrics_array",
            "pdm_score",
        }
        modified_any = False

        for column in human_pdm_result.columns:
            if column not in skip_columns and human_pdm_result[column].iloc[0] == 0:
                pdm_result.at[0, column] = 1
                modified_any = True

        # Keep all aggregate fields consistent with the human-filtered subscores.
        # This mirrors the upstream NAVSIM fix for Issue #151.
        if modified_any:
            pdm_result.at[0, "multiplicative_metrics_prod"] = (
                pdm_result.at[0, "no_at_fault_collisions"]
                * pdm_result.at[0, "drivable_area_compliance"]
                * pdm_result.at[0, "driving_direction_compliance"]
                * pdm_result.at[0, "traffic_light_compliance"]
            )

            weighted_metrics = pdm_result.at[0, "weighted_metrics"].copy()
            weighted_metrics[WeightedMetricIndex.PROGRESS] = pdm_result.at[
                0, "ego_progress"
            ]
            weighted_metrics[WeightedMetricIndex.TTC] = pdm_result.at[
                0, "time_to_collision_within_bound"
            ]
            weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = pdm_result.at[
                0, "lane_keeping"
            ]
            weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT] = pdm_result.at[
                0, "history_comfort"
            ]
            pdm_result.at[0, "weighted_metrics"] = weighted_metrics

    return pdm_result, simulated_states[pred_idx]
