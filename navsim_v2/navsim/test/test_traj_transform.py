import lzma
import pickle
import unittest
import time
from typing import List

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.evaluate.pdm_score import (
    fast_transform_trajectory,
    get_trajectory_as_array,
    transform_trajectory,
)
from navsim.common.dataclasses import Trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.simulation_v2.pdm_simulator import PDMSimulator as PDMSimulatorV2


def generate_perturbed_poses(
    poses: np.ndarray,
    num_samples: int = 32,
    scale: float = 10.0,
) -> np.ndarray:
    """
    Generate perturbed pose sequences from a base trajectory.

    Args:
        poses: (T, 3) array [x, y, heading] for a single trajectory.
        num_samples: number of perturbed trajectories to generate.
        scale: multiplicative scale for perturbation magnitude.

    Returns:
        (num_samples, T, 3) array of perturbed pose sequences.
    """
    assert poses.ndim == 2 and poses.shape[1] == 3, "poses must have shape (T, 3)"
    num_steps = poses.shape[0]

    # Compute approximate per-step motion to scale perturbations
    deltas = poses[1:] - poses[:-1]  # (T-1, 3)
    mean_step_xy = np.linalg.norm(deltas[:, :2], axis=1).mean() + 1e-6

    # Reasonable perturbation magnitudes (scaled by motion)
    pos_std = 0.2 * mean_step_xy * scale  # meters
    heading_std = np.deg2rad(2.0) * scale  # radians

    # Create base times to add small speed / heading-rate variations
    t = np.arange(num_steps, dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-6)  # normalized time

    samples = np.zeros((num_samples, num_steps, 3), dtype=poses.dtype)

    for i in range(num_samples):
        # Random small global translation and yaw offset
        global_dx = np.random.normal(0.0, pos_std)
        global_dy = np.random.normal(0.0, pos_std)
        global_dyaw = np.random.normal(0.0, heading_std)

        # Small time-correlated perturbations (simulate smooth deviations)
        speed_scale = 1.0 + np.random.normal(0.0, 0.05)  # ±5% speed
        yaw_rate_bias = np.random.normal(0.0, heading_std / 4.0)

        # Start from original
        pert = poses.copy()

        # Apply global transform
        pert[:, 0] += global_dx
        pert[:, 1] += global_dy
        pert[:, 2] += global_dyaw

        # Apply smooth, time-varying x/y perturbations along approximate heading
        headings = pert[:, 2]
        cos_h = np.cos(headings)
        sin_h = np.sin(headings)

        # Longitudinal and lateral offsets (smooth over time)
        long_offset = np.random.normal(0.0, pos_std / 2.0) * t
        lat_offset = np.random.normal(0.0, pos_std / 2.0) * t

        pert[:, 0] += long_offset * cos_h - lat_offset * sin_h
        pert[:, 1] += long_offset * sin_h + lat_offset * cos_h

        # Smooth heading drift (simulating slight curvature/swerve)
        heading_drift = yaw_rate_bias * t
        pert[:, 2] += heading_drift

        # Slight rescaling of total displacement to keep trajectory plausible
        # (approximate by scaling deviations from first pose)
        disp = pert - pert[0:1]
        pert = pert[0:1] + disp * speed_scale

        samples[i] = pert

    return samples


class TrajTransformTest(unittest.TestCase):
    def setUp(self) -> None:
        metric_cache_paths = [
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.06.14.20.14.09_veh-26_00612_01016/unknown/2ea60bb9a43b5d67/metric_cache.pkl",
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/e27346850f555e83/metric_cache.pkl",
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/83cc4a084e7c52b6/metric_cache.pkl",
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/e27346850f555e83/metric_cache.pkl",
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/83cc4a084e7c52b6/metric_cache.pkl",
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/e27346850f555e83/metric_cache.pkl",
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/83cc4a084e7c52b6/metric_cache.pkl",
            "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/9eba3b8ff5d55a1d/metric_cache.pkl",
        ]
        self.metric_caches: List[MetricCache] = []
        for metric_cache_path in metric_cache_paths:
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
                self.metric_caches.append(metric_cache)
        self.proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        self.simulator = PDMSimulator(self.proposal_sampling)
        self.simulator_v2 = PDMSimulatorV2(self.proposal_sampling)

    def test_trajectory_interpolation(self):
        for i in range(len(self.metric_caches)):
            metric_cache = self.metric_caches[i]
            initial_ego_state = metric_cache.ego_state
            human_trajectory = metric_cache.human_trajectory
            human_poses = human_trajectory.poses
            new_poses = generate_perturbed_poses(human_poses, num_samples=32)
            for poses in new_poses:
                new_trajectory = Trajectory(poses)
                trajectory = transform_trajectory(new_trajectory, initial_ego_state)
                gt_states = get_trajectory_as_array(
                    trajectory, self.proposal_sampling, initial_ego_state.time_point
                )

                fast_states = fast_transform_trajectory(
                    new_trajectory, self.proposal_sampling, initial_ego_state
                )

                np.testing.assert_allclose(
                    gt_states[:, :3],
                    fast_states,
                    rtol=0,
                    atol=1e-4,
                    err_msg=(
                        "Max diff between fast interpolation and nuplan interpolation "
                        "should be smaller than 1e-4"
                    ),
                )
    
    def test_speed_fast_tt(self):
        metric_cache = self.metric_caches[2]
        initial_ego_state = metric_cache.ego_state
        human_trajectory = metric_cache.human_trajectory
        human_poses = human_trajectory.poses
        new_poses = generate_perturbed_poses(human_poses, num_samples=32)

        times = []
        for _ in range(10):
            start = time.time()
            fast_states_list = []
            for poses in new_poses:
                new_trajectory = Trajectory(poses)
                fast_states = fast_transform_trajectory(
                    new_trajectory, self.proposal_sampling, initial_ego_state
                )
                fast_states_list.append(fast_states)
            final_states = np.stack(fast_states_list, axis=0)

            end = time.time()
            times.append(end - start)

        avg_time = sum(times) / len(times)
        print(f"test_speed_fast_tt average over 5 runs: {avg_time:.6f} s")
        
        extended = np.zeros((*final_states.shape[:-1], 11), dtype=final_states.dtype)
        extended[:, :, :3] = final_states
        
        simulated_states = self.simulator.simulate_proposals(extended, initial_ego_state)
        simulated_states_v2 = self.simulator_v2.simulate_proposals(extended, initial_ego_state)
        np.testing.assert_allclose(simulated_states, simulated_states_v2, rtol=0, atol=1e-3)
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            simulated_states = self.simulator.simulate_proposals(extended, initial_ego_state)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        print(f"simulator average over 5 runs: {avg_time:.6f} s")
        
        # import cProfile
        # pr = cProfile.Profile()
        # pr.enable()
        
        times = []
        for _ in range(10):
            start = time.perf_counter()
            simulated_states = self.simulator_v2.simulate_proposals(extended, initial_ego_state)
            end = time.perf_counter()
            times.append(end - start)
        # pr.disable()
        # pr.dump_stats('result.out')

        avg_time = sum(times) / len(times)
        print(f"simulator_v2 average over 5 runs: {avg_time:.6f} s")

if __name__ == "__main__":
    unittest.main()
