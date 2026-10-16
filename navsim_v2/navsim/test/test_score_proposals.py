import lzma
import pickle
import time
import unittest
from typing import List, Sequence

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.pad_vjepa.score_module_v2.compute_navsim_score import (
    after_score_proposals,
    before_score_proposals,
)
from navsim.agents.pad_vjepa.score_module_v2.train_pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.agents.pad_vjepa.score_module_v2.train_pdm_scorer_v2 import PDMScorer as PDMScorerV2 
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.test.test_traj_transform import generate_perturbed_poses


proposal_sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
_scorer_config = PDMScorerConfig()
scorer = PDMScorer(proposal_sampling, _scorer_config)
scorer_v2 = PDMScorerV2(proposal_sampling, _scorer_config)


def _load_metric_caches(paths: Sequence[str]) -> List[MetricCache]:
    caches: List[MetricCache] = []
    for metric_cache_path in paths:
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)
        caches.append(metric_cache)
    return caches


class ScoreProposalsTest(unittest.TestCase):
    METRIC_CACHE_PATHS = [
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/e27346850f555e83/metric_cache.pkl",
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.06.14.20.14.09_veh-26_00612_01016/unknown/2ea60bb9a43b5d67/metric_cache.pkl",
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/83cc4a084e7c52b6/metric_cache.pkl",
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/e27346850f555e83/metric_cache.pkl",
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/83cc4a084e7c52b6/metric_cache.pkl",
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/e27346850f555e83/metric_cache.pkl",
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/83cc4a084e7c52b6/metric_cache.pkl",
        "/scratch/linhan/train_metric_cache_v2_pruned/2021.05.12.19.36.12_veh-35_00568_01168/unknown/9eba3b8ff5d55a1d/metric_cache.pkl",
    ]
    NUM_SAMPLES = 32

    def setUp(self) -> None:
        self.metric_caches = _load_metric_caches(self.METRIC_CACHE_PATHS)

    def test_score_proposals_pipeline(self) -> None:
        np.random.seed(0)
        for metric_cache in self.metric_caches:
            poses = generate_perturbed_poses(
                metric_cache.human_trajectory.poses, num_samples=self.NUM_SAMPLES, scale=10.0
            )
            simulated_states = before_score_proposals(metric_cache, poses, metric_cache.ego_state)

            final_scores = scorer.score_proposals(
                simulated_states,
                metric_cache.observation,
                metric_cache.centerline,
                metric_cache.route_lane_ids,
                metric_cache.drivable_area_map,
                metric_cache.map_parameters,
                simulated_agent_detections_tracks=None,
                human_past_trajectory=metric_cache.past_human_trajectory,
            )
            (
                scores,
                key_agent_corners,
                key_agent_labels,
                ego_areas,
            ) = after_score_proposals(
                scorer, final_scores, metric_cache.ego_state, test=False
            )
            
            final_scores_v2 = scorer_v2.score_proposals(
                simulated_states,
                metric_cache.observation,
                metric_cache.centerline,
                metric_cache.route_lane_ids,
                metric_cache.drivable_area_map,
                metric_cache.map_parameters,
                simulated_agent_detections_tracks=None,
                human_past_trajectory=metric_cache.past_human_trajectory,
            )
            (
                scores_v2,
                key_agent_corners_v2,
                key_agent_labels_v2,
                ego_areas_v2,
            ) = after_score_proposals(
                scorer_v2, final_scores_v2, metric_cache.ego_state, test=False
            )

            np.testing.assert_allclose(final_scores, final_scores_v2, rtol=0, atol=1e-2)
            np.testing.assert_allclose(key_agent_corners, key_agent_corners_v2, rtol=0, atol=1e-2)
            np.testing.assert_allclose(key_agent_labels, key_agent_labels_v2, rtol=0, atol=1e-2)
            np.testing.assert_allclose(ego_areas, ego_areas_v2, rtol=0, atol=1e-2)
        
            num_proposals = poses.shape[0]

            self.assertTrue(np.isfinite(final_scores).all(), "final scores contain NaN/Inf values")
            self.assertTrue(np.isfinite(scores).all(), "stacked scores contain NaN/Inf values")
            self.assertEqual(scores.shape[0], num_proposals)
            self.assertEqual(scores.shape[-1], 9)

            self.assertEqual(key_agent_corners.shape[0], num_proposals)
            self.assertEqual(key_agent_corners.shape[1], scorer.proposal_sampling.num_poses)
            self.assertEqual(key_agent_corners.shape[2], 2)
            self.assertEqual(key_agent_corners.shape[-2:], (4, 2))
            self.assertEqual(key_agent_labels.shape, key_agent_corners.shape[:3])

            self.assertEqual(ego_areas.shape[0], num_proposals)
            self.assertTrue(np.isfinite(ego_areas).all(), "ego areas contain NaN/Inf values")

    def test_score_proposals_speed(self) -> None:
        np.random.seed(1)
        metric_cache = self.metric_caches[5]
        
        poses = generate_perturbed_poses(
            metric_cache.human_trajectory.poses, num_samples=self.NUM_SAMPLES, scale=10.0
        )
        simulated_states = before_score_proposals(metric_cache, poses, metric_cache.ego_state)
        
        for i in range(3):
            # Warm-up call
            scorer_v2.score_proposals(
                simulated_states,
                metric_cache.observation,
                metric_cache.centerline,
                metric_cache.route_lane_ids,
                metric_cache.drivable_area_map,
                metric_cache.map_parameters,
                simulated_agent_detections_tracks=None,
                human_past_trajectory=metric_cache.past_human_trajectory,
            )
        
        import cProfile
        pr = cProfile.Profile()
        pr.enable()

        timings = []
        for i in range(10):
            start = time.perf_counter()
            scorer_v2.score_proposals(
                simulated_states,
                metric_cache.observation,
                metric_cache.centerline,
                metric_cache.route_lane_ids,
                metric_cache.drivable_area_map,
                metric_cache.map_parameters,
                simulated_agent_detections_tracks=None,
                human_past_trajectory=metric_cache.past_human_trajectory,
            )
            end = time.perf_counter()
            timings.append(end - start)
        
        pr.disable()
        pr.dump_stats('result.out')

        avg = sum(timings) / len(timings)
        print(
            f"score_proposals speed benchmark over {len(timings)} cases: "
            f"avg={avg:.6f}s min={min(timings):.6f}s max={max(timings):.6f}s"
        )

    def test_calc_ego_area_speed(self) -> None:
        np.random.seed(1)
        metric_cache = self.metric_caches[5]
        
        poses = generate_perturbed_poses(
            metric_cache.human_trajectory.poses, num_samples=self.NUM_SAMPLES, scale=10.0
        )
        simulated_states = before_score_proposals(metric_cache, poses, metric_cache.ego_state)
        
        for i in range(3):
            # Warm-up call
            scorer_v2._reset(
                simulated_states,
                metric_cache.observation,
                metric_cache.centerline,
                metric_cache.route_lane_ids,
                metric_cache.drivable_area_map,
                metric_cache.past_human_trajectory,
            )

            scorer_v2._calculate_ego_area()
        
        # import cProfile
        # pr = cProfile.Profile()
        # pr.enable()

        timings = []
        for i in range(10):
            start = time.perf_counter()
            scorer_v2._calculate_ego_area()
            end = time.perf_counter()
            timings.append(end - start)
        
        # pr.disable()
        # pr.dump_stats('result.out')

        avg = sum(timings) / len(timings)
        print(
            f"_calculate_ego_area speed benchmark over {len(timings)} cases: "
            f"avg={avg:.6f}s min={min(timings):.6f}s max={max(timings):.6f}s"
        )

if __name__ == "__main__":
    unittest.main()
