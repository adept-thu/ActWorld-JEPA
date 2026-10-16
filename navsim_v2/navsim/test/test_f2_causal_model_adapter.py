"""CPU-only contract tests for the strict F2 causal model adapter."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import torch

from navsim.planning.validation.f2_causal_model_adapter import (
    F2_FROZEN_ACTOR_TENSOR_NAMES,
    F2CausalModelAdapter,
    F2CausalModelAdapterError,
    canonical_original_proposal_sha256,
    checkpoint_file_sha256,
    strip_state_dict_prefix,
)
from navsim.planning.validation.f2_causal_validator import (
    CANDIDATE_COUNT,
    F2_MODEL_NUMERIC_MODE,
    ModelTokenOutput,
)
from navsim.planning.validation.f2_live_provenance import VerifiedF2Provenance


class _MockActWorld(torch.nn.Module):
    def __init__(self, fault: str = "") -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(99.0, dtype=torch.float32))
        self.fault = fault
        self.observed: dict[str, Any] = {}

    def forward(self, features: Mapping[str, Any]) -> Mapping[str, Any]:
        feature = features["feature"]
        try:
            autocast_cpu = torch.is_autocast_enabled("cpu")
        except TypeError:
            autocast_cpu = torch.is_autocast_cpu_enabled()
        self.observed = {
            "autocast_cpu": autocast_cpu,
            "autocast_cuda": torch.is_autocast_enabled(),
            "feature_device": feature.device.type,
            "feature_dtype": feature.dtype,
            "grad_enabled": torch.is_grad_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
            "model_dtype": self.bias.dtype,
            "training": self.training,
        }
        dtype = feature.dtype
        device = feature.device
        proposals = torch.arange(
            CANDIDATE_COUNT * 8 * 3, device=device, dtype=dtype
        ).reshape(1, CANDIDATE_COUNT, 8, 3) / 1000.0
        if self.fault == "near_cancellation":
            residual = torch.full(
                (1, CANDIDATE_COUNT),
                -2.5724000930786133,
                device=device,
                dtype=dtype,
            )
            delta = torch.tanh(residual)
            base = -delta
        else:
            base = (
                torch.arange(CANDIDATE_COUNT, device=device, dtype=dtype)[None]
                / 100.0
                + self.bias
            )
            residual = (
                torch.arange(CANDIDATE_COUNT, device=device, dtype=dtype)[None]
                / 90.0
                - 0.1
            )
            delta = torch.tanh(residual)
        utility = base.detach() + delta
        pred_logit = torch.stack((base - 1.0, base), dim=-1)
        output: dict[str, Any] = {
            "proposals": proposals,
            "actworld_original_candidates": proposals.clone(),
            "pred_logit": pred_logit,
            "actworld_base_candidate_logits": base,
            "actworld_residual_logits": residual,
            "actworld_delta_logits": delta,
            "actworld_utility_logits": utility,
            # Deliberately different: the adapter must never hash this tensor.
            "refined_proposals": proposals + 1000.0,
        }
        if self.fault == "formula":
            output["actworld_utility_logits"] = utility.clone()
            output["actworld_utility_logits"][0, 7] += 0.25
        elif self.fault == "delta_one_ulp":
            output["actworld_delta_logits"] = delta.clone()
            output["actworld_delta_logits"][0, 7] = torch.nextafter(
                delta[0, 7], torch.full_like(delta[0, 7], float("inf"))
            )
        elif self.fault == "utility_one_ulp":
            output["actworld_utility_logits"] = utility.clone()
            output["actworld_utility_logits"][0, 7] = torch.nextafter(
                utility[0, 7], torch.full_like(utility[0, 7], float("inf"))
            )
        elif self.fault == "utility_small_drift":
            output["actworld_utility_logits"] = utility.clone()
            output["actworld_utility_logits"][0, 7] += 1.0e-6
        elif self.fault == "base":
            output["actworld_base_candidate_logits"] = base + 0.01
        elif self.fault == "original":
            output["actworld_original_candidates"] = proposals + 0.01
        elif self.fault == "half":
            output["actworld_residual_logits"] = residual.half()
        elif self.fault == "missing":
            del output["actworld_residual_logits"]
        elif self.fault == "batch":
            for key in (
                "proposals",
                "actworld_original_candidates",
                "pred_logit",
                "actworld_base_candidate_logits",
                "actworld_residual_logits",
                "actworld_delta_logits",
                "actworld_utility_logits",
            ):
                output[key] = output[key].repeat(
                    (2,) + (1,) * (output[key].ndim - 1)
                )
        elif self.fault == "nonfinite":
            output["actworld_residual_logits"] = residual.clone()
            output["actworld_residual_logits"][0, 0] = float("nan")
        elif self.fault == "not_mapping":
            return []
        return output


def _mock_frozen_actor_tensors(
    model: torch.nn.Module,
) -> Mapping[str, torch.Tensor]:
    zero = torch.zeros_like(model.bias)
    tensors = {name: zero.clone() for name in F2_FROZEN_ACTOR_TENSOR_NAMES}
    if getattr(model, "fault", "") == "actor_nonzero":
        tensors["refinement_gate_logits"] = torch.ones_like(model.bias)
    return tensors


class F2CausalModelAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.checkpoint_path = Path(self.temp_directory.name) / "epoch_2.ckpt"

    def _save_checkpoint(
        self,
        *,
        epoch: Any = 2,
        stage: Any = "critic",
        variant: Any = "f2_base_residual",
        state_dict: Any = None,
        payload: Any = None,
    ) -> str:
        if payload is None:
            if state_dict is None:
                state_dict = {"bias": torch.tensor(0.25, dtype=torch.float32)}
            payload = {
                "actworld_training_stage": stage,
                "actworld_critic_variant": variant,
                "epoch": epoch,
                "state_dict": state_dict,
            }
        torch.save(payload, self.checkpoint_path)
        return checkpoint_file_sha256(self.checkpoint_path)

    def _adapter(
        self,
        *,
        digest: str | None = None,
        expected_epoch: int = 2,
        fault: str = "",
        **kwargs: Any,
    ) -> F2CausalModelAdapter:
        if digest is None:
            digest = checkpoint_file_sha256(self.checkpoint_path)
        kwargs.setdefault(
            "frozen_actor_tensor_getter", _mock_frozen_actor_tensors
        )
        return F2CausalModelAdapter.from_checkpoint(
            self.checkpoint_path,
            expected_sha256=digest,
            expected_epoch=expected_epoch,
            model_factory=lambda: _MockActWorld(fault),
            **kwargs,
        )

    def test_strict_restore_and_float32_no_autocast_inference(self) -> None:
        digest = self._save_checkpoint()
        adapter = self._adapter(digest=digest.upper())
        original_features = {"feature": torch.tensor([3.0], dtype=torch.float64)}

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = adapter.infer_token("token-001", original_features)

        self.assertIsInstance(output, ModelTokenOutput)
        self.assertEqual(output.numeric_mode, F2_MODEL_NUMERIC_MODE)
        self.assertEqual(output.token, "token-001")
        self.assertEqual(len(output.base_logits), CANDIDATE_COUNT)
        self.assertEqual(len(output.delta_logits), CANDIDATE_COUNT)
        self.assertAlmostEqual(output.base_logits[0], 0.25, places=6)
        self.assertEqual(adapter.checkpoint_identity.sha256, digest)
        self.assertEqual(adapter.checkpoint_identity.epoch, 2)
        self.assertEqual(adapter.checkpoint_identity.stage, "critic")
        self.assertEqual(
            adapter.checkpoint_identity.variant, "f2_base_residual"
        )
        self.assertFalse(adapter.is_official_factory)
        self.assertIsNone(adapter.live_provenance)
        self.assertIsNone(adapter.factory_source)
        self.assertEqual(adapter.model.bias.dtype, torch.float32)
        self.assertFalse(adapter.model.training)
        self.assertEqual(original_features["feature"].dtype, torch.float64)
        self.assertEqual(
            adapter.model.observed,
            {
                "autocast_cpu": False,
                "autocast_cuda": False,
                "feature_device": "cpu",
                "feature_dtype": torch.float32,
                "grad_enabled": False,
                "inference_mode": True,
                "model_dtype": torch.float32,
                "training": False,
            },
        )

    def test_only_internal_official_restore_binds_live_provenance(self) -> None:
        digest = self._save_checkpoint()
        provenance = object.__new__(VerifiedF2Provenance)
        source_identity = object()

        adapter = F2CausalModelAdapter._from_official_checkpoint(
            self.checkpoint_path,
            expected_sha256=digest,
            expected_epoch=2,
            model_factory=_MockActWorld,
            forward_fn=lambda model, features: model(features),
            live_provenance=provenance,
            factory_source=source_identity,
            frozen_actor_tensor_getter=_mock_frozen_actor_tensors,
        )

        self.assertTrue(adapter.is_official_factory)
        self.assertIs(adapter.live_provenance, provenance)
        self.assertIs(adapter.factory_source, source_identity)

    def test_output_hash_uses_original_not_refined_proposals(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter()
        output = adapter.infer_token(
            "token-002", {"feature": torch.tensor([0.0])}
        )
        expected_original = torch.arange(
            CANDIDATE_COUNT * 8 * 3, dtype=torch.float32
        ).reshape(1, CANDIDATE_COUNT, 8, 3) / 1000.0
        refined = expected_original + 1000.0
        self.assertEqual(
            output.proposal_sha256,
            canonical_original_proposal_sha256(expected_original),
        )
        self.assertNotEqual(
            output.proposal_sha256,
            canonical_original_proposal_sha256(refined),
        )

    def test_proposal_hash_is_batch_independent_and_order_sensitive(self) -> None:
        proposals = torch.arange(
            CANDIDATE_COUNT * 2 * 3, dtype=torch.float64
        ).reshape(1, CANDIDATE_COUNT, 2, 3)
        batched = canonical_original_proposal_sha256(proposals)
        batchless = canonical_original_proposal_sha256(proposals[0].float())
        reordered = proposals.clone()
        reordered[:, [0, 1]] = reordered[:, [1, 0]]
        self.assertEqual(batched, batchless)
        self.assertNotEqual(
            batched, canonical_original_proposal_sha256(reordered)
        )

    def test_proposal_hash_collapses_signed_zero(self) -> None:
        positive = torch.zeros((1, CANDIDATE_COUNT, 1, 3), dtype=torch.float32)
        negative = positive.clone()
        negative[0, 0, 0, 0] = -0.0
        self.assertEqual(
            canonical_original_proposal_sha256(positive),
            canonical_original_proposal_sha256(negative),
        )

    def test_proposal_hash_rejects_invalid_shape_and_nonfinite(self) -> None:
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, r"\[32, poses>=1, 3\]"
        ):
            canonical_original_proposal_sha256(torch.zeros((1, 31, 8, 3)))
        invalid = torch.zeros((1, CANDIDATE_COUNT, 8, 3))
        invalid[0, 0, 0, 0] = float("inf")
        with self.assertRaisesRegex(F2CausalModelAdapterError, "non-finite"):
            canonical_original_proposal_sha256(invalid)

    def test_adapter_rejects_formula_mismatch_before_python_conversion(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="formula")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, r"base_logits\.detach\(\) \+ delta_logits"
        ):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_exact_check_rejects_one_ulp_delta_drift(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="delta_one_ulp")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "delta_logits do not exactly equal"
        ):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_exact_check_rejects_one_ulp_utility_drift(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="utility_one_ulp")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "utility_logits do not exactly equal"
        ):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_exact_check_rejects_small_utility_drift(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="utility_small_drift")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "utility_logits do not exactly equal"
        ):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def _assert_near_cancellation(self, device: str) -> None:
        adapter = self._adapter(fault="near_cancellation", device=device)
        output = adapter.infer_token(
            f"near-cancellation-{device}",
            {"feature": torch.tensor([0.0], device=device)},
        )
        self.assertEqual(output.utility_logits, (0.0,) * CANDIDATE_COUNT)
        self.assertEqual(
            output.delta_logits,
            (-0.9884083271026611,) * CANDIDATE_COUNT,
        )

    def test_cpu_near_cancellation_preserves_device_delta(self) -> None:
        self._save_checkpoint()
        self._assert_near_cancellation("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_near_cancellation_preserves_device_delta(self) -> None:
        self._save_checkpoint()
        self._assert_near_cancellation("cuda")

    def test_restore_rejects_nonzero_frozen_actor_functional_tensor(self) -> None:
        self._save_checkpoint()
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "frozen actor functional-zero"
        ):
            self._adapter(fault="actor_nonzero")

    def test_restore_requires_a_resolvable_frozen_actor_audit(self) -> None:
        digest = self._save_checkpoint()
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "cannot locate.*planner"
        ):
            self._adapter(
                digest=digest,
                frozen_actor_tensor_getter=None,
            )

    def test_adapter_rejects_base_logit_wiring_drift(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="base")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "pred_logit"
        ):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_rejects_original_proposal_wiring_drift(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="original")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "original candidates differ"
        ):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_rejects_non_float32_output(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="half")
        with self.assertRaisesRegex(F2CausalModelAdapterError, "must be float32"):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_rejects_missing_output(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="missing")
        with self.assertRaisesRegex(F2CausalModelAdapterError, "missing required"):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_rejects_multi_token_batch(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="batch")
        with self.assertRaisesRegex(F2CausalModelAdapterError, r"exactly one|\[1, 32"):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_rejects_nonfinite_output(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="nonfinite")
        with self.assertRaisesRegex(F2CausalModelAdapterError, "non-finite"):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_adapter_rejects_non_mapping_forward(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter(fault="not_mapping")
        with self.assertRaisesRegex(F2CausalModelAdapterError, "return a mapping"):
            adapter.infer_token("token", {"feature": torch.tensor([0.0])})

    def test_digest_mismatch_is_rejected_before_deserialization(self) -> None:
        self._save_checkpoint()
        loader_called = False

        def forbidden_loader(path: Path) -> Any:
            nonlocal loader_called
            loader_called = True
            raise AssertionError(path)

        with self.assertRaisesRegex(F2CausalModelAdapterError, "SHA-256 mismatch"):
            self._adapter(digest="0" * 64, checkpoint_loader=forbidden_loader)
        self.assertFalse(loader_called)

    def test_checkpoint_stage_is_strict(self) -> None:
        digest = self._save_checkpoint(stage="actor")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "training_stage='critic'"
        ):
            self._adapter(digest=digest)

    def test_checkpoint_variant_is_strict(self) -> None:
        digest = self._save_checkpoint(variant="legacy")
        with self.assertRaisesRegex(
            F2CausalModelAdapterError, "f2_base_residual"
        ):
            self._adapter(digest=digest)

    def test_checkpoint_epoch_is_strict(self) -> None:
        digest = self._save_checkpoint(epoch=1)
        with self.assertRaisesRegex(F2CausalModelAdapterError, "epoch mismatch"):
            self._adapter(digest=digest, expected_epoch=2)

    def test_expected_epoch_must_belong_to_frozen_schedule(self) -> None:
        digest = self._save_checkpoint()
        with self.assertRaisesRegex(F2CausalModelAdapterError, "0, 1, 2, 3"):
            self._adapter(digest=digest, expected_epoch=4)
        with self.assertRaisesRegex(F2CausalModelAdapterError, "must be an integer"):
            self._adapter(digest=digest, expected_epoch=True)

    def test_checkpoint_state_must_be_nonempty_tensor_mapping(self) -> None:
        digest = self._save_checkpoint(state_dict={})
        with self.assertRaisesRegex(F2CausalModelAdapterError, "non-empty mapping"):
            self._adapter(digest=digest)
        digest = self._save_checkpoint(state_dict={"bias": "not-a-tensor"})
        with self.assertRaisesRegex(F2CausalModelAdapterError, "torch.Tensor"):
            self._adapter(digest=digest)

    def test_checkpoint_floating_state_must_be_float32(self) -> None:
        digest = self._save_checkpoint(
            state_dict={"bias": torch.tensor(0.25, dtype=torch.float16)}
        )
        with self.assertRaisesRegex(F2CausalModelAdapterError, "must be float32"):
            self._adapter(digest=digest)

    def test_strict_restore_rejects_missing_and_unexpected_keys(self) -> None:
        digest = self._save_checkpoint(
            state_dict={"unexpected": torch.tensor(0.25, dtype=torch.float32)}
        )
        with self.assertRaisesRegex(F2CausalModelAdapterError, "state key mismatch"):
            self._adapter(digest=digest)

    def test_strict_restore_rejects_tensor_shape_mismatch(self) -> None:
        digest = self._save_checkpoint(
            state_dict={"bias": torch.ones(2, dtype=torch.float32)}
        )
        with self.assertRaisesRegex(F2CausalModelAdapterError, "shape mismatch"):
            self._adapter(digest=digest)

    def test_strict_restore_rejects_nonfloating_tensor_dtype_mismatch(self) -> None:
        class _BufferedMock(_MockActWorld):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

        digest = self._save_checkpoint(
            state_dict={
                "bias": torch.tensor(0.25, dtype=torch.float32),
                "counter": torch.tensor(0, dtype=torch.int32),
            }
        )
        with self.assertRaisesRegex(F2CausalModelAdapterError, "dtype mismatch"):
            F2CausalModelAdapter.from_checkpoint(
                self.checkpoint_path,
                expected_sha256=digest,
                expected_epoch=2,
                model_factory=_BufferedMock,
                frozen_actor_tensor_getter=_mock_frozen_actor_tensors,
            )

    def test_prefix_transform_supports_direct_model_restore(self) -> None:
        digest = self._save_checkpoint(
            state_dict={
                "agent._pad_model.bias": torch.tensor(0.75, dtype=torch.float32)
            }
        )
        adapter = self._adapter(
            digest=digest,
            state_dict_transform=strip_state_dict_prefix("agent._pad_model."),
        )
        self.assertAlmostEqual(float(adapter.model.bias.detach()), 0.75)

    def test_prefix_transform_rejects_mixed_lineage(self) -> None:
        digest = self._save_checkpoint(
            state_dict={"foreign.bias": torch.tensor(0.75, dtype=torch.float32)}
        )
        with self.assertRaisesRegex(F2CausalModelAdapterError, "outside.*prefix"):
            self._adapter(
                digest=digest,
                state_dict_transform=strip_state_dict_prefix("agent._pad_model."),
            )

    def test_checkpoint_payload_and_factory_must_have_expected_types(self) -> None:
        digest = self._save_checkpoint(payload=["not", "a", "mapping"])
        with self.assertRaisesRegex(F2CausalModelAdapterError, "payload.*mapping"):
            self._adapter(digest=digest)

        digest = self._save_checkpoint()
        with self.assertRaisesRegex(F2CausalModelAdapterError, "torch.nn.Module"):
            F2CausalModelAdapter.from_checkpoint(
                self.checkpoint_path,
                expected_sha256=digest,
                expected_epoch=2,
                model_factory=lambda: object(),
            )

    def test_token_and_feature_boundary_is_strict(self) -> None:
        self._save_checkpoint()
        adapter = self._adapter()
        with self.assertRaisesRegex(F2CausalModelAdapterError, "non-empty"):
            adapter.infer_token("", {"feature": torch.tensor([0.0])})
        with self.assertRaisesRegex(F2CausalModelAdapterError, "mapping"):
            adapter.infer_token("token", [torch.tensor([0.0])])


if __name__ == "__main__":
    unittest.main()
