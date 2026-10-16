"""Synthetic fixed-shape tests for the fail-closed F2 artifact binary I/O."""

from __future__ import annotations

import gc
import hashlib
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from navsim.planning.validation.f2_frozen_artifact_io import (
    ArtifactBindings,
    CoverageEntry,
    FrozenArtifactChunk,
    FrozenArtifactIOError,
    FrozenArtifactReader,
    FrozenTokenRecord,
    F2_VJEPA_CHECKPOINT_SHA256,
    RECORD_DTYPE,
    RECORD_SIZE,
    TARGET_CHANNEL_ORDER,
    build_artifact_coverage,
    seal_frozen_artifact,
    verify_frozen_artifact,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        qualification_manifest_sha256=_sha("qualification-manifest"),
        causal_manifest_sha256=_sha("causal-manifest"),
        baseline_checkpoint_sha256=_sha("baseline"),
        vjepa_checkpoint_sha256=F2_VJEPA_CHECKPOINT_SHA256,
        cache_metadata_sha256=_sha("cache"),
        metric_cache_manifest_sha256=_sha("metric-cache-manifest"),
        raw_input_manifest_sha256=_sha("raw-input-manifest"),
        qualification_config_sha256=_sha("qualification-config"),
        runtime_config_sha256=_sha("runtime-config"),
        qualification_source_manifest_sha256=_sha("qualification-source"),
        runtime_source_manifest_sha256=_sha("runtime-source"),
        source_revision="synthetic-source-revision",
        proposal_schema_name="drive-jepa-32-fixed-synthetic",
        proposal_schema_sha256=_sha("proposal-schema"),
        simulator_name="navsim-v2-synthetic",
        simulator_sha256=_sha("simulator"),
        scorer_name="official-scorer-synthetic",
        scorer_sha256=_sha("scorer"),
        channel_order=TARGET_CHANNEL_ORDER,
    )


def _coverage():
    # Deliberately unsorted input; the coverage object canonicalizes it.
    return build_artifact_coverage(
        [
            CoverageEntry("log-b", 200, "token-d"),
            CoverageEntry("log-a", 100, "token-a"),
            CoverageEntry("log-b", 100, "token-c"),
            CoverageEntry("log-a", 100, "token-b"),
        ]
    )


def _record(token: str, time_us: int, value: float) -> FrozenTokenRecord:
    return FrozenTokenRecord(
        token=token,
        time_us=time_us,
        proposal_sha256=_sha("proposals/" + token),
        base=np.full((32,), value, dtype="<f4"),
        targets=np.full((32, 9), value + 0.1, dtype="<f4"),
        calibration=np.full((32, 41, 11), value + 0.2, dtype="<f8"),
        carry=np.full((32, 41, 11), value + 0.3, dtype="<f8"),
    )


def _chunks():
    return [
        FrozenArtifactChunk(
            "log-a",
            [
                _record("token-a", 100, 0.1),
                _record("token-b", 100, 0.2),
            ],
        ),
        FrozenArtifactChunk(
            "log-b",
            [
                _record("token-c", 100, 0.3),
                _record("token-d", 200, 0.4),
            ],
        ),
    ]


class FrozenArtifactLayoutTest(unittest.TestCase):
    def test_wire_dtype_is_exact_and_unpadded(self) -> None:
        self.assertEqual(RECORD_SIZE, 232_264)
        self.assertFalse(RECORD_DTYPE.isalignedstruct)
        self.assertEqual(RECORD_DTYPE.fields["token_sha256"][1], 0)
        self.assertEqual(RECORD_DTYPE.fields["time_us"][1], 32)
        self.assertEqual(RECORD_DTYPE.fields["proposal_sha256"][1], 40)
        self.assertEqual(RECORD_DTYPE.fields["base"][1], 72)
        self.assertEqual(RECORD_DTYPE.fields["targets"][1], 200)
        self.assertEqual(RECORD_DTYPE.fields["calibration"][1], 1_352)
        self.assertEqual(RECORD_DTYPE.fields["carry"][1], 116_808)

    def test_bindings_and_coverage_reject_contract_drift(self) -> None:
        with self.assertRaisesRegex(FrozenArtifactIOError, "channel_order"):
            replace(_bindings(), channel_order=tuple(reversed(TARGET_CHANNEL_ORDER)))
        with self.assertRaisesRegex(FrozenArtifactIOError, "lowercase hexadecimal"):
            replace(
                _bindings(),
                causal_manifest_sha256=_bindings().causal_manifest_sha256.upper(),
            )
        with self.assertRaisesRegex(FrozenArtifactIOError, "lowercase hexadecimal"):
            replace(
                _bindings(),
                raw_input_manifest_sha256=(
                    _bindings().raw_input_manifest_sha256.upper()
                ),
            )
        with self.assertRaisesRegex(FrozenArtifactIOError, "duplicate tokens"):
            build_artifact_coverage(
                [
                    CoverageEntry("log-a", 1, "same"),
                    CoverageEntry("log-a", 2, "same"),
                ]
            )
        with self.assertRaisesRegex(FrozenArtifactIOError, "at least one token"):
            build_artifact_coverage([])


class FrozenArtifactRoundTripTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "f2-artifact.bin"
        self.bindings = _bindings()
        self.coverage = _coverage()

    def tearDown(self) -> None:
        gc.collect()
        self.temporary.cleanup()

    def _seal(self, path: Path | None = None):
        return seal_frozen_artifact(
            self.path if path is None else path,
            bindings=self.bindings,
            coverage=self.coverage,
            chunks=_chunks(),
        )

    def test_partial_is_verified_then_atomically_published(self) -> None:
        descriptor = self._seal()
        self.assertTrue(self.path.is_file())
        self.assertFalse(Path(str(self.path) + ".partial").exists())
        self.assertEqual(descriptor.path, self.path)
        self.assertEqual(descriptor.record_count, 4)
        self.assertEqual(
            tuple(chunk.log_name for chunk in descriptor.chunks),
            ("log-a", "log-b"),
        )
        self.assertEqual(
            sum(chunk.byte_size for chunk in descriptor.chunks),
            4 * RECORD_SIZE,
        )
        self.assertEqual(len(descriptor.root_sha256), 64)

        independently_verified = verify_frozen_artifact(
            self.path,
            expected_bindings=self.bindings,
            expected_coverage=self.coverage,
        )
        self.assertEqual(independently_verified.root_sha256, descriptor.root_sha256)
        self.assertEqual(independently_verified.file_size, self.path.stat().st_size)

    def test_stream_and_mmap_readers_return_identical_fixed_views(self) -> None:
        self._seal()
        observed = {}
        with FrozenArtifactReader.open(
            self.path,
            expected_bindings=self.bindings,
            expected_coverage=self.coverage,
            mode="stream",
        ) as reader:
            for record in reader.iter_records(batch_records=1):
                self.assertFalse(record.base.flags.writeable)
                self.assertEqual(record.base.shape, (32,))
                self.assertEqual(record.targets.shape, (32, 9))
                self.assertEqual(record.calibration.shape, (32, 41, 11))
                self.assertEqual(record.carry.shape, (32, 41, 11))
                observed[record.token] = (
                    record.log_name,
                    record.time_us,
                    float(record.base[0]),
                    record.proposal_sha256,
                )
        self.assertEqual(tuple(observed), ("token-a", "token-b", "token-c", "token-d"))
        self.assertEqual(observed["token-d"][:2], ("log-b", 200))

        mmap_rows = []
        reader = FrozenArtifactReader.open(
            self.path,
            expected_bindings=self.bindings,
            expected_coverage=self.coverage,
            mode="mmap",
        )
        for record in reader.iter_records(log_name="log-a", batch_records=2):
            mmap_rows.append((record.token, float(record.base[0])))
        # Release the last zero-copy record view before closing the mmap.
        del record
        gc.collect()
        reader.close()
        self.assertEqual([name for name, _ in mmap_rows], ["token-a", "token-b"])
        self.assertAlmostEqual(mmap_rows[1][1], observed["token-b"][2], places=7)

    def test_canonical_root_is_deterministic(self) -> None:
        first = self._seal()
        second = self._seal(self.root / "second.bin")
        self.assertEqual(first.root_sha256, second.root_sha256)
        self.assertEqual(self.path.read_bytes(), (self.root / "second.bin").read_bytes())

    def test_existing_target_is_never_overwritten(self) -> None:
        original = self._seal()
        original_bytes = self.path.read_bytes()
        with self.assertRaisesRegex(FrozenArtifactIOError, "already exists"):
            self._seal()
        self.assertEqual(self.path.read_bytes(), original_bytes)
        verified = verify_frozen_artifact(
            self.path,
            expected_bindings=self.bindings,
            expected_coverage=self.coverage,
        )
        self.assertEqual(verified.root_sha256, original.root_sha256)


class FrozenArtifactFailureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.bindings = _bindings()
        self.coverage = _coverage()

    def tearDown(self) -> None:
        gc.collect()
        self.temporary.cleanup()

    def _seal_valid(self, name: str = "valid.bin") -> Path:
        path = self.root / name
        seal_frozen_artifact(
            path,
            bindings=self.bindings,
            coverage=self.coverage,
            chunks=_chunks(),
        )
        return path

    def test_wrong_dtype_shape_and_nonfinite_values_fail_closed(self) -> None:
        invalid_records = [
            replace(_record("token-a", 100, 0.1), base=np.zeros((32,), dtype="<f8")),
            replace(_record("token-a", 100, 0.1), targets=np.zeros((32, 8), dtype="<f4")),
            replace(
                _record("token-a", 100, 0.1),
                calibration=np.full((32, 41, 11), np.nan, dtype="<f8"),
            ),
            replace(
                _record("token-a", 100, 0.1),
                targets=np.full((32, 9), 1.01, dtype="<f4"),
            ),
        ]
        messages = ("dtype", "shape", "NaN", "closed interval")
        for index, (invalid, message) in enumerate(zip(invalid_records, messages)):
            path = self.root / f"invalid-{index}.bin"
            chunks = [
                FrozenArtifactChunk("log-a", [invalid, _record("token-b", 100, 0.2)]),
                FrozenArtifactChunk(
                    "log-b",
                    [_record("token-c", 100, 0.3), _record("token-d", 200, 0.4)],
                ),
            ]
            with self.assertRaisesRegex(FrozenArtifactIOError, message):
                seal_frozen_artifact(
                    path,
                    bindings=self.bindings,
                    coverage=self.coverage,
                    chunks=chunks,
                )
            self.assertFalse(path.exists())
            self.assertTrue(Path(str(path) + ".partial").exists())

    def test_unsorted_missing_and_wrong_log_chunks_never_seal(self) -> None:
        cases = [
            (
                "unsorted.bin",
                [
                    FrozenArtifactChunk(
                        "log-a",
                        [_record("token-b", 100, 0.2), _record("token-a", 100, 0.1)],
                    ),
                    _chunks()[1],
                ],
                "order|coverage mismatch",
            ),
            ("missing.bin", [_chunks()[0]], "missing log chunks"),
            (
                "wrong-log.bin",
                [FrozenArtifactChunk("log-b", _chunks()[1].records), _chunks()[0]],
                "log order/coverage mismatch",
            ),
        ]
        for name, chunks, message in cases:
            path = self.root / name
            with self.assertRaisesRegex(FrozenArtifactIOError, message):
                seal_frozen_artifact(
                    path,
                    bindings=self.bindings,
                    coverage=self.coverage,
                    chunks=chunks,
                )
            self.assertFalse(path.exists())

    def test_exact_size_chunk_hash_root_and_external_bindings_are_checked(self) -> None:
        valid = self._seal_valid()

        trailing = self.root / "trailing.bin"
        shutil.copyfile(valid, trailing)
        with trailing.open("ab") as stream:
            stream.write(b"unexpected")
        with self.assertRaisesRegex(FrozenArtifactIOError, "byte size|magic"):
            verify_frozen_artifact(
                trailing,
                expected_bindings=self.bindings,
                expected_coverage=self.coverage,
            )

        tampered_chunk = self.root / "tampered-chunk.bin"
        shutil.copyfile(valid, tampered_chunk)
        with tampered_chunk.open("r+b") as stream:
            stream.seek(100)
            original = stream.read(1)
            stream.seek(100)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(FrozenArtifactIOError, "SHA-256 mismatch|coverage"):
            verify_frozen_artifact(
                tampered_chunk,
                expected_bindings=self.bindings,
                expected_coverage=self.coverage,
            )

        tampered_header = self.root / "tampered-header.bin"
        shutil.copyfile(valid, tampered_header)
        descriptor = verify_frozen_artifact(
            valid,
            expected_bindings=self.bindings,
            expected_coverage=self.coverage,
        )
        with tampered_header.open("r+b") as stream:
            stream.seek(descriptor.header_offset + 10)
            original = stream.read(1)
            stream.seek(descriptor.header_offset + 10)
            stream.write(bytes([original[0] ^ 1]))
        with self.assertRaisesRegex(FrozenArtifactIOError, "root SHA-256"):
            verify_frozen_artifact(
                tampered_header,
                expected_bindings=self.bindings,
                expected_coverage=self.coverage,
            )

        with self.assertRaisesRegex(FrozenArtifactIOError, "external bindings"):
            verify_frozen_artifact(
                valid,
                expected_bindings=replace(
                    self.bindings, baseline_checkpoint_sha256=_sha("other")
                ),
                expected_coverage=self.coverage,
            )
        different_coverage = build_artifact_coverage(
            [
                CoverageEntry("log-a", 100, "token-a"),
                CoverageEntry("log-a", 100, "token-b"),
                CoverageEntry("log-b", 100, "token-c"),
                CoverageEntry("log-b", 201, "token-d"),
            ]
        )
        with self.assertRaisesRegex(FrozenArtifactIOError, "coverage SHA-256"):
            verify_frozen_artifact(
                valid,
                expected_bindings=self.bindings,
                expected_coverage=different_coverage,
            )

    def test_reader_rejects_unknown_mode_log_and_nonpositive_batch(self) -> None:
        valid = self._seal_valid()
        with self.assertRaisesRegex(FrozenArtifactIOError, "mode"):
            FrozenArtifactReader.open(
                valid,
                expected_bindings=self.bindings,
                expected_coverage=self.coverage,
                mode="pickle",
            )
        with FrozenArtifactReader.open(
            valid,
            expected_bindings=self.bindings,
            expected_coverage=self.coverage,
            mode="stream",
        ) as reader:
            with self.assertRaisesRegex(FrozenArtifactIOError, "positive"):
                list(reader.iter_batches(batch_records=0))
            with self.assertRaisesRegex(FrozenArtifactIOError, "unknown log"):
                list(reader.iter_batches(log_name="absent"))

    def test_unsupported_atomic_publication_leaves_only_verified_partial(self) -> None:
        path = self.root / "unsupported-link.bin"
        with mock.patch(
            "navsim.planning.validation.f2_frozen_artifact_io.os.link",
            side_effect=OSError("synthetic filesystem without hard links"),
        ):
            with self.assertRaisesRegex(FrozenArtifactIOError, "hard-link"):
                seal_frozen_artifact(
                    path,
                    bindings=self.bindings,
                    coverage=self.coverage,
                    chunks=_chunks(),
                )
        self.assertFalse(path.exists())
        partial = Path(str(path) + ".partial")
        self.assertTrue(partial.is_file())
        verified = verify_frozen_artifact(
            partial,
            expected_bindings=self.bindings,
            expected_coverage=self.coverage,
        )
        self.assertEqual(verified.record_count, len(self.coverage.entries))


if __name__ == "__main__":
    unittest.main()
