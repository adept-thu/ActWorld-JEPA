"""CPU-only adapter tests; toy trees, not task-model fitting or evaluation."""
import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from navsim.agents.actworld_jepa.inference.fast_hist import FastHist, FastForest


class NativeHistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(42)
        cls.x = rng.normal(size=(160, 6)).astype(np.float32)
        cls.x[::13, 1] = np.nan
        y = np.nan_to_num(cls.x[:, 1]) + cls.x[:, 0] * 0.7
        cls.reg = HistGradientBoostingRegressor(max_iter=17, max_leaf_nodes=9, random_state=0).fit(cls.x, y)
        cls.clf = HistGradientBoostingClassifier(max_iter=17, max_leaf_nodes=9, random_state=0).fit(cls.x, y > 0)

    def test_regression_float32_float64_missing_and_batch_one(self):
        fast = FastHist(self.reg)
        for x in (self.x, self.x.astype(np.float64), self.x[:1], self.x[::2]):
            np.testing.assert_array_equal(fast.predict(x), self.reg.predict(x))

    def test_classification(self):
        fast = FastHist(self.clf)
        np.testing.assert_array_equal(fast.predict_proba(self.x), self.clf.predict_proba(self.x))
        np.testing.assert_array_equal(fast.predict(self.x), self.clf.predict(self.x))

    def test_exact_threshold_neighbors(self):
        fast = FastHist(self.reg)
        for tree in self.reg._predictors[:3]:
            node = tree[0].nodes[0]
            threshold, feature = node["num_threshold"], node["feature_idx"]
            x = np.zeros((3, self.x.shape[1]), dtype=np.float64)
            x[:, feature] = [np.nextafter(threshold, -np.inf), threshold, np.nextafter(threshold, np.inf)]
            np.testing.assert_array_equal(fast.predict(x), self.reg.predict(x))

    def test_shape_guard(self):
        with self.assertRaises(ValueError):
            FastHist(self.reg).predict(np.zeros((1, 7)))

    def test_forests_float32_conversion(self):
        x = np.nan_to_num(self.x)
        for cls in (ExtraTreesRegressor, RandomForestRegressor):
            forest = cls(n_estimators=13, max_depth=7, n_jobs=1, random_state=0).fit(x, x[:, 0] + x[:, 1])
            fast = FastForest(forest)
            for values in (x, x.astype(np.float64) + 1e-12, x[:1], x[::2]):
                np.testing.assert_array_equal(fast.predict(values), forest.predict(values))



# Lightweight contract tests do not load task weights or allocate a CUDA device.
import ast
import copy
import hashlib
import os
from pathlib import Path
import pytest
import torch
from navsim.agents.actworld_jepa.inference.cuda_graph_runtime import GraphCall, staged_constants
from navsim.agents.actworld_jepa.inference.fast_hist import install_tree_adapters, library
from navsim.agents.actworld_jepa.inference.source_contract import validate_source

ROOT = Path(__file__).resolve().parents[1]
PLANNER = ROOT / "navsim/navsim/agents/actworld_jepa/actworld_planner.py"


def test_public_planner_source_and_inference_ast():
    assert validate_source(PLANNER) == "0cecff239883df7ca86d1a0ad8daa3a402be39f6996af9f84eb4f5442aeb66e8"
    source = PLANNER.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActWorldPlanner")
    methods = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    # Exact method text extracted using the audited deployment's AST. Source
    # text avoids ast.dump differences between Python 3.9 and newer versions.
    expected = {
        "forward": "38c70b72c8784bb2b63e608cec61635a1cf245e3061457dd06dac8124b0b7118",
        "_post_r94_selection": "e35ecfa4d5a16cae30b2e224354aa2b49342cec4c0889c5e918e70418703e21c",
    }
    for name, digest in expected.items():
        assert hashlib.sha256(ast.get_source_segment(source, methods[name]).encode()).hexdigest() == digest


def test_unknown_source_rejected(tmp_path):
    source = tmp_path / "unknown.py"
    source.write_text("# not the audited planner\n")
    with pytest.raises(RuntimeError, match="Unreviewed"):
        validate_source(source)


def test_kernel_packaging_and_lazy_library():
    import setuptools
    assert "navsim.agents.actworld_jepa.inference" in setuptools.find_packages(str(ROOT / "navsim"))
    package = ROOT / "navsim/navsim/agents/actworld_jepa/inference"
    assert (package / "fast_hist.c").is_file()
    assert "fast_hist.c" in (ROOT / "navsim/setup.py").read_text()
    assert library() is library()
    assert not (package / "libfast_hist_kernel.so").exists()


@pytest.fixture(scope="module")
def toy_models():
    NativeHistTest.setUpClass()
    return NativeHistTest


def test_unsupported_trees_rejected(toy_models):
    model = copy.deepcopy(toy_models.reg)
    model._predictors[0][0].nodes["is_categorical"][0] = 1
    with pytest.raises(ValueError, match="Categorical"):
        FastHist(model)
    model = copy.deepcopy(toy_models.reg)
    model.n_trees_per_iteration_ = 2
    with pytest.raises(ValueError, match="scalar"):
        FastHist(model)


def test_shared_estimator_adapted_once(toy_models):
    class Planner:
        pass
    planner = Planner()
    estimator = copy.deepcopy(toy_models.reg)
    planner.tree_meta_models = {"a": estimator, "b": [estimator]}
    planner.post_r94_selector_model = estimator
    adapters = install_tree_adapters(planner, validation_inputs=1)
    assert len(adapters) == 1
    assert planner.tree_meta_models["a"] is planner.tree_meta_models["b"][0]
    assert planner.post_r94_selector_model is adapters[0]
    np.testing.assert_array_equal(adapters[0].predict(toy_models.x), estimator.predict(toy_models.x))
    assert adapters[0].validated == 1


def test_graph_keeps_grad_path_eager_and_rejects_cpu_capture():
    function = GraphCall(lambda x: x * 2, "toy")
    x = torch.tensor([1.0], requires_grad=True)
    function(x).sum().backward()
    assert x.grad.item() == 2
    assert not function.graphs
    with torch.no_grad(), pytest.raises(ValueError, match="CPU tensor"):
        function(x)


def test_staging_restores_torch_functions_on_error():
    originals = (torch.tensor, torch.as_tensor, torch.Tensor.new_tensor)
    with pytest.raises(RuntimeError, match="intentional"):
        with staged_constants():
            assert torch.tensor([1]).item() == 1
            raise RuntimeError("intentional")
    assert (torch.tensor, torch.as_tensor, torch.Tensor.new_tensor) == originals


def test_prepare_rejects_training_or_empty_validation():
    from types import SimpleNamespace
    from navsim.agents.actworld_jepa.inference import prepare
    saved = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with pytest.raises(ValueError, match="eval agent"):
            prepare(SimpleNamespace(training=True), [object()])
        with pytest.raises(ValueError, match="real validation"):
            prepare(SimpleNamespace(training=False), [])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


def test_prepare_rejects_matmul_tf32():
    from navsim.agents.actworld_jepa.inference import prepare
    saved = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        with pytest.raises(RuntimeError, match="matmul TF32"):
            prepare(None, [])
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


@pytest.mark.skipif(os.environ.get("ACTWORLD_TEST_CUDA") != "1",
                    reason="GPU smoke requires explicit ACTWORLD_TEST_CUDA=1")
def test_cuda_graph_uses_fresh_inputs_and_retains_outputs():
    function = GraphCall(lambda x: {"y": x * 2 + 1}, "toy")
    with torch.inference_mode():
        first = function(torch.tensor([1.0, 2.0], device="cuda"))["y"]
        saved = first.clone()
        second = function(torch.tensor([4.0, 8.0], device="cuda"))["y"]
        assert torch.equal(second, torch.tensor([9.0, 17.0], device="cuda"))
        assert torch.equal(first, saved)
        different_shape = function(torch.tensor([10.0], device="cuda"))["y"]
        assert different_shape.item() == 21.0
        assert len(function.graphs) == 2
        assert function.replays == 3
