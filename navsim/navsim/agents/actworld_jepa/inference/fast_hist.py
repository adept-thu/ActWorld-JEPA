"""Exact numeric-split HistGradientBoosting inference without per-tree Python calls.

Original sklearn estimators are retained for validation and fallback. No fitting.
This is a CPU deployment adapter, not a new model or a distilled selector.
"""
import ctypes
from pathlib import Path
import subprocess
import tempfile
from functools import lru_cache
import numpy as np
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor


@lru_cache(maxsize=1)
def library():
    # Compile once per worker without writing into a potentially read-only
    # installation. CDLL retains the loaded mapping after the temp file is
    # removed. TMPDIR may point to the worker's writable scratch disk.
    source = Path(__file__).with_name("fast_hist.c")
    with tempfile.TemporaryDirectory(prefix="actworld-hist-") as folder:
        target = Path(folder) / "libfast_hist_kernel.so"
        subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
                        str(source), "-o", str(target)], check=True)
        lib = ctypes.CDLL(str(target))
    ptr = ctypes.c_void_p
    lib.hist_predict.argtypes = [ptr, ctypes.c_int64, ctypes.c_int64, ptr, ctypes.c_int64,
                                ptr, ptr, ptr, ptr, ptr, ptr, ptr, ctypes.c_double, ptr]
    lib.hist_predict.restype = None
    lib.hist_predict_packed.argtypes = [ptr, ctypes.c_int64, ctypes.c_int64,
                                      ptr, ctypes.c_int64, ptr, ctypes.c_double, ptr]
    lib.hist_predict_packed.restype = None
    return lib


class FastHist:
    def pack(self):
        dtype = np.dtype([("left", "u4"), ("right", "u4"), ("feature", "i4"),
                          ("leaf", "u1"), ("missing_left", "u1"), ("padding", "u2"),
                          ("threshold", "f8"), ("value", "f8")], align=True)
        assert dtype.itemsize == 32 and len(self.value) < 2 ** 32
        self.nodes = np.zeros(len(self.value), dtype=dtype)
        for name in ("left", "right", "feature", "leaf", "missing_left", "threshold", "value"):
            self.nodes[name] = getattr(self, name).astype(dtype.fields[name][0])

    def __init__(self, estimator, validate_calls=0):
        if not isinstance(estimator, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
            raise TypeError(type(estimator))
        if estimator.n_trees_per_iteration_ != 1:
            raise ValueError("Only scalar/binary HGB models supported")
        predictors = [group[0] for group in estimator._predictors]
        if any(np.any(tree.nodes["is_categorical"]) for tree in predictors):
            raise ValueError("Categorical trees must use original sklearn")
        self.original = estimator
        self.classifier = isinstance(estimator, HistGradientBoostingClassifier)
        if not self.classifier and type(estimator._loss.link).__name__ != "IdentityLink":
            raise ValueError("Nonidentity regression link must use original sklearn")
        self.n_features_in_ = estimator.n_features_in_
        self.classes_ = getattr(estimator, "classes_", None)
        self.baseline = float(estimator._baseline_prediction.reshape(-1)[0])
        sizes = [len(tree.nodes) for tree in predictors]
        self.roots = np.ascontiguousarray(np.cumsum([0] + sizes[:-1]), dtype=np.int64)
        def combine(name, dtype):
            return np.ascontiguousarray(np.concatenate([tree.nodes[name] for tree in predictors]), dtype=dtype)
        self.left = np.concatenate([tree.nodes["left"].astype(np.int64) + root for tree, root in zip(predictors, self.roots)])
        self.right = np.concatenate([tree.nodes["right"].astype(np.int64) + root for tree, root in zip(predictors, self.roots)])
        self.feature = combine("feature_idx", np.int64)
        self.threshold = combine("num_threshold", np.float64)
        self.value = combine("value", np.float64)
        self.leaf = combine("is_leaf", np.uint8)
        self.missing_left = combine("missing_go_to_left", np.uint8)
        self.pack()
        self.lib = library()
        self.validate_calls = validate_calls
        self.validated = 0
        self.max_error = 0.0

    def raw(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError("Feature shape mismatch")
        out = np.empty(x.shape[0], dtype=np.float64)
        self.lib.hist_predict_packed(x.ctypes.data, len(x), x.shape[1], self.roots.ctypes.data,
                                    len(self.roots), self.nodes.ctypes.data, self.baseline, out.ctypes.data)
        return out

    def check(self, x, result, method):
        if self.validated < self.validate_calls:
            expected = getattr(self.original, method)(x)
            error = float(np.max(np.abs(result - expected)))
            self.max_error = max(self.max_error, error)
            np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)
            self.validated += 1
        return result

    def predict_proba(self, x):
        if not self.classifier:
            raise TypeError("Not a classifier")
        p = expit(self.raw(x))
        return self.check(x, np.column_stack((1.0 - p, p)), "predict_proba")

    def predict(self, x):
        raw = self.raw(x)
        result = self.classes_[(raw > 0).astype(np.int64)] if self.classifier else raw
        return self.check(x, result, "predict")


class FastForest(FastHist):
    """Scalar sklearn forests: float32 inputs, double splits/sum, final mean."""
    def __init__(self, estimator, validate_calls=0):
        assert isinstance(estimator, (ExtraTreesRegressor, RandomForestRegressor))
        assert estimator.n_outputs_ == 1
        self.original = estimator
        self.classifier = False
        self.classes_ = None
        self.n_features_in_ = estimator.n_features_in_
        self.baseline = 0.0
        trees = [e.tree_ for e in estimator.estimators_]
        sizes = [t.node_count for t in trees]
        self.roots = np.ascontiguousarray(np.cumsum([0] + sizes[:-1]), dtype=np.int64)
        self.left = np.concatenate([t.children_left.astype(np.int64) + root for t, root in zip(trees, self.roots)])
        self.right = np.concatenate([t.children_right.astype(np.int64) + root for t, root in zip(trees, self.roots)])
        self.feature = np.ascontiguousarray(np.concatenate([t.feature for t in trees]), dtype=np.int64)
        self.threshold = np.ascontiguousarray(np.concatenate([t.threshold for t in trees]), dtype=np.float64)
        self.value = np.ascontiguousarray(np.concatenate([t.value.reshape(-1) for t in trees]), dtype=np.float64)
        self.leaf = np.ascontiguousarray(np.concatenate([t.children_left == -1 for t in trees]), dtype=np.uint8)
        self.missing_left = np.zeros_like(self.leaf)
        self.pack()
        self.lib = library()
        self.validate_calls, self.validated, self.max_error = validate_calls, 0, 0.0

    def predict(self, x):
        values = np.asarray(x, dtype=np.float32)
        if not np.isfinite(values).all():
            return self.original.predict(x)
        result = self.raw(values) / len(self.roots)
        return self.check(x, result, "predict")


def install_tree_adapters(planner, validation_inputs=32):
    adapters = []
    seen = {}
    def convert(item):
        if id(item) in seen:
            return seen[id(item)]
        seen[id(item)] = item
        if isinstance(item, dict):
            for k, v in list(item.items()):
                item[k] = convert(v)
            return item
        if isinstance(item, list):
            item[:] = [convert(v) for v in item]
            return item
        if isinstance(item, tuple):
            result = tuple(convert(v) for v in item)
            seen[id(item)] = result
            return result
        if isinstance(item, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
            adapter = FastHist(item, validate_calls=validation_inputs)
            adapters.append(adapter)
            seen[id(item)] = adapter
            return adapter
        if isinstance(item, (ExtraTreesRegressor, RandomForestRegressor)):
            adapter = FastForest(item, validate_calls=validation_inputs)
            adapters.append(adapter)
            seen[id(item)] = adapter
            return adapter
        # Explicit deployed wrapper attributes only; never traverse arbitrary
        # sklearn internals, tensors, modules, or file-system artifacts.
        module = type(item).__module__
        if hasattr(item, "__dict__") and (module.startswith("navsim.agents.actworld_jepa") or
                                          module.endswith(("_model", "_estimator"))):
            for name, value in list(vars(item).items()):
                if name not in ("_last_ref", "_last_outputs"):
                    setattr(item, name, convert(value))
        return item
    for name in ("tree_meta_models", "post_r94_selector_model"):
        planner.__dict__[name] = convert(planner.__dict__.get(name))
    return adapters
