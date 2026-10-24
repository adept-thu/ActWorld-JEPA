"""Single-request, fixed-shape FP32 CUDA graph adapter; no output memoization.

Every request copies fresh tensors and replays GPU computation. Outputs are
cloned to avoid a subsequent replay overwriting a caller's retained outputs.
Only wrap audited CUDA-only callables. CPU selectors must remain outside.
"""
import torch
from contextlib import contextmanager
from torch.utils._pytree import tree_flatten, tree_unflatten


@contextmanager
def staged_constants():
    """Stage small literal CUDA constants during warm-up, never during capture.

    Each use clones the staged tensor, retaining fresh-storage semantics. Keys
    include the complete literal and options. Not suitable for concurrent model
    initialization; this context exists only while creating a graph.
    """
    originals = (torch.tensor, torch.as_tensor, torch.Tensor.new_tensor)
    cache = {}
    def literal(value):
        if isinstance(value, (bool, int, float)):
            return (type(value).__name__, value)
        if isinstance(value, (tuple, list)) and len(value) <= 64:
            return tuple(literal(v) for v in value)
        raise TypeError("not a small literal")
    def staged(tag, original, data, args, kwargs):
        device = kwargs.get("device")
        if device is None or torch.device(device).type != "cuda":
            return original(data, *args, **kwargs)
        try:
            key = (tag, literal(data), repr(args), repr(sorted(kwargs.items())))
        except TypeError:
            return original(data, *args, **kwargs)
        if key not in cache:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("CUDA literal was not seen during warm-up")
            cache[key] = original(data, *args, **kwargs)
        return cache[key].clone()
    def tensor(data, *args, **kwargs):
        return staged("tensor", originals[0], data, args, kwargs)
    def as_tensor(data, *args, **kwargs):
        return staged("as_tensor", originals[1], data, args, kwargs)
    def new_tensor(self, data, *args, **kwargs):
        options = dict(kwargs)
        options.setdefault("device", self.device)
        options.setdefault("dtype", self.dtype)
        return staged("new_tensor", originals[0], data, args, options)
    torch.tensor, torch.as_tensor, torch.Tensor.new_tensor = tensor, as_tensor, new_tensor
    try:
        yield cache
    finally:
        torch.tensor, torch.as_tensor, torch.Tensor.new_tensor = originals


class GraphCall:
    def __init__(self, function, name):
        self.function, self.name = function, name
        self.graphs = {}
        self.replays = 0

    def __call__(self, *args, **kwargs):
        if torch.is_grad_enabled():
            return self.function(*args, **kwargs)
        leaves, spec = tree_flatten((args, kwargs))
        signature = []
        for value in leaves:
            if torch.is_tensor(value):
                if value.device.type != "cuda":
                    raise ValueError(self.name + ": CPU tensor cannot be captured")
                signature.append((tuple(value.shape), value.dtype, value.device))
            elif isinstance(value, (int, float, bool, str, type(None))):
                signature.append((type(value), value))
            else:
                raise TypeError(self.name + ": unsupported graph argument " + str(type(value)))
        key = (repr(spec), tuple(signature))
        if key not in self.graphs:
            if len(self.graphs) >= 8:
                return self.function(*args, **kwargs)
            static = [v.detach().clone() if torch.is_tensor(v) else v for v in leaves]
            with staged_constants() as constant_buffers:
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        static_args, static_kwargs = tree_unflatten(static, spec)
                        self.function(*static_args, **static_kwargs)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                static_args, static_kwargs = tree_unflatten(static, spec)
                with torch.cuda.graph(graph):
                    output = self.function(*static_args, **static_kwargs)
            output_leaves, output_spec = tree_flatten(output)
            if any(v is not None and (not torch.is_tensor(v) or v.device.type != "cuda") for v in output_leaves):
                raise ValueError(self.name + ": graph must return only CUDA tensors")
            self.graphs[key] = (graph, static, output_leaves, output_spec, constant_buffers)
        graph, static, output_leaves, output_spec, _ = self.graphs[key]
        with torch.inference_mode():
            for incoming, target in zip(leaves, static):
                if torch.is_tensor(incoming):
                    target.copy_(incoming)
            graph.replay()
            self.replays += 1
            return tree_unflatten([v.clone() if torch.is_tensor(v) else v for v in output_leaves], output_spec)
