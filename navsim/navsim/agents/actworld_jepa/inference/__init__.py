"""Opt-in, equivalence-gated r383 FP32 inference acceleration."""


def prepare(agent, validation_inputs, *, use_graphs=True, use_cpu_post=True):
    """Prepare one dedicated CUDA inference worker using real NAVTRAIN inputs.

    See docs/NAVSIM_V1_INFERENCE_ACCELERATION.md before use. Installation may
    mutate the worker before a failure; discard the worker if it raises.
    """
    from .deploy_fast import prepare as prepare_runtime
    return prepare_runtime(agent, validation_inputs, use_graphs, use_cpu_post)
