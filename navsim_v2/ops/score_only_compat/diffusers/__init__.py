"""Score-only compatibility shim for an optional training dependency.

The NAVSIM v2 score-only entrypoint imports the DP training module eagerly,
although it never constructs a diffusion scheduler while evaluating a merged
trajectory file.  Keep that unrelated optional dependency out of the scoring
environment and fail loudly if training code ever tries to use it.
"""


class DDPMScheduler:  # pragma: no cover - must never be instantiated in scoring
    def __init__(self, *args, **kwargs):
        raise RuntimeError("DDPMScheduler is unavailable in score-only mode")
