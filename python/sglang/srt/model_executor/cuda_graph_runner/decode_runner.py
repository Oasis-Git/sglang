"""DecodeCudaGraphRunner — runs the DECODE phase under a pluggable backend.

Phase 3 — minimal renaming-with-inheritance:
  - ``DecodeCudaGraphRunner`` is a subclass of the legacy
    ``CudaGraphRunner``. It changes nothing structurally yet; it just
    establishes the canonical name new code should use.
  - The legacy class still exists at
    ``cuda_graph_runner.legacy.CudaGraphRunner`` and is re-exported
    from the package for backwards compatibility (eagle workers,
    speculative paths, hardware stubs).

Phase 3 (subsequent commits) will migrate the body of
``CudaGraphRunner`` into this class and have ``legacy.CudaGraphRunner``
become a shim. Phase 4+ exposes per-phase backend selection.

Speculative variants (``TARGET_VERIFY``, ``DLLM_EXTEND``) per the plan's
Q9 are dispatched here through the same ``CudaGraphRunner`` machinery —
no behavior change.
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_runner.legacy import CudaGraphRunner


class DecodeCudaGraphRunner(CudaGraphRunner):
    """Captures and replays the decode forward pass.

    Phase 3 placeholder: identical behavior to ``CudaGraphRunner`` via
    inheritance. The class exists so that ``model_runner`` and other
    upstream call sites can switch to the canonical name while the
    legacy class continues to host the implementation.
    """

    pass
