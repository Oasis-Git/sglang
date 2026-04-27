"""PrefillCudaGraphRunner — runs the EXTEND phase under a pluggable backend.

Phase 0 placeholder. Implementation lands in Phase 3.
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
    BaseCudaGraphRunner,
)


class PrefillCudaGraphRunner(BaseCudaGraphRunner):
    """Captures and replays the prefill (extend) forward pass."""

    pass
