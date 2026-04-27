"""DecodeCudaGraphRunner — runs the DECODE phase under a pluggable backend.

Also handles speculative variants (TARGET_VERIFY, DLLM_EXTEND) per the
plan's Q9 decision. Phase 0 placeholder. Implementation lands in Phase 3.
"""

from __future__ import annotations

from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
    BaseCudaGraphRunner,
)


class DecodeCudaGraphRunner(BaseCudaGraphRunner):
    """Captures and replays the decode forward pass."""

    pass
