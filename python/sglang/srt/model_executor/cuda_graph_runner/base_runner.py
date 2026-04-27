"""BaseCudaGraphRunner — phase-agnostic orchestration of capture and replay.

Concrete subclasses (``PrefillCudaGraphRunner``, ``DecodeCudaGraphRunner``)
own phase-specific concerns: forward mode, bucket axis, buffer layout,
phase-specific ``can_run`` checks. The backend (a ``BaseCudaGraphBackend``)
owns the capture-mechanism-specific concerns: graph object, capture
context, replay primitive.

Phase 0 placeholder. Implementation lands in Phase 3.
"""

from __future__ import annotations


class BaseCudaGraphRunner:
    """Phase-agnostic CUDA graph runner skeleton."""

    pass
