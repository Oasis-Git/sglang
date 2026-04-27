"""FullCudaGraphBackend — captures the entire model forward as one
``torch.cuda.CUDAGraph`` per shape.

Phase 2b — minimal extraction:
  - ``make_graph()`` and ``capture_into(...)`` provide the primitive
    capture machinery as a backend-owned, runner-coupling-free pair.
  - ``legacy.CudaGraphRunner`` delegates to them on the FULL path while
    keeping its existing env-var-driven BCG path inline.
  - The ABC methods (``prepare`` / ``capture_one`` / ``replay``) stay
    NotImplementedError — the runner does not yet drive the backend
    through the abstract interface; that integration lands in Phase 3
    when runners are unified and the dict-based dispatch moves into
    the backend.

Lifted from ``cuda_graph_runner/legacy.py`` ``_create_device_graph`` and
the FULL branch of ``_capture_graph`` (around lines 910–948 at the time
of the cg-refactor).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
        BaseCudaGraphRunner,
    )


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One ``torch.cuda.CUDAGraph`` per shape; attention metadata captured
    inside the graph.

    Phase 2b exposes the *primitive* operations
    (``make_graph`` / ``capture_into``) that the legacy
    ``CudaGraphRunner`` delegates to. Phase 3 will rewire the runner to
    drive this backend through the abstract ``BaseCudaGraphBackend``
    interface (``prepare``/``capture_one``/``replay``).
    """

    captures_attn_metadata = True

    # ------------------------------------------------------------------
    # Phase 2b primitives — direct, no runner-coupling.
    # ------------------------------------------------------------------

    @staticmethod
    def make_graph() -> torch.cuda.CUDAGraph:
        """Allocate an empty CUDA graph object. Mirrors the FULL branch
        of ``CudaGraphRunner._create_device_graph``.
        """
        return torch.cuda.CUDAGraph()

    @staticmethod
    def capture_into(
        graph: torch.cuda.CUDAGraph,
        pool: Any,
        stream: torch.cuda.Stream,
        device_module: Any,
        memory_saver_adapter: Optional[Any],
        run_once_fn: Callable[[], Any],
    ) -> Any:
        """Run ``run_once_fn`` under a CUDA graph capture context bound to
        ``graph``. Returns whatever ``run_once_fn`` returned.

        Mirrors the FULL branch of ``CudaGraphRunner._capture_graph`` —
        memory-saver adapter wrapping when enabled, else
        ``device_module.graph(...)``.
        """
        graph_ctx: Callable[..., AbstractContextManager]
        if memory_saver_adapter is not None and memory_saver_adapter.enabled:
            graph_ctx = partial(
                memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = device_module.graph

        with graph_ctx(cuda_graph=graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    # ------------------------------------------------------------------
    # Abstract interface — wired up in Phase 3.
    # ------------------------------------------------------------------

    def prepare(self, runner: "BaseCudaGraphRunner") -> None:
        raise NotImplementedError(
            "FullCudaGraphBackend.prepare lands in Phase 3 (runner unification)"
        )

    def capture_one(
        self, shape_key: int, forward_fn: Callable[[], Any]
    ) -> None:
        raise NotImplementedError(
            "FullCudaGraphBackend.capture_one lands in Phase 3 (runner unification)"
        )

    def replay(self, shape_key: int) -> None:
        raise NotImplementedError(
            "FullCudaGraphBackend.replay lands in Phase 3 (runner unification)"
        )
