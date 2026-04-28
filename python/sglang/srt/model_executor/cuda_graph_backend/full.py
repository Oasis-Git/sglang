"""FullCudaGraphBackend — captures the entire model forward as one
``torch.cuda.CUDAGraph`` per shape.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from functools import partial
from typing import Any, Callable, Optional

import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH


class FullCudaGraphBackend:
    """Single-graph capture: one ``torch.cuda.CUDAGraph`` per shape;
    attention metadata is captured inside the graph.
    """

    @staticmethod
    def make_graph() -> torch.cuda.CUDAGraph:
        """Allocate an empty CUDA graph object."""
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

        Memory-saver-aware: when the adapter is enabled, capture goes
        through its wrapper so the graph allocation is tagged correctly;
        otherwise the standard ``device_module.graph`` context is used.
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
