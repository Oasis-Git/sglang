"""TCPiecewiseCudaGraphBackend — torch.compile-based piecewise CUDA graph.

Uses ``CompilationConfig``, the FX/inductor pipeline from
``sglang.srt.compilation``, and the warmup-compile flag from
``compilation/compile_phase``. Produces piecewise graphs by FX-splitting
the model forward at attention layers; per-shape compiled callables
each internally capture sub-graphs via
``compilation/cuda_piecewise_backend``.

Phase 2a placeholder declaring the ABC contract. Real extraction lands
in Phase 2d — lifted from today's
``PiecewiseCudaGraphRunner.capture_one_batch_size`` and the surrounding
warmup machinery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
        BaseCudaGraphRunner,
    )


class TCPiecewiseCudaGraphBackend(BaseCudaGraphBackend):
    """torch.compile-driven piecewise capture; attention metadata
    recomputed at replay (outside the compiled callable's sub-graphs).
    """

    captures_attn_metadata = False

    def prepare(self, runner: "BaseCudaGraphRunner") -> None:
        raise NotImplementedError("TCPiecewiseCudaGraphBackend lands in Phase 2d")

    def capture_one(
        self, shape_key: int, forward_fn: Callable[[], Any]
    ) -> None:
        raise NotImplementedError("TCPiecewiseCudaGraphBackend lands in Phase 2d")

    def replay(self, shape_key: int) -> None:
        raise NotImplementedError("TCPiecewiseCudaGraphBackend lands in Phase 2d")
