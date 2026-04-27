"""BaseCudaGraphBackend — abstract capture-mechanism interface.

A backend owns *how* a captured artifact is produced and replayed for one
shape. It is **phase-agnostic** — the backend never sees a
``forward_mode``, only an opaque shape key (token count for prefill,
batch size for decode) supplied by the runner.

The runner owns:
  - building the dummy ``ForwardBatch`` for capture
  - populating the static buffers from the live ``ForwardBatch`` at replay
  - slicing the output back to ``raw_num_tokens``

The backend owns:
  - the underlying graph object (``torch.cuda.CUDAGraph``,
    ``BreakableCUDAGraph``, or torch.compile compiled callable)
  - the begin/end-capture mechanism
  - the replay primitive
  - whether attention metadata is set up inside or outside the captured
    region (``captures_attn_metadata`` flag)
  - any framework-specific one-time setup (e.g. tcpcg warmup compile)

Implementations land in Phases 2b–2d:
  - ``cuda_graph_backend.full.FullCudaGraphBackend`` — single-graph capture
  - ``cuda_graph_backend.breakable.BreakableCudaGraphBackend`` — segmented
  - ``cuda_graph_backend.tcpcg.TCPiecewiseCudaGraphBackend`` — torch.compile
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
        BaseCudaGraphRunner,
    )


class BaseCudaGraphBackend(ABC):
    """Abstract capture-mechanism backend.

    Concrete subclasses are owned 1:1 by a runner. The runner constructs
    the backend, calls :meth:`prepare` once, then iterates over shapes
    calling :meth:`capture_one`. At serve time the runner calls
    :meth:`replay` per request.
    """

    #: If ``True`` the backend captures attention metadata *inside* the
    #: graph: the runner should call
    #: ``attn_backend.init_forward_metadata_capture_cuda_graph`` before
    #: handing over the forward closure, and the captured graph is
    #: replayed bit-exactly per shape.
    #:
    #: If ``False`` the runner calls plain ``init_forward_metadata`` on
    #: each replay (between captured segments), and metadata is
    #: recomputed against the live ``ForwardBatch``. PCG/BCG style.
    captures_attn_metadata: bool = False

    @abstractmethod
    def prepare(self, runner: "BaseCudaGraphRunner") -> None:
        """One-time setup before any capture happens.

        Stash a back-reference to the runner if needed, configure
        compilation, kick off any framework warmup. Called exactly once
        per backend instance, before the first ``capture_one``.
        """

    @abstractmethod
    def capture_one(
        self,
        shape_key: int,
        forward_fn: Callable[[], Any],
    ) -> None:
        """Capture a single shape.

        Parameters
        ----------
        shape_key
            Opaque integer key — the runner uses token count for prefill,
            batch size for decode. The backend does not interpret it,
            only stores its captured artifact under this key.
        forward_fn
            A zero-arg closure produced by the runner. Calling it runs
            ``model.forward(dummy_forward_batch)`` against the runner's
            pre-allocated static buffers and returns the model output.
            The backend wraps this callable in whatever capture context
            it needs (``torch.cuda.graph(...)``,
            ``BreakableCUDAGraphCapture(...)``, or torch.compile).
        """

    @abstractmethod
    def replay(self, shape_key: int) -> None:
        """Replay the artifact previously captured at ``shape_key``.

        The runner has already populated the static input buffers from
        the live ``ForwardBatch``; the backend just executes the
        captured artifact, leaving output in the runner-owned output
        buffers. The runner then slices output back to ``raw_num_tokens``.
        """

    def cleanup(self) -> None:
        """Tear down any backend-specific resources. Default: no-op."""
