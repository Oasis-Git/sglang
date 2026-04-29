"""Factory: ``cuda_graph_mode`` → ``BaseCudaGraphBackend`` instance.

Centralizes per-phase backend resolution so platform overrides (NPU,
out-of-tree) and future backend additions can plug in without
modifying the runner files. Phase / backend identifiers used here
live in :mod:`.cuda_graph_mode`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.model_executor.cuda_graph_backend.base_cudagraph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.breakable_cudagraph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_mode import (  # noqa: F401
    ALL_PHASES,
    ALLOWED_BACKENDS_PER_PHASE,
    BACKEND_BREAKABLE,
    BACKEND_DISABLED,
    BACKEND_FULL,
    BACKEND_TCPIECEWISE,
    DEFAULT_CUDA_GRAPH_MODE,
    PHASE_DECODE,
    PHASE_PREFILL,
)
from sglang.srt.model_executor.cuda_graph_backend.full_cudagraph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.tcpiecewise_cudagraph_backend import (
    TCPiecewiseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Track first occurrence of each fallback warning to avoid log spam.
_TCPIECEWISE_DECODE_FALLBACK_LOGGED = False


def resolve_decode_backend(model_runner: "ModelRunner") -> BaseCudaGraphBackend:
    """Pick a backend instance from ``cuda_graph_mode['decode']``.

    NPU device returns ``NPUCudaGraphBackend`` regardless of mode (only
    the Full-style backend is wired for NPU today).
    """
    mode = model_runner.server_args.cuda_graph_mode or {}
    backend_name = mode.get(PHASE_DECODE, BACKEND_FULL)

    enable_memory_saver = model_runner.server_args.enable_memory_saver

    if model_runner.device == "npu":
        from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
            NPUCudaGraphBackend,
        )

        return NPUCudaGraphBackend(enable_memory_saver=enable_memory_saver)

    if backend_name == BACKEND_BREAKABLE:
        return BreakableCudaGraphBackend(
            enable_memory_saver=enable_memory_saver,
            debug_eager=model_runner.server_args.debug_cuda_graph,
        )
    if backend_name == BACKEND_TCPIECEWISE:
        global _TCPIECEWISE_DECODE_FALLBACK_LOGGED
        if not _TCPIECEWISE_DECODE_FALLBACK_LOGGED:
            logger.warning(
                "cuda_graph_mode decode='tcpiecewise' is not yet implemented; "
                "falling back to 'full'."
            )
            _TCPIECEWISE_DECODE_FALLBACK_LOGGED = True
    return FullCudaGraphBackend(enable_memory_saver=enable_memory_saver)


def resolve_prefill_backend(model_runner: "ModelRunner") -> BaseCudaGraphBackend:
    """Pick a backend instance from ``cuda_graph_mode['prefill']``."""
    mode = model_runner.server_args.cuda_graph_mode or {}
    backend_name = mode.get(PHASE_PREFILL, BACKEND_TCPIECEWISE)

    if backend_name == BACKEND_BREAKABLE:
        return BreakableCudaGraphBackend(
            enable_memory_saver=model_runner.server_args.enable_memory_saver,
            debug_eager=model_runner.server_args.debug_cuda_graph,
        )
    # Default: tcpiecewise. ``(prefill, full)`` is rejected at config validation.
    return TCPiecewiseCudaGraphBackend()
