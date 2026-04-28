"""Factory: ``cuda_graph_mode`` → ``BaseCudaGraphBackend`` instance.

Centralizes per-phase backend resolution so platform overrides (NPU,
out-of-tree) and future backend additions can plug in without
modifying the runner files.

Phase / backend constants used both here and during the
``ServerArgs`` config-resolution pass live in this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.model_executor.cuda_graph_backend.base import BaseCudaGraphBackend
from sglang.srt.model_executor.cuda_graph_backend.breakable_cudagraph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.full_cudagraph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.cuda_graph_backend.tcpcg_cudagraph_backend import (
    TCPiecewiseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase / backend constants
# ---------------------------------------------------------------------------
PHASE_DECODE = "decode"
PHASE_PREFILL = "prefill"
ALL_PHASES = (PHASE_DECODE, PHASE_PREFILL)

BACKEND_FULL = "full"
BACKEND_BREAKABLE = "breakable"
BACKEND_TCPCG = "tcpcg"
BACKEND_DISABLED = "disabled"

ALLOWED_BACKENDS_PER_PHASE = {
    PHASE_DECODE: (BACKEND_FULL, BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
    # ``full`` is rejected for prefill — full CUDA graph capture only
    # fits fixed-shape and prefill is variable-shape. Use ``breakable``
    # or ``tcpcg`` for prefill.
    PHASE_PREFILL: (BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
}

DEFAULT_CUDA_GRAPH_MODE = {
    PHASE_DECODE: BACKEND_FULL,
    PHASE_PREFILL: BACKEND_TCPCG,
}

# Track first occurrence of each fallback warning to avoid log spam.
_TCPCG_DECODE_FALLBACK_LOGGED = False


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
    if backend_name == BACKEND_TCPCG:
        global _TCPCG_DECODE_FALLBACK_LOGGED
        if not _TCPCG_DECODE_FALLBACK_LOGGED:
            logger.warning(
                "cuda_graph_mode decode='tcpcg' is not yet implemented; "
                "falling back to 'full'."
            )
            _TCPCG_DECODE_FALLBACK_LOGGED = True
    return FullCudaGraphBackend(enable_memory_saver=enable_memory_saver)


def resolve_prefill_backend(model_runner: "ModelRunner") -> BaseCudaGraphBackend:
    """Pick a backend instance from ``cuda_graph_mode['prefill']``."""
    mode = model_runner.server_args.cuda_graph_mode or {}
    backend_name = mode.get(PHASE_PREFILL, BACKEND_TCPCG)

    if backend_name == BACKEND_BREAKABLE:
        return BreakableCudaGraphBackend(
            enable_memory_saver=model_runner.server_args.enable_memory_saver,
            debug_eager=model_runner.server_args.debug_cuda_graph,
        )
    # Default: tcpcg. ``(prefill, full)`` is rejected at config validation.
    return TCPiecewiseCudaGraphBackend()
