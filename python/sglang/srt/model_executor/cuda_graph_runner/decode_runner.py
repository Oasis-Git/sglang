"""DecodeCudaGraphRunner — runs the DECODE phase under a pluggable backend.

Phase 5 turns this into a factory like ``PrefillCudaGraphRunner``,
driving backend selection off ``cuda_graph_mode["decode"]``:

  - ``"full"`` (default): returns a ``CudaGraphRunner`` instance, which
    captures one ``torch.cuda.CUDAGraph`` per shape.
  - ``"breakable"``: experimental; bridges to today's
    ``SGLANG_USE_BREAKABLE_CUDA_GRAPH`` env-var path inside
    ``CudaGraphRunner._capture_graph`` /
    ``_create_device_graph``. Sets the env var if not already set.
  - ``"tcpcg"``: not implemented for decode in v1; falls back to
    ``"full"`` with a one-time warning so the user knows their config
    didn't take effect.

Phase 3b/c will migrate the body of ``CudaGraphRunner`` into a true
runner-with-backend split. Until then, this factory is a thin
naming + bridging layer — speculative variants (``TARGET_VERIFY``,
``DLLM_EXTEND``) per plan Q9 dispatch through the same machinery,
unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.model_executor.cuda_graph_runner.legacy import CudaGraphRunner

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_TCPCG_DECODE_FALLBACK_LOGGED = False


class DecodeCudaGraphRunner:
    """Factory for the decode-phase CUDA graph runner.

    Returns a ``CudaGraphRunner`` instance configured for the requested
    decode backend. The class is a factory rather than a subclass so
    that future backend variants can return distinct types without
    breaking ``isinstance(..., DecodeCudaGraphRunner)`` checks
    (none exist today).
    """

    def __new__(cls, model_runner: "ModelRunner") -> CudaGraphRunner:
        from sglang.srt.model_executor.cuda_graph_runner.config_resolution import (
            BACKEND_BREAKABLE,
            BACKEND_FULL,
            BACKEND_TCPCG,
            PHASE_DECODE,
        )

        mode = model_runner.server_args.cuda_graph_mode or {}
        backend = mode.get(PHASE_DECODE, BACKEND_FULL)

        use_breakable_capture = backend == BACKEND_BREAKABLE
        if backend == BACKEND_TCPCG:
            global _TCPCG_DECODE_FALLBACK_LOGGED
            if not _TCPCG_DECODE_FALLBACK_LOGGED:
                logger.warning(
                    "cuda_graph_mode decode='tcpcg' is not yet implemented; "
                    "falling back to 'full'. Track the follow-up that wires "
                    "tcpcg into the decode runner in refactor/progress.md."
                )
                _TCPCG_DECODE_FALLBACK_LOGGED = True

        return CudaGraphRunner(
            model_runner, use_breakable_capture=use_breakable_capture
        )
