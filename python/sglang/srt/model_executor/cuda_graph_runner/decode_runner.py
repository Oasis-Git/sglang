"""DecodeCudaGraphRunner — runs the DECODE phase under a pluggable backend.

Factory class: returns a ``CudaGraphRunner`` instance configured for the
requested decode backend. Backend selection comes from
``cuda_graph_mode["decode"]``:

  - ``"full"`` (default): standard ``torch.cuda.CUDAGraph`` per shape.
  - ``"breakable"``: experimental segmented capture; passed into
    ``CudaGraphRunner`` via ``use_breakable_capture=True``.
  - ``"tcpcg"``: not yet implemented for the decode phase; logs a
    one-shot warning and falls back to ``"full"`` so the server boots.

Speculative variants (``TARGET_VERIFY``, ``DLLM_EXTEND``) dispatch
through the same machinery, unchanged.
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

    The class is a factory rather than a subclass so that future backend
    variants can return distinct types without breaking
    ``isinstance(..., DecodeCudaGraphRunner)`` checks (none exist today).
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
                    "falling back to 'full'."
                )
                _TCPCG_DECODE_FALLBACK_LOGGED = True

        return CudaGraphRunner(
            model_runner, use_breakable_capture=use_breakable_capture
        )
