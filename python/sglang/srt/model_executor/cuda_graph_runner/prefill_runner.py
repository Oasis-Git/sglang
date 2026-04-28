"""PrefillCudaGraphRunner — runs the EXTEND phase under a pluggable backend.

Factory class: returns either a ``BreakableCudaGraphRunner`` or a
``PiecewiseCudaGraphRunner`` instance based on
``cuda_graph_mode["prefill"]``. The "disabled" branch is handled at the
model_runner level — if prefill is disabled, the factory is not
constructed.

The "full" backend is silently downgraded to "disabled" at config
resolution time (see ``config_resolution._downgrade_unsupported_combinations``)
because full CUDA graph capture only fits fixed-shape deployments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class PrefillCudaGraphRunner:
    """Factory for the prefill-phase CUDA graph runner.

    Returns an instance of one of the legacy runner classes today. The
    surface (``can_run`` / ``replay`` / ``capture_hidden_mode`` etc.)
    is whatever the legacy class exposes — call sites unchanged.
    """

    def __new__(cls, model_runner: "ModelRunner"):
        from sglang.srt.model_executor.cuda_graph_runner.config_resolution import (
            BACKEND_BREAKABLE,
            BACKEND_TCPCG,
            PHASE_PREFILL,
        )

        mode = model_runner.server_args.cuda_graph_mode or {}
        backend = mode.get(PHASE_PREFILL, BACKEND_TCPCG)

        # Late imports to avoid circular dependencies (these modules
        # transitively import from cuda_graph_runner).
        if backend == BACKEND_BREAKABLE:
            from sglang.srt.model_executor.breakable_cuda_graph_runner import (
                BreakableCudaGraphRunner,
            )

            return BreakableCudaGraphRunner(model_runner)

        # Default + tcpcg: torch.compile-based piecewise.
        from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
            PiecewiseCudaGraphRunner,
        )

        return PiecewiseCudaGraphRunner(model_runner)
