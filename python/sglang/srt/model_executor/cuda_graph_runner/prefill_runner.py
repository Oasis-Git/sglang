"""PrefillCudaGraphRunner — runs the EXTEND phase under a pluggable backend.

Phase 3 — minimal naming + factory:
  - ``PrefillCudaGraphRunner`` is a factory that returns either a
    ``BreakableCudaGraphRunner`` or a ``PiecewiseCudaGraphRunner``
    based on ``ServerArgs.enable_breakable_cuda_graph``. Mirrors
    today's branch in ``model_runner.init_cuda_graphs``.
  - Phase 4 will drive the choice via the canonical
    ``cuda_graph_mode`` config (``"breakable"`` vs ``"tcpcg"`` vs
    ``"disabled"``); the factory bridges that gap.

The actual capture/replay machinery still lives in the legacy
``Piecewise`` / ``Breakable`` runner classes; this is a naming layer,
not a behavior change. Phase 3 (subsequent commits) will migrate the
bodies into a single ``PrefillCudaGraphRunner`` driven by a backend.
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
        # Phase 4a: drive selection from canonical ``cuda_graph_mode``
        # (resolved from legacy flags + JSON in
        # ``cuda_graph_runner.config_resolution._parse_canonical``).
        # The "disabled" branch is handled at the model_runner level —
        # if prefill is disabled, the factory is not constructed.
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
