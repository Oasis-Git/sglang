"""TCPiecewiseCudaGraphBackend — torch.compile-based piecewise CUDA graph.

Uses ``CompilationConfig``, the FX/inductor pipeline from
``sglang.srt.compilation``, and the warmup-compile flag from
``compilation/compile_phase``. Produces piecewise graphs by FX-splitting
the model forward at attention layers; per-shape compiled callables
each internally capture sub-graphs via
``compilation/cuda_piecewise_backend``.

Phase 2d — minimal extraction:
  - ``build_compilation_config(server_args)`` consolidates the
    ``CompilationConfig`` construction (compiler choice, debug mode,
    MoE A2A split-op adjustments).
  - ``install_compile(language_model, ...)`` wraps the language model
    with ``install_torch_compiled``.
  - ``PiecewiseCudaGraphRunner`` delegates to these primitives in its
    ``__init__`` and ``capture()``.
  - The ABC methods (``prepare`` / ``capture_one`` / ``replay``) stay
    NotImplementedError until Phase 3 (runner unification).

Lifted from ``model_executor/piecewise_cuda_graph_runner.py``
``__init__`` (lines 173–188 at the time of the cg-refactor) and the
``install_torch_compiled`` call site in ``capture()`` (line 295).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.layers.moe.utils import get_moe_a2a_backend
from sglang.srt.model_executor.cuda_graph_backend.base import (
    BaseCudaGraphBackend,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.cuda_graph_runner.base_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.server_args import ServerArgs


_VALID_COMPILERS = ("eager", "inductor")


class TCPiecewiseCudaGraphBackend(BaseCudaGraphBackend):
    """torch.compile-driven piecewise capture; attention metadata
    recomputed at replay (outside the compiled callable's sub-graphs).
    """

    captures_attn_metadata = False

    # ------------------------------------------------------------------
    # Phase 2d primitives — direct, no runner-coupling.
    # ------------------------------------------------------------------

    @staticmethod
    def build_compilation_config(server_args: "ServerArgs") -> CompilationConfig:
        """Construct the ``CompilationConfig`` from ``ServerArgs``.

        Mirrors the legacy ``PiecewiseCudaGraphRunner.__init__``
        sequence: validates the compiler choice, builds the config,
        adds the MoE A2A split-op when DeepEP/Mooncake is in use.
        """
        assert server_args.piecewise_cuda_graph_tokens is not None, (
            "piecewise_cuda_graph_tokens is not set"
        )
        assert server_args.piecewise_cuda_graph_compiler in _VALID_COMPILERS, (
            f"By now, only {_VALID_COMPILERS} are supported for piecewise "
            "cuda graph compiler."
        )

        config = CompilationConfig(
            server_args.piecewise_cuda_graph_tokens,
            server_args.piecewise_cuda_graph_compiler,
            server_args.enable_torch_compile_debug_mode,
        )

        if get_moe_a2a_backend().is_deepep() or get_moe_a2a_backend().is_mooncake():
            config.add_split_op("sglang.moe_forward_piecewise_cuda_graph_impl")

        return config

    @staticmethod
    def install_compile(
        language_model: Any,
        *,
        compile_config: CompilationConfig,
        graph_pool: Any,
        fullgraph: bool = True,
        dynamic_arg_dims: Optional[Any] = None,
    ) -> None:
        """Wrap ``language_model`` with ``torch.compile`` via
        ``install_torch_compiled``. Side effect: model's forward is
        replaced with the compiled trampoline.
        """
        install_torch_compiled(
            language_model,
            fullgraph=fullgraph,
            dynamic_arg_dims=dynamic_arg_dims,
            compile_config=compile_config,
            graph_pool=graph_pool,
        )

    # ------------------------------------------------------------------
    # Abstract interface — wired up in Phase 3.
    # ------------------------------------------------------------------

    def prepare(self, runner: "BaseCudaGraphRunner") -> None:
        raise NotImplementedError(
            "TCPiecewiseCudaGraphBackend.prepare lands in Phase 3 (runner unification)"
        )

    def capture_one(
        self, shape_key: int, forward_fn: Callable[[], Any]
    ) -> None:
        raise NotImplementedError(
            "TCPiecewiseCudaGraphBackend.capture_one lands in Phase 3 (runner unification)"
        )

    def replay(self, shape_key: int) -> None:
        raise NotImplementedError(
            "TCPiecewiseCudaGraphBackend.replay lands in Phase 3 (runner unification)"
        )
