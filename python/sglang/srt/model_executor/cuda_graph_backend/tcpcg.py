"""TCPiecewiseCudaGraphBackend — torch.compile-based piecewise CUDA graph.

Uses ``CompilationConfig``, the FX/inductor pipeline from
``sglang.srt.compilation``, and the warmup-compile flag from
``compilation/compile_phase``. Produces piecewise graphs by FX-splitting
the model forward at attention layers; per-shape compiled callables
each internally capture sub-graphs via
``compilation/cuda_piecewise_backend``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.compilation.compilation_config import CompilationConfig
from sglang.srt.compilation.compile import install_torch_compiled
from sglang.srt.layers.moe.utils import get_moe_a2a_backend

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


_VALID_COMPILERS = ("eager", "inductor")


class TCPiecewiseCudaGraphBackend:
    """torch.compile-driven piecewise capture; attention metadata
    recomputed at replay (outside the compiled callable's sub-graphs).
    """

    @staticmethod
    def build_compilation_config(server_args: "ServerArgs") -> CompilationConfig:
        """Construct the ``CompilationConfig`` from ``ServerArgs``.

        Validates the ``--piecewise-cuda-graph-compiler`` choice, builds
        the config, and registers the MoE A2A split-op when DeepEP /
        Mooncake is in use.
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
