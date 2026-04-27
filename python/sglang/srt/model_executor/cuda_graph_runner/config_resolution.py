"""Resolves the CUDA graph configuration from ``ServerArgs``.

Final design (per ``refactor/plan.md``) is a four-stage pipeline:
  1. **Parse** — convenience + deprecated flags → canonical ``cuda_graph_mode``.
  2. **Default** — GPU-memory-based sizes, bucket lists, per-backend defaults.
  3. **Compatibility** — per-(phase, backend) auto-disable conditions.
  4. **Validate** — reject impossible combinations.

Phase 1 lands stage 3 (compatibility, 16 conditions for piecewise/tcpcg).
Stages 1, 2, 4 land in Phase 4 alongside the new CLI surface; today's
GPU-memory defaulting in ``server_args._handle_gpu_memory_settings``
remains in place until then.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

from sglang.srt.utils import is_cpu, is_hip, is_mps, is_npu, is_xpu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 3 — compatibility checks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PiecewiseDisableRule:
    """One reason to auto-disable piecewise CUDA graph for the prefill phase.

    Replaces the long if/if cascade in the old
    ``ServerArgs._handle_piecewise_cuda_graph`` with a data-driven table
    so reviewers can see all conditions in one place and add new ones
    without thinking about ordering.
    """

    name: str
    predicate: Callable[["ServerArgs"], bool]


def _gguf_predicate(sa: "ServerArgs") -> bool:
    # Heavy import deferred to call site (matches what server_args.py does).
    from sglang.srt.utils.hf_transformers_utils import check_gguf_file

    return (
        sa.load_format == "gguf"
        or sa.quantization == "gguf"
        or check_gguf_file(sa.model_path)
    )


def _oot_platform_predicate(sa: "ServerArgs") -> bool:
    from sglang.srt.platforms import current_platform

    return (
        current_platform.is_out_of_tree()
        and not current_platform.support_piecewise_cuda_graph()
    )


def _multimodal_predicate(sa: "ServerArgs") -> bool:
    return sa.get_model_config().is_multimodal


def _model_blacklist_predicate(sa: "ServerArgs") -> bool:
    return sa.get_model_config().is_piecewise_cuda_graph_disabled_model


_PIECEWISE_DISABLE_RULES: List[_PiecewiseDisableRule] = [
    _PiecewiseDisableRule("model-arch blacklist", _model_blacklist_predicate),
    _PiecewiseDisableRule("DP attention", lambda sa: sa.enable_dp_attention),
    _PiecewiseDisableRule("full torch.compile mode", lambda sa: sa.enable_torch_compile),
    _PiecewiseDisableRule("pipeline parallelism (pp_size > 1)", lambda sa: sa.pp_size > 1),
    _PiecewiseDisableRule(
        "non-CUDA hardware (HIP/NPU/CPU/MPS/XPU)",
        lambda sa: is_hip() or is_npu() or is_cpu() or is_mps() or is_xpu(),
    ),
    _PiecewiseDisableRule("OOT platform without piecewise support", _oot_platform_predicate),
    _PiecewiseDisableRule("MoE A2A backend", lambda sa: sa.moe_a2a_backend != "none"),
    _PiecewiseDisableRule("LoRA", lambda sa: bool(sa.lora_paths) or sa.enable_lora),
    _PiecewiseDisableRule("multimodal model", _multimodal_predicate),
    _PiecewiseDisableRule("GGUF quantization", _gguf_predicate),
    _PiecewiseDisableRule("DLLM (diffusion LLM)", lambda sa: sa.dllm_algorithm is not None),
    _PiecewiseDisableRule(
        "CPU offload / hierarchical cache",
        lambda sa: sa.cpu_offload_gb > 0 or sa.enable_hierarchical_cache,
    ),
    _PiecewiseDisableRule("deterministic inference", lambda sa: sa.enable_deterministic_inference),
    _PiecewiseDisableRule("PD disaggregation", lambda sa: sa.disaggregation_mode != "null"),
    _PiecewiseDisableRule("symmetric memory", lambda sa: sa.enable_symm_mem),
    _PiecewiseDisableRule(
        "expert distribution recorder",
        lambda sa: sa.enable_eplb or sa.expert_distribution_recorder_mode is not None,
    ),
    _PiecewiseDisableRule("context parallel (attn_cp_size > 1)", lambda sa: sa.attn_cp_size > 1),
    _PiecewiseDisableRule("CUDA graph debug mode", lambda sa: sa.debug_cuda_graph),
]


def _apply_piecewise_compatibility(server_args: "ServerArgs") -> None:
    """Apply the prefill (piecewise) auto-disable rules.

    Mirrors the legacy ``ServerArgs._handle_piecewise_cuda_graph`` exactly,
    but as a data table for readability. ``--enforce-piecewise-cuda-graph``
    bypasses the entire table (testing override).
    """
    if server_args.enforce_piecewise_cuda_graph:
        server_args.disable_piecewise_cuda_graph = False
        return

    for rule in _PIECEWISE_DISABLE_RULES:
        if rule.predicate(server_args):
            server_args.disable_piecewise_cuda_graph = True
            # Legacy behavior did not short-circuit; preserve that in case
            # any predicate has incidental side effects.


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def resolve_cuda_graph_config(server_args: "ServerArgs") -> None:
    """Mutates ``server_args`` in place with resolved CUDA graph state.

    Called from ``ServerArgs.__post_init__``. Phase 1 implements stage 3
    (compatibility checks for piecewise/tcpcg). Other stages are stubs.
    """
    # _stage_parse(server_args)         # Phase 4
    # _stage_default(server_args)       # Phase 4 (today: handled by _handle_gpu_memory_settings)
    _apply_piecewise_compatibility(server_args)
    # _stage_validate(server_args)      # Phase 4
