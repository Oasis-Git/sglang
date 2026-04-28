"""Resolves the CUDA graph configuration from ``ServerArgs``.

Final design (per ``refactor/plan.md``) is a four-stage pipeline:
  1. **Parse** — convenience + deprecated flags → canonical ``cuda_graph_mode``.
  2. **Default** — GPU-memory-based sizes, bucket lists, per-backend defaults.
  3. **Compatibility** — per-(phase, backend) auto-disable conditions.
  4. **Validate** — reject impossible combinations.

Phase 1 landed stage 3 (compatibility, 18 conditions for piecewise/tcpcg).
Phase 4a lands stage 1 (parse) + stage 4 (validate). GPU-memory-based
defaulting (stage 2) still lives in
``server_args._handle_gpu_memory_settings`` and migrates here later.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

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


PHASE_DECODE = "decode"
PHASE_PREFILL = "prefill"
ALL_PHASES = (PHASE_DECODE, PHASE_PREFILL)

BACKEND_FULL = "full"
BACKEND_BREAKABLE = "breakable"
BACKEND_TCPCG = "tcpcg"
BACKEND_DISABLED = "disabled"

ALLOWED_BACKENDS_PER_PHASE = {
    PHASE_DECODE: (BACKEND_FULL, BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
    PHASE_PREFILL: (BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
    # (prefill, full) is intentionally a NotImplementedError stub — see
    # plan §6 Q1. The validator rejects it explicitly.
}

_DEFAULT_CUDA_GRAPH_MODE = {
    # Today's defaults: full decode + tcpcg prefill. Phase 5 flips the
    # prefill default to breakable per the plan goal once the breakable
    # path is exercised end-to-end across the model coverage matrix.
    PHASE_DECODE: BACKEND_FULL,
    PHASE_PREFILL: BACKEND_TCPCG,
}


def _parse_canonical(server_args: "ServerArgs") -> None:
    """Stage 1 — populate ``server_args.cuda_graph_mode`` if not yet set
    by translating today's legacy flags.

    Precedence (matches plan §3.2 + §6 Q8):
      1. Explicit ``--cuda-graph-mode`` JSON wins (per phase that it
         specifies).
      2. Legacy ``--disable-cuda-graph`` overrides decode to ``disabled``.
      3. Legacy ``--disable-piecewise-cuda-graph`` overrides prefill to
         ``disabled``.
      4. Legacy ``--enable-breakable-cuda-graph`` chooses prefill backend
         (``breakable``).
      5. Otherwise, defaults from ``_DEFAULT_CUDA_GRAPH_MODE``.

    When the explicit JSON conflicts with a legacy convenience flag, the
    JSON wins and we emit a warning naming both — Q8 from the plan.
    """
    explicit: Dict[str, str] = dict(server_args.cuda_graph_mode or {})

    # Reject unknown phases up front so validate() catches them even
    # though parse() drops them on the floor when merging.
    for phase in explicit:
        if phase not in ALL_PHASES:
            raise ValueError(
                f"--cuda-graph-mode has unknown phase {phase!r}; "
                f"allowed: {ALL_PHASES}"
            )

    resolved: Dict[str, str] = dict(_DEFAULT_CUDA_GRAPH_MODE)

    # Build the legacy-flag-derived view of what the canonical mode
    # would be without --cuda-graph-mode. Used both as a fallback when
    # JSON is unset, and to detect conflicts when JSON is set.
    legacy_view: Dict[str, str] = dict(_DEFAULT_CUDA_GRAPH_MODE)
    if server_args.disable_cuda_graph:
        legacy_view[PHASE_DECODE] = BACKEND_DISABLED
    if server_args.disable_piecewise_cuda_graph:
        legacy_view[PHASE_PREFILL] = BACKEND_DISABLED
    elif server_args.enable_breakable_cuda_graph:
        legacy_view[PHASE_PREFILL] = BACKEND_BREAKABLE

    # Merge: explicit JSON wins per-phase; legacy view fills gaps.
    for phase in ALL_PHASES:
        if phase in explicit:
            resolved[phase] = explicit[phase]
            if (
                phase in legacy_view
                and legacy_view[phase] != _DEFAULT_CUDA_GRAPH_MODE[phase]
                and legacy_view[phase] != explicit[phase]
            ):
                logger.warning(
                    "--cuda-graph-mode %s=%r overrides legacy convenience "
                    "flag (which would have set %s=%r). Using JSON.",
                    phase,
                    explicit[phase],
                    phase,
                    legacy_view[phase],
                )
        else:
            resolved[phase] = legacy_view[phase]

    server_args.cuda_graph_mode = resolved


def _validate_canonical(server_args: "ServerArgs") -> None:
    """Stage 4 — reject invalid backend choices per phase (plan §6 Q1)."""
    mode = server_args.cuda_graph_mode or {}
    for phase, backend in mode.items():
        if phase not in ALLOWED_BACKENDS_PER_PHASE:
            raise ValueError(
                f"--cuda-graph-mode has unknown phase {phase!r}; "
                f"allowed: {ALL_PHASES}"
            )
        allowed = ALLOWED_BACKENDS_PER_PHASE[phase]
        if backend not in allowed:
            if phase == PHASE_PREFILL and backend == BACKEND_FULL:
                raise NotImplementedError(
                    "(prefill, full) is not implemented — full CUDA graph "
                    "capture for prefill only fits fixed-shape deployments. "
                    "Use 'breakable' or 'tcpcg' for prefill."
                )
            raise ValueError(
                f"--cuda-graph-mode {phase}={backend!r} is not allowed; "
                f"allowed values for {phase}: {allowed}"
            )


def resolve_cuda_graph_config(server_args: "ServerArgs") -> None:
    """Mutates ``server_args`` in place with resolved CUDA graph state.

    Called from ``ServerArgs.__post_init__``. Phase 1 implemented stage 3
    (compatibility checks for piecewise/tcpcg). Phase 4a adds stage 1
    (parse legacy + JSON into canonical ``cuda_graph_mode``) and stage 4
    (validate). Stage 2 (default sizes) still lives in ``server_args``.

    Stage order matters: parse first so legacy fields stay authoritative
    inputs; then run the existing legacy-flag-driven compatibility table
    (which may mutate ``disable_piecewise_cuda_graph``); then re-parse
    so any compatibility-driven flag changes propagate into
    ``cuda_graph_mode``; finally validate.
    """
    _parse_canonical(server_args)
    _apply_piecewise_compatibility(server_args)
    _parse_canonical(server_args)
    _validate_canonical(server_args)
