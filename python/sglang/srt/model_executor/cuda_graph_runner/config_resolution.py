"""Resolves the CUDA graph configuration on ``ServerArgs``.

The pipeline runs in four stages:
  1. **Parse** — convenience + legacy flags merge into the canonical
     ``cuda_graph_mode`` dict.
  2. **Compatibility** — per-phase auto-disable rules (e.g. multimodal,
     LoRA, DP attention) flip the legacy ``disable_piecewise_cuda_graph``
     field.
  3. **Downgrade** — silently rewrite combinations that are accepted at
     parse time but not actually wired (today: ``(prefill, full)``).
  4. **Validate** — reject unknown phases / backends.

Stage 1 runs once before the compatibility table and once after, so any
auto-disable flips propagate into the resolved mode. GPU-memory-driven
size defaulting (``cuda_graph_max_bs``, ``piecewise_cuda_graph_tokens``)
still lives in ``server_args._handle_gpu_memory_settings``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List

from sglang.srt.utils import is_cpu, is_hip, is_mps, is_npu, is_xpu

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


PHASE_DECODE = "decode"
PHASE_PREFILL = "prefill"
ALL_PHASES = (PHASE_DECODE, PHASE_PREFILL)

BACKEND_FULL = "full"
BACKEND_BREAKABLE = "breakable"
BACKEND_TCPCG = "tcpcg"
BACKEND_DISABLED = "disabled"

ALLOWED_BACKENDS_PER_PHASE = {
    PHASE_DECODE: (BACKEND_FULL, BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
    # ``(prefill, full)`` is accepted at parse time but downgraded to
    # ``(prefill, disabled)`` with a warning — full CUDA graph capture
    # doesn't fit prefill's variable-shape model. Decode keeps full as
    # the default.
    PHASE_PREFILL: (BACKEND_FULL, BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
}

_DEFAULT_CUDA_GRAPH_MODE = {
    PHASE_DECODE: BACKEND_FULL,
    PHASE_PREFILL: BACKEND_TCPCG,
}


# ---------------------------------------------------------------------------
# Stage 2 — per-phase auto-disable rules for piecewise (prefill).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PiecewiseDisableRule:
    """One reason to auto-disable piecewise CUDA graph for the prefill
    phase. The data-table form replaces a long if/if cascade so the full
    set of conditions is visible in one place.
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

    ``--enforce-piecewise-cuda-graph`` bypasses the entire table for
    testing. Predicates do not short-circuit — every rule runs even
    after a match, in case any has incidental side effects.
    """
    if server_args.enforce_piecewise_cuda_graph:
        server_args.disable_piecewise_cuda_graph = False
        return

    for rule in _PIECEWISE_DISABLE_RULES:
        if rule.predicate(server_args):
            server_args.disable_piecewise_cuda_graph = True


# ---------------------------------------------------------------------------
# Stage 1 — parse: legacy + convenience + JSON → canonical mode.
# ---------------------------------------------------------------------------


def _parse_canonical(server_args: "ServerArgs") -> None:
    """Populate ``server_args.cuda_graph_mode`` from explicit JSON,
    convenience flags, and legacy flags.

    Precedence (highest first):
      1. ``--cuda-graph-mode`` JSON (per phase it specifies).
      2. Per-phase convenience flags:
         - ``--{prefill,decode}-disable-cuda-graph`` → ``disabled``.
         - ``--{prefill,decode}-cuda-graph-backend`` → that backend.
      3. Legacy flags:
         - ``--disable-cuda-graph`` → both phases ``disabled``.
         - ``--disable-piecewise-cuda-graph`` → prefill ``disabled``.
         - ``--enable-breakable-cuda-graph`` → prefill ``breakable``.
      4. Defaults: ``{decode: full, prefill: tcpcg}``.

    A warning is emitted whenever a higher-precedence source overrides
    a lower one. Re-runs are idempotent — the resolved
    ``cuda_graph_mode`` becomes the explicit input for any subsequent
    re-parse.
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

    # Build per-source views, lowest precedence first.
    legacy_view: Dict[str, str] = dict(_DEFAULT_CUDA_GRAPH_MODE)
    if server_args.disable_cuda_graph:
        legacy_view[PHASE_DECODE] = BACKEND_DISABLED
        legacy_view[PHASE_PREFILL] = BACKEND_DISABLED
    if server_args.disable_piecewise_cuda_graph:
        legacy_view[PHASE_PREFILL] = BACKEND_DISABLED
    elif server_args.enable_breakable_cuda_graph:
        legacy_view[PHASE_PREFILL] = BACKEND_BREAKABLE

    convenience_view: Dict[str, str] = {}
    if server_args.prefill_disable_cuda_graph:
        convenience_view[PHASE_PREFILL] = BACKEND_DISABLED
    if server_args.decode_disable_cuda_graph:
        convenience_view[PHASE_DECODE] = BACKEND_DISABLED
    if server_args.prefill_cuda_graph_backend is not None:
        convenience_view[PHASE_PREFILL] = server_args.prefill_cuda_graph_backend
    if server_args.decode_cuda_graph_backend is not None:
        convenience_view[PHASE_DECODE] = server_args.decode_cuda_graph_backend

    # Merge highest-precedence first. Explicit JSON wins; convenience
    # flags fill remaining phases; legacy flags fill the rest; defaults
    # cover anything still missing.
    resolved: Dict[str, str] = dict(_DEFAULT_CUDA_GRAPH_MODE)
    for phase in ALL_PHASES:
        if phase in explicit:
            resolved[phase] = explicit[phase]
            # Warn on lower-precedence conflicts.
            for src_name, src in (
                ("--{}-cuda-graph-backend / --{}-disable-cuda-graph".format(phase, phase),
                 convenience_view),
                ("legacy convenience flags", legacy_view),
            ):
                if (
                    phase in src
                    and src[phase] != _DEFAULT_CUDA_GRAPH_MODE[phase]
                    and src[phase] != explicit[phase]
                ):
                    logger.warning(
                        "--cuda-graph-mode %s=%r overrides %s "
                        "(which would have set %s=%r). Using JSON.",
                        phase,
                        explicit[phase],
                        src_name,
                        phase,
                        src[phase],
                    )
        elif phase in convenience_view:
            resolved[phase] = convenience_view[phase]
            if (
                phase in legacy_view
                and legacy_view[phase] != _DEFAULT_CUDA_GRAPH_MODE[phase]
                and legacy_view[phase] != convenience_view[phase]
            ):
                logger.warning(
                    "Per-phase convenience flag for %s=%r overrides legacy "
                    "flag (which would have set %s=%r).",
                    phase,
                    convenience_view[phase],
                    phase,
                    legacy_view[phase],
                )
        else:
            resolved[phase] = legacy_view[phase]

    server_args.cuda_graph_mode = resolved


# ---------------------------------------------------------------------------
# Stage 3 — downgrade unsupported combinations.
# ---------------------------------------------------------------------------


def _reject_unsupported_combinations(server_args: "ServerArgs") -> None:
    """Reject (phase, backend) combinations that are accepted at parse
    time but not implemented. Today this is only ``(prefill, full)``.
    """
    mode = server_args.cuda_graph_mode or {}
    if mode.get(PHASE_PREFILL) == BACKEND_FULL:
        raise NotImplementedError(
            "--cuda-graph-mode prefill='full' is not supported. Full CUDA "
            "graph capture only fits fixed-shape deployments and prefill "
            "is variable-shape. Use 'breakable' or 'tcpcg' for prefill, "
            "or 'disabled' to skip cuda graphs there."
        )


# ---------------------------------------------------------------------------
# Stage 4 — validate.
# ---------------------------------------------------------------------------


def _validate_canonical(server_args: "ServerArgs") -> None:
    """Reject invalid backend choices per phase."""
    mode = server_args.cuda_graph_mode or {}
    for phase, backend in mode.items():
        if phase not in ALLOWED_BACKENDS_PER_PHASE:
            raise ValueError(
                f"--cuda-graph-mode has unknown phase {phase!r}; "
                f"allowed: {ALL_PHASES}"
            )
        allowed = ALLOWED_BACKENDS_PER_PHASE[phase]
        if backend not in allowed:
            raise ValueError(
                f"--cuda-graph-mode {phase}={backend!r} is not allowed; "
                f"allowed values for {phase}: {allowed}"
            )


# ---------------------------------------------------------------------------
# Top-level entry point — invoked from ``ServerArgs.__post_init__``.
# ---------------------------------------------------------------------------


def resolve_cuda_graph_config(server_args: "ServerArgs") -> None:
    """Mutates ``server_args`` in place with the resolved CUDA graph state.

    Stage order matters: parse first so legacy fields stay authoritative
    inputs; then run the legacy-flag-driven compatibility table (which may
    mutate ``disable_piecewise_cuda_graph``); then re-parse so any
    compatibility-driven flag changes propagate into ``cuda_graph_mode``;
    then downgrade unsupported combinations; finally validate.
    """
    _parse_canonical(server_args)
    _apply_piecewise_compatibility(server_args)
    _parse_canonical(server_args)
    _reject_unsupported_combinations(server_args)
    _validate_canonical(server_args)
