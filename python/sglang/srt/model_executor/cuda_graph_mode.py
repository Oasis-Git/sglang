"""Phase / backend identifiers, the canonical default for
``cuda_graph_mode``, and the ``--cuda-graph-mode`` JSON CLI parser.

Module-level imports are pure stdlib — no torch / sglang.srt deps — so
``ServerArgs`` can import everything here without pulling in backend
classes. ``check_cuda_graph_enable`` lazy-imports ``get_global_server_args``
inside the function body to preserve that invariant.
"""

import argparse
import json
from typing import Dict, Optional


class Phase:
    """The two phases of model forward."""

    DECODE = "decode"
    PREFILL = "prefill"
    ALL = (DECODE, PREFILL)


class Backend:
    """CUDA graph capture backends a phase can use."""

    FULL = "full"
    BREAKABLE = "breakable"
    TCPIECEWISE = "tcpiecewise"
    DISABLED = "disabled"
    ALL = (FULL, BREAKABLE, TCPIECEWISE, DISABLED)


ALLOWED_BACKENDS_PER_PHASE = {
    Phase.DECODE: (
        Backend.FULL,
        Backend.BREAKABLE,
        Backend.TCPIECEWISE,
        Backend.DISABLED,
    ),
    # ``full`` is rejected for prefill — full CUDA graph capture only
    # fits fixed-shape and prefill is variable-shape. Use ``breakable``
    # or ``tcpiecewise`` for prefill.
    Phase.PREFILL: (Backend.BREAKABLE, Backend.TCPIECEWISE, Backend.DISABLED),
}

DEFAULT_CUDA_GRAPH_MODE = {
    Phase.DECODE: Backend.FULL,
    Phase.PREFILL: Backend.TCPIECEWISE,
}


def check_cuda_graph_enable(phase: str, backend: Optional[str] = None) -> bool:
    """True if cuda graph is enabled for ``phase`` on the global server args.

    If ``backend`` is given, return True only when that specific backend is
    selected for the phase. Returns False if the global server args have not
    been initialized yet (e.g. unit tests, early startup).
    """
    from sglang.srt.server_args import get_global_server_args

    try:
        server_args = get_global_server_args()
    except ValueError:
        return False
    if server_args.cuda_graph_mode is None:
        return False
    current = server_args.cuda_graph_mode[phase]
    if backend is None:
        return current != Backend.DISABLED
    return current == backend


def parse_cuda_graph_mode_arg(raw: str) -> Dict[str, str]:
    """argparse type for ``--cuda-graph-mode``: parse JSON dict of phase → backend."""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"--cuda-graph-mode must be JSON: {e}")
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"--cuda-graph-mode must be a JSON object, got {type(parsed).__name__}"
        )
    return {str(k): str(v) for k, v in parsed.items()}
