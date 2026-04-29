"""Phase / backend identifiers, the canonical default for
``cuda_graph_mode``, and the ``--cuda-graph-mode`` JSON CLI parser.

Pure stdlib — no torch / sglang.srt deps — so ``ServerArgs`` can
import everything here at module level without pulling in backend
classes.
"""

import argparse
import json
from typing import Dict


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
    Phase.DECODE: (Backend.FULL, Backend.BREAKABLE, Backend.TCPIECEWISE, Backend.DISABLED),
    # ``full`` is rejected for prefill — full CUDA graph capture only
    # fits fixed-shape and prefill is variable-shape. Use ``breakable``
    # or ``tcpiecewise`` for prefill.
    Phase.PREFILL: (Backend.BREAKABLE, Backend.TCPIECEWISE, Backend.DISABLED),
}

DEFAULT_CUDA_GRAPH_MODE = {
    Phase.DECODE: Backend.FULL,
    Phase.PREFILL: Backend.TCPIECEWISE,
}


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
