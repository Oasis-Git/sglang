"""Phase / backend identifiers, the canonical default for
``cuda_graph_mode``, and the ``--cuda-graph-mode`` JSON CLI parser.
Pure stdlib — no torch / sglang.srt deps — so ``ServerArgs`` can
import everything here at module level without pulling in backend
classes.
"""

import argparse
import json
from typing import Dict

PHASE_DECODE = "decode"
PHASE_PREFILL = "prefill"
ALL_PHASES = (PHASE_DECODE, PHASE_PREFILL)

BACKEND_FULL = "full"
BACKEND_BREAKABLE = "breakable"
BACKEND_TCPIECEWISE = "tcpiecewise"
BACKEND_DISABLED = "disabled"
ALL_BACKENDS = (BACKEND_FULL, BACKEND_BREAKABLE, BACKEND_TCPIECEWISE, BACKEND_DISABLED)

ALLOWED_BACKENDS_PER_PHASE = {
    PHASE_DECODE: (BACKEND_FULL, BACKEND_BREAKABLE, BACKEND_TCPIECEWISE, BACKEND_DISABLED),
    # ``full`` is rejected for prefill — full CUDA graph capture only
    # fits fixed-shape and prefill is variable-shape. Use ``breakable``
    # or ``tcpiecewise`` for prefill.
    PHASE_PREFILL: (BACKEND_BREAKABLE, BACKEND_TCPIECEWISE, BACKEND_DISABLED),
}

DEFAULT_CUDA_GRAPH_MODE = {
    PHASE_DECODE: BACKEND_FULL,
    PHASE_PREFILL: BACKEND_TCPIECEWISE,
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
