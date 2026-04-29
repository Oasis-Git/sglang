"""Phase / backend identifiers and the canonical default for
``cuda_graph_mode``. Pure constants — no torch / sglang.srt deps —
so ``ServerArgs`` and other lightweight modules can import these at
module level without pulling in backend classes.
"""

PHASE_DECODE = "decode"
PHASE_PREFILL = "prefill"
ALL_PHASES = (PHASE_DECODE, PHASE_PREFILL)

BACKEND_FULL = "full"
BACKEND_BREAKABLE = "breakable"
BACKEND_TCPCG = "tcpcg"
BACKEND_DISABLED = "disabled"

ALLOWED_BACKENDS_PER_PHASE = {
    PHASE_DECODE: (BACKEND_FULL, BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
    # ``full`` is rejected for prefill — full CUDA graph capture only
    # fits fixed-shape and prefill is variable-shape. Use ``breakable``
    # or ``tcpcg`` for prefill.
    PHASE_PREFILL: (BACKEND_BREAKABLE, BACKEND_TCPCG, BACKEND_DISABLED),
}

DEFAULT_CUDA_GRAPH_MODE = {
    PHASE_DECODE: BACKEND_FULL,
    PHASE_PREFILL: BACKEND_TCPCG,
}
