"""Phase-aware CUDA graph runners.

Public API:
  - ``CudaGraphRunner`` — the decode-phase runner (lives in ``.legacy``;
    re-exported here for backwards-compatible import paths).
  - ``DecodeCudaGraphRunner`` — factory selecting the decode backend
    (full / breakable; tcpcg falls back to full with a warning) from
    ``cuda_graph_mode["decode"]``.
  - ``PrefillCudaGraphRunner`` — factory selecting the prefill backend
    (breakable / tcpcg; full is silently downgraded to disabled) from
    ``cuda_graph_mode["prefill"]``.

The legacy module hosts the implementation today and will be folded
into the new runner classes in a follow-up.
"""

# Re-export legacy public API. Public names propagate via a star import;
# a few underscore-prefixed names are explicitly listed so the surface
# remains 1:1 with the original module.
from sglang.srt.model_executor.cuda_graph_runner.legacy import *  # noqa: F401,F403
from sglang.srt.model_executor.cuda_graph_runner.legacy import (  # noqa: F401
    PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG,
    _default_make_graph_key,
    _grouped_foreach_copy_,
    _set_capture_lora_variant,
    _to_torch,
)
from sglang.srt.model_executor.cuda_graph_runner.decode_runner import (  # noqa: F401
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.cuda_graph_runner.prefill_runner import (  # noqa: F401
    PrefillCudaGraphRunner,
)
