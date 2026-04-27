"""Phase-aware CUDA graph runners.

During the cg-refactor migration this package re-exports the legacy
public API (now in ``.legacy``) so the 31 existing import sites
continue to work transparently. The new abstractions
(``BaseCudaGraphRunner``, ``PrefillCudaGraphRunner``,
``DecodeCudaGraphRunner``) are exposed alongside.

The ``legacy`` module is removed in Phase 6 of the refactor.
"""

# Re-export legacy public API. Public names propagate via a star import;
# private helpers actually referenced externally (none today, but keep
# the shim transparent) come along too.
from sglang.srt.model_executor.cuda_graph_runner.legacy import *  # noqa: F401,F403

# Explicit re-exports of underscore-prefixed names, in case any caller
# reaches in. Keeps the shim 1:1 with the old module surface.
from sglang.srt.model_executor.cuda_graph_runner.legacy import (  # noqa: F401
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    _default_make_graph_key,
    _grouped_foreach_copy_,
    _set_capture_lora_variant,
    _to_torch,
)

# New (Phase 0 scaffold) abstractions. Not yet wired anywhere.
from sglang.srt.model_executor.cuda_graph_runner.base_runner import (  # noqa: F401
    BaseCudaGraphRunner,
)
from sglang.srt.model_executor.cuda_graph_runner.decode_runner import (  # noqa: F401
    DecodeCudaGraphRunner,
)
from sglang.srt.model_executor.cuda_graph_runner.prefill_runner import (  # noqa: F401
    PrefillCudaGraphRunner,
)
