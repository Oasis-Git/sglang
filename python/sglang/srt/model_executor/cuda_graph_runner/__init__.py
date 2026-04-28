"""Phase-aware CUDA graph runners.

Public API:
  - ``BaseCudaGraphRunner`` — abstract base.
  - ``DecodeCudaGraphRunner`` — concrete decode-phase runner.
  - ``PrefillCudaGraphRunner`` — prefill-phase factory (for now still
    selecting between the legacy BCG / PCG runners; will be lifted into
    a real concrete runner in Phase F).
  - Helpers re-exported for the EAGLE / multi-step draft cuda graph
    runners that were authored against the legacy public surface.
"""

from sglang.srt.model_executor.cuda_graph_backend_utils.piecewise_cuda_graph import (  # noqa: F401
    PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG,
)
from sglang.srt.model_executor.cuda_graph_runner.base_runner import (  # noqa: F401
    BaseCudaGraphRunner,
    freeze_gc,
    get_batch_sizes_to_capture,
)
from sglang.srt.model_executor.cuda_graph_runner.buffers import (  # noqa: F401
    DecodeInputBuffers,
    PrefillInputBuffers,
    _grouped_foreach_copy_,
)
from sglang.srt.model_executor.cuda_graph_runner.capture_mode import (  # noqa: F401
    _set_capture_lora_variant,
    get_capture_lora_variant,
    get_is_capture_mode,
    model_capture_mode,
)
from sglang.srt.model_executor.cuda_graph_runner.decode_runner import (  # noqa: F401
    DecodeCudaGraphRunner,
    _make_graph_key as _default_make_graph_key,
)
from sglang.srt.model_executor.cuda_graph_runner.deepep_adapter import (  # noqa: F401
    DeepEPCudaGraphRunnerAdapter,
)
from sglang.srt.model_executor.cuda_graph_runner.pool import (  # noqa: F401
    get_global_graph_memory_pool,
    set_global_graph_memory_pool,
)
from sglang.srt.model_executor.cuda_graph_runner.prefill_runner import (  # noqa: F401
    PrefillCudaGraphRunner,
)
