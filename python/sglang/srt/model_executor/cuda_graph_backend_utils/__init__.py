"""Low-level primitives used by the CUDA graph backends.

Subpackages:
  - ``breakable_cuda_graph``: BreakableCUDAGraph + capture context, eager break
    decorators.
  - ``piecewise_cuda_graph``: shared piecewise context manager
    (set_forward_context, is_in_cuda_graph_capture).

Backends ``cuda_graph_backend/*`` import from here. Runners do not.

Phase 0 scaffold. Contents land in Phase 1 (relocated from
``model_executor/breakable_cuda_graph/`` and
``compilation/piecewise_context_manager.py``).
"""
