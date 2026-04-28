"""Low-level primitives used by the CUDA graph backends.

Subpackages:
  - ``breakable_cuda_graph``: ``BreakableCUDAGraph`` + capture context,
    ``eager_on_graph`` decorator, ``is_in_breakable_cuda_graph`` flag.
  - ``piecewise_cuda_graph``: shared piecewise context manager
    (``set_forward_context``, ``is_in_cuda_graph_capture``).

Backends in ``cuda_graph_backend/`` import from here. Runners do not.
"""
