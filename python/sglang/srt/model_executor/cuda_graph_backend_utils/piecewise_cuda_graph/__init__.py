"""Piecewise CUDA graph context utilities — relocated in Phase 1 from
``compilation/piecewise_context_manager.py``.

After Phase 1 the renamed primitives live here:

  - ``is_in_cuda_graph_capture()`` / ``enable_cuda_graph_capture()``
    (was ``is_in_piecewise_cuda_graph`` / ``enable_piecewise_cuda_graph``)
  - ``set_forward_context``, ``get_forward_context``, ``ForwardContext``

The torch.compile-warmup flag (``is_in_torch_compile_warmup`` /
``enable_torch_compile_warmup``, was ``is_in_pcg_torch_compile``) does
**not** move here — it stays under ``compilation/`` since it is
torch.compile-internal and only consumed by ``cuda_piecewise_backend``.

Phase 0 scaffold; empty until Phase 1.
"""
