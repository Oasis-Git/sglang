"""torch.compile-internal phase markers used by the tcpcg backend.

Two pieces of state, both private to the torch.compile path (the
``cuda_piecewise_backend`` FX backend and the ``PiecewiseCudaGraphRunner``
that drives it):

* ``_in_torch_compile_warmup`` — true during the warmup-compile loop
  where we run the compiled callable to trigger inductor compilation
  but explicitly do **not** capture into a CUDA graph yet.
  ``cuda_piecewise_backend`` reads this to short-circuit the capture
  branch. See ``refactor/plan.md`` §6.5.
* ``_pcg_capture_stream`` — the CUDA stream on which the tcpcg runner
  is performing capture, surfaced so the FX backend can use the same
  stream for its own ``torch.cuda.graph(...)`` calls.

Renamed from ``_in_pcg_torch_compile`` so the name describes the phase
("torch.compile is doing warmup compile") rather than the consumer
runner. ``compilation/piecewise_context_manager.py`` is a transition
shim re-exporting under the old names.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


_in_torch_compile_warmup = False
_pcg_capture_stream: "torch.cuda.Stream | None" = None


def is_in_torch_compile_warmup() -> bool:
    """True while inside the tcpcg warmup-compile pass. Strict subset of
    ``torch.compiler.is_compiling()``. See plan §6.5.
    """
    return _in_torch_compile_warmup


@contextmanager
def enable_torch_compile_warmup():
    """Mark the enclosed scope as the tcpcg warmup-compile pass. The FX
    piecewise backend uses this to skip CUDA graph capture during warmup.
    """
    global _in_torch_compile_warmup
    _in_torch_compile_warmup = True
    try:
        yield
    finally:
        _in_torch_compile_warmup = False


def get_pcg_capture_stream() -> "torch.cuda.Stream | None":
    return _pcg_capture_stream


@contextmanager
def set_pcg_capture_stream(stream: torch.cuda.Stream):
    global _pcg_capture_stream
    _pcg_capture_stream = stream
    try:
        yield
    finally:
        _pcg_capture_stream = None
