"""BaseCudaGraphBackend — abstract interface for one capture mechanism.

The backend sees an opaque shape key (token count for prefill, batch size
for decode) and a forward closure. It does not know whether it serves
prefill or decode.

Phase 0 placeholder. Concrete protocol lands in Phase 2.
"""

from __future__ import annotations


class BaseCudaGraphBackend:
    """Abstract capture-mechanism backend skeleton."""

    pass
