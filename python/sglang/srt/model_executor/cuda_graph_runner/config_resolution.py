"""Resolves the CUDA graph configuration from ``ServerArgs``.

Owns the four-stage pipeline:
  1. Parse: convenience + deprecated flags → canonical ``cuda_graph_mode``.
  2. Default: GPU-memory-based sizes, bucket lists, per-backend defaults.
  3. Compatibility: per-(phase, backend) auto-disable conditions.
  4. Validate: reject impossible combinations.

Phase 0 placeholder. The pipeline lands in Phase 1 (relocated from
``server_args.py``'s ``_handle_piecewise_cuda_graph`` and friends).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


def resolve_cuda_graph_config(server_args: "ServerArgs") -> None:
    """Mutates ``server_args`` in place with resolved CUDA graph state.

    Phase 0: no-op stub. Replaced in Phase 1 with the full pipeline.
    """
    return None
