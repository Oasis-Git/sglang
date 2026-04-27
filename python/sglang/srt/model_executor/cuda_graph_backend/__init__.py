"""Capture-mechanism backends for CUDA graphs.

A backend owns *how* a captured artifact is produced and replayed for one
shape; it is phase-agnostic. Runners (``cuda_graph_runner/``) own *what*
data flows in and out.

Phase 0 scaffold. Concrete backends land in Phase 2.
"""
