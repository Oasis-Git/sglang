"""Build DSv4.1 flattened-K and query metadata in one eager prefill launch."""

import itertools

import torch
import triton
import triton.language as tl

from sglang.srt.utils.common import async_h2d


@triton.jit
def _floor_div(value, divisor: tl.constexpr):
    # Preserve PyTorch floor division even for a negative sentinel pool slot.
    return tl.where(value < 0, -((-value + divisor - 1) // divisor), value // divisor)


@triton.jit
def _build_metadata(
    R2T,
    REQ,
    POS,
    META,
    SLOTS,
    STARTS,
    LENS,
    PAIRS,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    RATIO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    columns = tl.load(META + request * 4)
    column_start = tl.load(META + request * 4 + 1)
    rows = tl.load(META + request * 4 + 2)
    row_start = tl.load(META + request * 4 + 3)
    req = tl.load(REQ + request * REQ_STRIDE).to(tl.int64)
    physical = tl.load(
        R2T + req * ROW_STRIDE + offset * RATIO * COL_STRIDE,
        offset < columns,
        0,
    ).to(tl.int64)
    tl.store(
        SLOTS + column_start + offset, _floor_div(physical, RATIO), offset < columns
    )
    pos = tl.load(POS + (row_start + offset) * POS_STRIDE, offset < rows, 0).to(
        tl.int64
    )
    tl.store(STARTS + row_start + offset, column_start.to(tl.int32), offset < rows)
    tl.store(
        LENS + row_start + offset,
        _floor_div(pos + 1, RATIO).to(tl.int32),
        offset < rows,
    )
    tl.store(
        PAIRS + row_start + offset,
        (row_start + offset - (offset & 1)).to(tl.int32),
        offset < rows,
    )


def build_indexer_prefill_metadata(
    req_to_token, req_pool_indices, positions, lens_per_request, rows_per_request, ratio
):
    """Return flattened K slots, per-query column starts/lengths and row-pair IDs.

    All input lengths are already available on the CPU. No device-to-host read,
    cross-forward cache, or assumptions about BS1 are needed.
    """
    assert ratio > 0
    assert len(lens_per_request) == len(rows_per_request) == req_pool_indices.numel()
    assert sum(rows_per_request) == positions.numel()
    starts = list(itertools.accumulate(lens_per_request, initial=0))[:-1]
    row_starts = list(itertools.accumulate(rows_per_request, initial=0))[:-1]
    device = positions.device
    metadata = async_h2d(
        list(zip(lens_per_request, starts, rows_per_request, row_starts)),
        dtype=torch.int64,
        device=device,
    )
    outputs = (
        torch.empty(sum(lens_per_request), device=device, dtype=torch.int64),
        *(
            torch.empty(positions.numel(), device=device, dtype=torch.int32)
            for _ in range(3)
        ),
    )
    max_items = max(lens_per_request + rows_per_request, default=0)
    if max_items:
        _build_metadata[(len(rows_per_request), triton.cdiv(max_items, 256))](
            req_to_token,
            req_pool_indices,
            positions,
            metadata,
            *outputs,
            *req_to_token.stride(),
            req_pool_indices.stride(0),
            positions.stride(0),
            ratio,
            256,
        )
    return outputs
