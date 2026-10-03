"""Compare fused metadata with the original tensor expressions, including BS64."""

import itertools
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.indexer_prefill_metadata import (
    build_indexer_prefill_metadata,
)


def reference(table, req, pos, lens, rows, ratio):
    device = pos.device
    starts = list(itertools.accumulate(lens, initial=0))[:-1]
    lengths = torch.tensor(lens, device=device)
    starts_dev = torch.tensor(starts, device=device)
    rows_dev = torch.tensor(rows, device=device)
    request = torch.repeat_interleave(
        torch.arange(len(lens), device=device), lengths, output_size=sum(lens)
    )
    position = torch.arange(sum(lens), device=device) - starts_dev[request]
    slots = table[req.long()[request], position * ratio].long() // ratio
    request_starts = torch.repeat_interleave(
        starts_dev.int(), rows_dev, output_size=sum(rows)
    )
    compress_lens = ((pos + 1) // ratio).int()
    row = torch.arange(sum(rows), device=device)
    request = torch.repeat_interleave(
        torch.arange(len(rows), device=device), rows_dev, output_size=sum(rows)
    )
    first = rows_dev.cumsum(0) - rows_dev
    pair = (row - ((row - first[request]) & 1)).int()
    return slots, request_starts, compress_lens, pair


class TestIndexerPrefillMetadata(unittest.TestCase):
    @torch.inference_mode()
    def test_reference(self):
        for bs in [1, 2, 8, 64]:
            for ratio in [1, 2, 4, 128]:
                for strided in [False, True]:
                    with self.subTest(bs=bs, ratio=ratio, strided=strided):
                        torch.manual_seed(bs + ratio)
                        seqs = [
                            95500 if bs == 1 else 128 * (1 + (i * 71) % 512) + i % 128
                            for i in range(bs)
                        ]
                        rows = [
                            min(s, 2060 if bs == 1 else (i * 337) % 4096)
                            for i, s in enumerate(seqs)
                        ]
                        lens = [s // ratio for s in seqs]
                        step = 2 if strided else 1
                        table = torch.randint(
                            -128,
                            1 << 20,
                            (bs, max(seqs) * step),
                            device="cuda",
                            dtype=torch.int32,
                        )[:, ::step]
                        table[:, ::13] = -1
                        req = torch.empty(bs * step, device="cuda", dtype=torch.int32)[
                            ::step
                        ]
                        req.copy_(torch.randperm(bs, device="cuda"))
                        pos = torch.empty(
                            sum(rows) * step, device="cuda", dtype=torch.int64
                        )[::step]
                        pos.copy_(
                            torch.cat(
                                [
                                    torch.arange(s - n, s, device="cuda")
                                    for s, n in zip(seqs, rows)
                                ]
                            )
                        )
                        actual = build_indexer_prefill_metadata(
                            table, req, pos, lens, rows, ratio
                        )
                        expected = reference(table, req, pos, lens, rows, ratio)
                        for a, b in zip(actual, expected):
                            self.assertTrue(torch.equal(a, b))

    @torch.inference_mode()
    def test_empty_and_negative_sentinels(self):
        for rows, lens in [([0, 0], [0, 0]), ([0, 3], [0, 0]), ([0, 0], [3, 0])]:
            table = torch.arange(-512, 0, device="cuda", dtype=torch.int64).reshape(
                2, 256
            )
            req = torch.tensor([1, 0], device="cuda", dtype=torch.int64)
            pos = torch.arange(-2, -2 + sum(rows), device="cuda", dtype=torch.int64)
            for a, b in zip(
                build_indexer_prefill_metadata(table, req, pos, lens, rows, 4),
                reference(table, req, pos, lens, rows, 4),
            ):
                self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
