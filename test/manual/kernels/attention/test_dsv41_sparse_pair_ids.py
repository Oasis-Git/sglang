"""Regression for preserving already-bounded sparse prefill row-pair IDs."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsv4.v41_indexer import sparse_table


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA tensors and events")
class TestSparsePairIds(unittest.TestCase):
    def build_table(self, counts):
        ids = sparse_table._row_pair_ids(counts, device=torch.device("cuda"))
        expected = []
        offset = 0
        for count in counts:
            expected.extend(offset + (row // 2) * 2 for row in range(count))
            offset += count
        self.assertEqual(ids.tolist(), expected)
        rows = sum(counts)
        blocks = torch.zeros((rows, 1), dtype=torch.int32, device="cuda")
        lens = torch.ones(rows, dtype=torch.int32, device="cuda")
        # The DeepGEMM kernel is validated separately. Here, intercept its input
        # to catch confusing precomputed pair IDs with dense request indices.
        with (
            patch.object(sparse_table, "sort_candidate_blocks", return_value=blocks),
            patch.object(
                sparse_table,
                "build_sparse_indexer_schedule",
                return_value=torch.empty(1, dtype=torch.uint8, device="cuda"),
            ) as schedule,
        ):
            table = sparse_table._build_prefill_table(
                blocks=blocks,
                compress_lens=lens,
                page_table=blocks,
                page_size=256,
                request_ids=ids,
                rows_per_request=counts,
                q_dtype=torch.uint8,
                valid_lens=lens,
            )
            torch.cuda.synchronize()
            self.assertIs(schedule.call_args.args[-1], ids)
            self.assertIs(table.request_ids, ids)
        return table

    def test_long_single_request(self):
        self.build_table([8192])

    def test_odd_and_empty_requests(self):
        self.build_table([0, 2049, 0, 3, 1025, 0])

    def test_tail_preserves_pairing(self):
        table = self.build_table([2049, 2050])
        counts = [1025, 1025]
        expected = torch.cat((table.request_ids[1024:2049], table.request_ids[-1025:]))
        with (
            patch.object(
                sparse_table, "sort_candidate_blocks", return_value=table.blocks
            ),
            patch.object(
                sparse_table,
                "build_sparse_indexer_schedule",
                return_value=table.schedule,
            ) as schedule,
        ):
            tail = table.tail(counts)
            torch.cuda.synchronize()
            torch.testing.assert_close(tail.request_ids, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                schedule.call_args.args[-1], expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
