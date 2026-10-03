"""Validate production wo_a dispatch against einsum on packed and padded rows.

The production function is extracted to avoid initializing the model stack.
Run on Blackwell with --source pointing at the candidate deepseek_v4.py.
"""

from __future__ import annotations

import argparse
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

TEST_PATH = Path(__file__).resolve()
SOURCE = (
    TEST_PATH.parents[4] / "python/sglang/srt/models/deepseek_v4.py"
    if len(TEST_PATH.parents) > 4
    else TEST_PATH.with_name("deepseek_v4.py")
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestSmallPrefillWoA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability()[0] != 10:
            raise unittest.SkipTest("Blackwell required")
        tree = ast.parse(SOURCE.read_text())
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_apply_wo_a_bf16_matmul"
        )
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        module = ast.fix_missing_locations(
            ast.Module(body=[future, function], type_ignores=[])
        )
        namespace = dict(
            torch=torch,
            _is_cuda=True,
            _is_hip=False,
            _is_gfx95_supported=False,
            _wo_a_aiter_batched_gemm_enabled=False,
            _wo_a_aiter_batched_gemm_disabled=False,
            get_platform=lambda: SimpleNamespace(is_blackwell=True, is_sm90=False),
        )
        exec(compile(module, str(SOURCE), "exec"), namespace)
        cls.apply = staticmethod(namespace["_apply_wo_a_bf16_matmul"])

    def check(self, rows, padded=False, groups=2, dtype=torch.bfloat16):
        torch.manual_seed(42)
        storage = torch.randn(
            rows, groups + int(padded), 4096, device="cuda", dtype=dtype
        )
        x = storage[:, :groups]
        weight = torch.randn(groups, 1024, 4096, device="cuda", dtype=dtype) * 0.02
        expected = torch.einsum("tgd,grd->tgr", x, weight)
        with mock.patch.object(torch, "bmm", wraps=torch.bmm) as bmm:
            actual = self.apply(
                x, weight, is_decode=False, is_prefill=True, fast_path=True
            )
        supported = rows >= 9 and groups == 2 and dtype == torch.bfloat16
        self.assertEqual(bmm.call_count, int(supported))
        self.assertEqual(actual.shape, expected.shape)
        if supported:
            self.assertTrue(actual.is_contiguous())
        torch.testing.assert_close(actual, expected, atol=0.016, rtol=0.01)
        print(
            f"rows={rows} padded={padded} groups={groups} dtype={dtype} "
            f"exact={torch.equal(actual, expected)} "
            f"max_abs={(actual.float() - expected.float()).abs().max().item()}",
            flush=True,
        )

    def test_small_and_large_rows(self):
        for rows in (9, 32, 128, 365, 511, 1024, 2048, 4095, 4096):
            for padded in (False, True):
                with self.subTest(rows=rows, padded=padded):
                    self.check(rows, padded)

    def test_fallbacks(self):
        for rows in (1, 2, 8):
            self.check(rows)
        self.check(365, groups=1)
        self.check(365, dtype=torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source
    unittest.main(argv=[sys.argv[0], *remaining])
