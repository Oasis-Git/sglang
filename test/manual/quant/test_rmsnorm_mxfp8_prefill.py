"""Bitwise parity against FlashInfer 0.7 BF16 RMSNorm followed by MXFP8."""

import unittest

import flashinfer
import torch

from sglang.kernels.ops.layernorm.rmsnorm_mxfp8_prefill import rmsnorm_mxfp8_prefill


class TestRmsnormMxfp8Prefill(unittest.TestCase):
    @torch.inference_mode()
    def test_reference_parity(self):
        for m in [9, 32, 127, 128, 129, 365, 511, 1024, 2048, 4095, 4096, 8192, 16384]:
            for stride in [1280, 1792]:
                for scale in [0.0, 1e-3, 1.0, 1e3]:
                    with self.subTest(rows=m, stride=stride, scale=scale):
                        torch.manual_seed(m + stride)
                        x = (
                            torch.randn(m, stride, device="cuda", dtype=torch.bfloat16)
                            * scale
                        )[:, :1280]
                        weight = torch.randn(1280, device="cuda", dtype=torch.bfloat16)
                        expected_y = flashinfer.norm.rmsnorm(x, weight, 1e-6)
                        expected_q, expected_sf = flashinfer.mxfp8_quantize(
                            expected_y, is_sf_swizzled_layout=True
                        )
                        got = rmsnorm_mxfp8_prefill(x, weight, 1e-6)
                        for actual, expected in zip(
                            got, (expected_y, expected_q, expected_sf)
                        ):
                            self.assertTrue(
                                torch.equal(
                                    actual.view(torch.uint8).reshape(-1),
                                    expected.view(torch.uint8).reshape(-1),
                                )
                            )

    @torch.inference_mode()
    def test_runtime_epsilon_and_offset(self):
        torch.manual_seed(511)
        for eps in [1e-8, 1e-5, 1e-3]:
            # Nonzero storage offset with a packed QKV row stride.
            x = torch.randn(2060, 1792, device="cuda", dtype=torch.bfloat16)[
                :, 128:1408
            ]
            weight = torch.empty(1280, device="cuda", dtype=torch.bfloat16).uniform_(
                0.9, 1.1
            )
            expected_y = flashinfer.norm.rmsnorm(x, weight, eps)
            expected_q, expected_sf = flashinfer.mxfp8_quantize(
                expected_y, is_sf_swizzled_layout=True
            )
            for actual, expected in zip(
                rmsnorm_mxfp8_prefill(x, weight, eps),
                (expected_y, expected_q, expected_sf),
            ):
                self.assertTrue(
                    torch.equal(
                        actual.view(torch.uint8).reshape(-1),
                        expected.view(torch.uint8).reshape(-1),
                    )
                )


if __name__ == "__main__":
    unittest.main()
