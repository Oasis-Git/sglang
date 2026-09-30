# DSv4.1 optimization lab

This is an evolving local experiment branch, **not a merge candidate**.
Start point: Oasis-Git `main` at `5781206833d3296f6dc6846a9eb1ac9eab0cd522`.
Exact patch revisions are pinned in [manifest.json](manifest.json).

## Included optimizations

- PR #41657: fuse BF16 Q-RoPE output into `fused_q_norm_rope`.
- PR #41658: faster FP4 index-K gather and combined sparse indices.
- PR #41659: prefill indexer selection and scheduling; remove host syncs.
- PR #41660: eager-extend c1/c2 fusion and faster c2 decode.
- PR #41603: opt-in TRT-LLM sparse attention for DSv4.1.

The first four use the exact combined patch previously tested for AgentX and
8K profiling, including its Engram host-sync changes. Runtime settings such as
prefill/decode interval are launch choices, not hard-coded branch defaults.

## Attention variants

Keep the existing launch configuration for the FlashMLA control. To select the
TRT-LLM candidate, append:

```sh
--dsv4-attn-backend trtllm --cuda-graph-backend-prefill disabled
```

For a controlled backend comparison, use `--cuda-graph-backend-prefill disabled`
for both runs. The candidate retains the PR's restrictions: Blackwell CUDA,
FP8 KV, positive chunk size, no context parallelism, no PD disaggregation, and
no SWA bounded replay. Default backend selection is unchanged.

## Tokenizer runtime option

Install `fastokens==0.1.2` and append `--tokenizer-backend fastokens` to opt in.
The standalone fix `c61e671f85` (based on `Oasis-Git/main`) forwards this choice through the DSv4.1 processor
while preserving the explicit HuggingFace tokenizer used for Engram hashing.
The manifest records exact-token and full Engram-map parity checks separately
from full-model and combined-lab validation. Defaults are unchanged.

## Integration details

PR #41657 removed `q_rope_store.py`, whereas #41603 needs its FP8 output path.
The combined branch retains that helper for FP8 only. BF16 Q continues through
`fused_q_norm_rope`; FP8 Q preserves BF16 RoPE rounding before the FP8 cast.
The regression test covers decode-sized and 8K prefill inputs, strided Q and
padded output, both position integer widths, and exact reference equality.

The combined TRT-LLM serving path has now completed the 8K prefill experiment
below. This is a serving and kernel validation, not a model-quality evaluation.
Earlier measurements of the four-PR stack are not results for this integration.

The standalone sparse-indexer pairing optimization overlaps PR #41659: this lab
already creates bounded row-pair IDs. Pass those IDs through unchanged; applying
a second request-to-pair conversion indexes out of bounds on long prefills.
The regression test `test_dsv41_sparse_pair_ids.py` covers long and odd request
lengths, empty request segments, and sliced tails.

## Updating this branch

Append scoped optimization commits and update the manifest with exact source
revisions and validation results. Keep baseline and candidate run configurations
explicit. Preserve the local test-only status; do not merge into main or push
this branch without the user's explicit instruction.

## Opt-in Mega mHC prefill experiment

Set `SGLANG_OPT_DSV41_MEGA_MHC_PREFILL=1` to fuse the residual update,
shifted collapse/RMSNorm, and next-sublayer mixing statistics with DeepGEMM
`mega_mhc`. The flag defaults off. This experiment requires an installed
DeepGEMM exposing that entry point. It handles eager Blackwell DSv4.1 prefill
with HC4, hidden size 5120, and 4K–64K rows. Decode/verify, CP/SP/DP-attention,
prefill graphs, and late-layer tail selection keep the existing path.
Engram boundaries invalidate the hand-off; DSPARK captures materialized residuals.

On the same 4xGB300 node, BS1, 8192 identical input tokens, one output token,
zero cached tokens, five warmups and ten timed requests:

| Measurement | Baseline | Mega mHC |
| --- | ---: | ---: |
| Median HTTP latency | 218.08 ms | 203.17 ms |
| Median engine prefill | 212.94 ms | 198.72 ms |
| Matched named mHC GPU sum | 19.83 ms | 13.13 ms |
| mHC kernel executions/request | 243 | 89 |
| Non-collective prefill GPU sum | 95.73 ms | 89.00 ms |

Both runtimes used the existing FlashInfer 0.7.0/Cutlass DSL 4.7.1 overlay,
TRT-LLM Q16 attention, interval 4, and prefill graphs disabled. Dependency pins
in this branch are unchanged by the mHC experiment. Kernel totals come from
three profiled requests on each of four ranks; baseline kernels use the preceding
FlashInfer-0.7 capture. Latency was measured separately before any profiler in
each process. Kernel sums can overlap and are not critical-path wall time.

The numerical test is
`test/manual/kernels/attention/test_dsv41_mega_mhc_prefill.py` (run directly with
Python). It checks two consecutive boundaries at 4K/8K/16K against the production
decomposition. All three tests pass with BF16 atol=0.016/rtol=0.01 and FP32
coefficient atol=0.00002/rtol=0.001. All 30 server requests return the same greedy
output token. This does not substitute for a full quality evaluation.

Development-only Engram annotations activate under torch profiling. Across the
two Engram layers, the per-request GPU sums are 0.23 ms host gather, 0.63 ms
lookup all-reduce (including waits), 1.51 ms projection, and 0.41 ms gate. The
gather reads host memory directly on the GPU; no separate CPU lookup/H2D copy
stage exists in this per-rank configuration. Shared hash/history preparation is
not included in these four annotated stage groups.
