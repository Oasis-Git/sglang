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
The standalone fix `81204764c2` forwards this choice through the DSv4.1 processor
while preserving the explicit HuggingFace tokenizer used for Engram hashing.
The manifest records exact-token and full Engram-map parity checks separately
from full-model and combined-lab validation. Defaults are unchanged.

## Integration details

PR #41657 removed `q_rope_store.py`, whereas #41603 needs its FP8 output path.
The combined branch retains that helper for FP8 only. BF16 Q continues through
`fused_q_norm_rope`; FP8 Q preserves BF16 RoPE rounding before the FP8 cast.
The regression test covers decode-sized and 8K prefill inputs, strided Q and
padded output, both position integer widths, and exact reference equality.

GPU unit checks do not validate the complete combined serving path. Full-model
TRT-LLM correctness and performance on this branch must be established before
claiming a speedup. Earlier measurements of the four-PR stack are not results
for this five-PR integration.

## Updating this branch

Append scoped optimization commits and update the manifest with exact source
revisions and validation results. Keep baseline and candidate run configurations
explicit. Preserve the local test-only status; do not merge into main or push
this branch without the user's explicit instruction.
