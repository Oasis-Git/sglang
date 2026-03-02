# Piecewise CUDA Graph

## Motivation

Standard CUDA graphs capture the entire model forward pass as a single graph. This works well for decode (fixed batch size), but not for extend/prefill where the number of tokens varies across iterations.

Piecewise CUDA Graph (PCG) solves this by splitting the model's computation graph into pieces (roughly one per layer) at "split points" (e.g., MoE dispatch ops). Each piece is captured as a separate CUDA graph for a set of pre-defined token lengths. At runtime, the input is padded to the nearest captured size, and each piece is replayed. This eliminates kernel launch overhead for prefill/extend while still supporting dynamic shapes.

Recently we **enabled PCG by default**, which means that the old `--enable-piecewise-cuda-graph` flag is deprecated. Use `--disable-piecewise-cuda-graph` to turn it off.

## Usage

PCG is enabled by default for supported configurations. No extra flags needed:

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct
```

### Disable PCG

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --disable-piecewise-cuda-graph
```

### Custom capture sizes

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --piecewise-cuda-graph-max-tokens 2048
```

### Server Args

| Argument | Default | Description |
|---|---|---|
| `--disable-piecewise-cuda-graph` | `False` | Disable PCG for extend/prefill. |
| `--enforce-piecewise-cuda-graph` | `False` | Force-enable PCG, skipping all auto-disable conditions. For testing only. |
| `--piecewise-cuda-graph-max-tokens` | `None` (auto) | Maximum token count to capture. Defaults to `chunked_prefill_size` (non-MLA) or `2048` (MLA). |
| `--piecewise-cuda-graph-tokens` | `None` (auto) | Explicit list of token lengths to capture. Auto-generated if not set. |
| `--piecewise-cuda-graph-compiler` | `"eager"` | Compiler backend for the captured subgraphs. Choices: `eager`, `inductor`. |
| ~~`--enable-piecewise-cuda-graph`~~ | — | **Deprecated.** PCG is now enabled by default. Use `--enforce-piecewise-cuda-graph` to skip auto-disable conditions. |

## Bug Report

PCG is enabled by default but is still in an experimental stage. Since PCG relies on `torch.compile` to trace the model's forward pass, most bugs are introduced by torch compile tracing failures (e.g., untraceable ops, dynamic control flow, or graph breaks). If you encounter any issues related to PCG, please disable it by adding `--disable-piecewise-cuda-graph` to your launch command and report the bug at [GitHub Issues](https://github.com/sgl-project/sglang/issues/new/choose). We greatly appreciate your help in improving this feature.

### For Users

If you see an error message like the following during server startup, it is a PCG bug:

```
Piecewise CUDA Graph is enabled by default as an experimental feature.
To work around this error, add --disable-piecewise-cuda-graph to your launch command.
Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose
```

To work around it, add `--disable-piecewise-cuda-graph` to your launch command. When filing a bug report, please include:
1. The full error traceback
2. Model name and quantization method
3. Launch command with all arguments
4. GPU type and driver version

### For Developers

Since PCG relies on `torch.compile` to trace the model's forward pass, newly developed CUDA kernels (both JIT kernels and sgl-kernels) are typically not compatible with `torch.compile` out of the box. The tracing will fail on untraceable operations such as JIT compilation, file I/O, or dynamic module loading inside the kernel.

To make a kernel compatible with PCG, you need to register it as a custom op using `register_custom_op` from `sglang.srt.utils.custom_op`. This wraps the kernel as an opaque node in the compiled graph so that `torch.compile` will not trace inside it.

**Example usage (JIT kernel):**

```python
from sglang.srt.utils.custom_op import register_custom_op

# Inplace operator (no return value)
@register_custom_op(mutates_args=["output_q", "output_s"])
def per_token_group_quant_8bit(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
) -> None:
    # kernel implementation ...
```

**Example usage (operator with output):**

```python
# out_shape indicates which argument has the same shape as the output
@register_custom_op(mutates_args=["x"], out_shape=0)
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x.add_(y)
```

For wrapping external library functions (e.g., FlashInfer kernels), use `register_custom_op_from_extern` instead. See `python/sglang/srt/utils/custom_op.py` for full API documentation.

## How it works
### Torch compile backend

### Piecewise cuda graph runner

### Memory optimization

The memory cost of PCG comes from two parts: **torch memory allocator** and **non-torch memory**.

The torch memory allocator overhead is trivial thanks to several optimizations: a global shared memory pool is reused across all CUDA graph runners and capture sizes, capture is done in reverse order (large to small) so smaller graphs reuse memory allocated by larger ones, and output tensors of the last subgraph are stored as weak references to maximize memory reuse.

The main memory overhead comes from non-torch memory — the CUDA graph objects themselves require GPU memory to store the recorded kernel launch parameters and internal state. This overhead scales with the number of captured sizes, which is why `piecewise_cuda_graph_max_tokens` is capped conservatively by default.

### Shape configuration


## Compatibility

PCG is auto-disabled in the following scenarios. We are actively working on expanding compatibility — support for many of these will be coming soon.

- Disabled model architectures (e.g., `DeepseekV32ForCausalLM`)
- Speculative decoding
- DP attention
- Pipeline parallelism (`pp_size > 1`)
- Non-CUDA hardware (AMD ROCm, Ascend NPU)
- MoE A2A backend
- LoRA
- Multimodal / VLM models
- DLLM (diffusion LLM)
- Deterministic inference
- PD disaggregation
- Expert distribution recorder / EPLB

Use `--enforce-piecewise-cuda-graph` to skip all auto-disable checks (for testing/debugging only).

## Code Reference

| File | Description |
|---|---|
| `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py` | Main runner: init, capture, replay |
| `python/sglang/srt/compilation/compile.py` | `install_torch_compiled` trampoline |
| `python/sglang/srt/compilation/backend.py` | `SGLangBackend`, graph splitting, piecewise compilation |
| `python/sglang/srt/compilation/cuda_piecewise_backend.py` | Per-subgraph CUDA graph capture/replay |
| `python/sglang/srt/compilation/piecewise_context_manager.py` | Global context flags and `ForwardContext` |
| `python/sglang/srt/compilation/compilation_config.py` | Capture sizes, split ops, compiler config |
| `python/sglang/srt/utils/custom_op.py` | `register_custom_op` for torch.compile compatibility |
| `python/sglang/srt/server_args.py` | Server arguments and auto-disable logic |