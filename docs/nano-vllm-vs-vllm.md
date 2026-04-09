# Nano-vLLM vs Production vLLM

Nano-vLLM implements the core architectural ideas behind [vLLM](https://github.com/vllm-project/vllm) in a compact, readable codebase (~1500 lines of Python). This document describes what it keeps, what it simplifies, and what it omits entirely.

---

## At a Glance

| Area | Nano-vLLM | Production vLLM |
|------|-----------|-----------------|
| **Scheduling** | Prefill-or-decode per step | Chunked prefill — prefill and decode mixed in one step |
| **Preemption** | Recompute only | Recompute or swap (KV offload to CPU) |
| **KV cache** | Paged blocks + prefix caching | Paged blocks + prefix caching + KV offload + speculative cache management |
| **Sampling** | Temperature only (Gumbel-max) | Temperature, top-k, top-p, min-p, repetition/presence/frequency penalties, beam search, best-of-N |
| **Models** | Qwen3 only | 50+ architectures (LLaMA, Mistral, GPT-NeoX, Gemma, …) |
| **Quantization** | None (FP16/BF16 only) | GPTQ, AWQ, SqueezeLLM, FP8, GGUF, Marlin, … |
| **Parallelism** | Tensor parallelism | Tensor + pipeline + expert parallelism, multi-node |
| **Serving** | Offline `generate()` only | OpenAI-compatible HTTP server, async streaming, multi-user |
| **Speculative decoding** | Not implemented | Draft-model, Medusa, Eagle, ngram, MLPSpeculator |
| **LoRA** | Not supported | Multi-LoRA serving with adapter hot-swapping |

---

## Scheduling

### Nano-vLLM: Prefill-First, One Mode Per Step

The scheduler operates in two mutually exclusive phases each step:

1. **Phase 1 (Prefill)** — Walk the waiting queue and greedily admit sequences that fit within `max_num_seqs` and `max_num_batched_tokens`. If any are admitted, return immediately — this step is pure prefill.
2. **Phase 2 (Decode)** — Only reached when no waiting sequence fits. Schedule all running sequences for one decode token each.

A batch is always homogeneous: every sequence is either prefilling or decoding.

### Production vLLM: Chunked Prefill

Production vLLM breaks large prompts into **chunks** (e.g., 2048 tokens) and packs prefill chunks and decode tokens into a single batch up to a combined token budget. A single step might contain:

```
seq_A: 512 prefill tokens (chunk 3 of 6)
seq_B: 1 decode token
seq_C: 1 decode token
seq_D: 1024 prefill tokens (chunk 1 of 2)
```

**Why it matters:** In nano-vLLM, a 10K-token prefill blocks all decode sequences for that step, causing latency spikes. Chunked prefill interleaves the two, keeping time-to-first-token (TTFT) bounded and decode latency smooth.

**Complexity cost:** The scheduler must track per-sequence chunk offsets, the model runner must handle mixed attention masks (causal for prefill positions, single-query for decode positions), and the block manager must allocate partial-prefill blocks.

---

## Preemption Strategy

### Nano-vLLM: Recompute Only

When the KV cache is full during decode, the scheduler evicts the youngest running sequence (LIFO) back to the front of the waiting queue. All its blocks are freed. When re-admitted, the entire prompt is re-prefilled from scratch.

### Production vLLM: Recompute or Swap

Production vLLM supports a second strategy: **swap**, which offloads evicted KV blocks to CPU memory and restores them later when GPU blocks become available. This avoids re-computing the full prompt at the cost of CPU memory and PCIe transfer time.

---

## Sampling

### Nano-vLLM: Temperature + Gumbel-Max

Nano-vLLM supports only temperature-scaled sampling with `temperature > 0`. It uses the Gumbel-max trick (divide softmax probabilities by Exponential(1) samples, then argmax) compiled via `@torch.compile`. There is no greedy mode (temperature=0 is disallowed).

### Production vLLM: Full Sampling Suite

Production vLLM supports a rich set of sampling strategies:

- **Greedy** (temperature=0)
- **Top-k, top-p (nucleus), min-p** filtering
- **Repetition, presence, and frequency penalties**
- **Beam search** with configurable beam width
- **Best-of-N** sampling (generate N candidates, return the best)
- **Guided decoding** (constrained output via grammar/JSON schema)
- **Logprobs** (return per-token log probabilities)

---

## Model Support and Weight Loading

### Nano-vLLM: Single Architecture

Nano-vLLM hardcodes `Qwen3ForCausalLM` and loads weights from a local directory of safetensors files. The weight loader uses a `packed_modules_mapping` to handle fused QKV and gate-up projections, and each parameter has a custom `weight_loader` method for TP sharding.

### Production vLLM: 50+ Architectures

Production vLLM has a model registry with 50+ architectures, automatic architecture detection from HuggingFace configs, support for multiple weight formats (safetensors, PyTorch, GGUF), and quantized weight loaders for every major quantization scheme.

---

## Quantization

Nano-vLLM runs in FP16/BF16 only. Production vLLM supports GPTQ, AWQ, SqueezeLLM, FP8 (both weights-only and W8A8), GGUF, Marlin (GPU-optimized GPTQ kernels), and bitsandbytes — enabling models 2–4x larger to fit in the same GPU memory.

---

## Parallelism

### Nano-vLLM: Tensor Parallelism Only

Nano-vLLM supports tensor parallelism (TP) across GPUs on a single node. Column-parallel layers shard the output dimension, row-parallel layers shard the input and all-reduce. Worker processes are coordinated via shared memory + pickle + multiprocessing Events with NCCL for collectives. Rendezvous is hardcoded to `tcp://localhost:2333`.

### Production vLLM: TP + PP + EP + Multi-Node

Production vLLM supports:

- **Pipeline parallelism (PP):** Splits model layers across GPUs, overlapping compute and communication.
- **Expert parallelism (EP):** For mixture-of-experts models, distributes experts across GPUs.
- **Multi-node:** Distributed across machines via Ray or native NCCL with proper rendezvous.
- **Disaggregated prefill/decode:** Separate GPU pools for prefill and decode workloads.

---

## Serving and API

### Nano-vLLM: Offline Batch Only

Nano-vLLM exposes a synchronous `generate()` method that blocks until all sequences finish, then returns the full text. There is no HTTP server, no streaming, no async path. The `step()` method is documented as a building block for streaming, but no streaming interface is built on top of it.

### Production vLLM: Full Serving Stack

Production vLLM ships an **OpenAI-compatible API server** built on FastAPI/uvicorn with:

- Async engine with non-blocking request submission
- Server-sent events (SSE) for token streaming
- `/v1/completions` and `/v1/chat/completions` endpoints
- Multi-user concurrent request handling
- Request cancellation and abort
- Usage metrics (Prometheus)
- Request-level logging and tracing

---

## CUDA Graphs

### Nano-vLLM: Decode Only, Fixed Ladder

Nano-vLLM captures CUDA graphs for decode batches at a fixed set of batch sizes (1, 2, 4, ..., up to `max_num_seqs`, capped at 512). Prefill always runs eagerly because prompt lengths vary widely. Inputs are copied into pre-allocated static buffers before graph replay.

### Production vLLM: Broader Graph Coverage

Production vLLM has a more sophisticated CUDA graph strategy with padding to fit captured sizes, automatic graph invalidation when batch composition changes, and piecewise CUDA graphs that handle more forward-pass variations.

---

## Speculative Decoding

Not present in nano-vLLM. Production vLLM supports multiple speculative decoding strategies to improve decode throughput:

- **Draft model:** A smaller model proposes candidate tokens verified by the main model.
- **Medusa / Eagle:** Adds lightweight prediction heads for parallel token proposals.
- **N-gram:** Uses token frequency patterns as a zero-cost draft.
- **MLPSpeculator:** A small MLP that predicts future tokens.

These can yield 2–3x decode speedup on memory-bandwidth-bound workloads.

---

## LoRA and Adapter Serving

Not present in nano-vLLM. Production vLLM supports serving multiple LoRA adapters simultaneously, with per-request adapter selection and hot-swapping — enabling a single deployment to serve many fine-tuned variants of a base model.

---

## Prefix Caching

Both systems implement prefix caching at block granularity using content-based hashing. Nano-vLLM uses chained xxhash (each block's hash includes the previous block's hash as a prefix). When sequences share a common prompt prefix aligned to block boundaries, they share physical KV blocks.

The core mechanism is the same. Production vLLM adds LRU eviction policies, cache-aware scheduling, and more sophisticated cache management for multi-turn conversations.

---

## Summary

Nano-vLLM captures the three foundational ideas of vLLM — **PagedAttention**, **continuous batching**, and **tensor parallelism** — in a codebase small enough to read in an afternoon. What it omits are the features needed for production serving at scale: mixed prefill/decode scheduling, rich sampling, model breadth, quantization, speculative decoding, multi-node parallelism, and a full API server. These omissions are intentional: they allow the core architecture to stay visible without being buried under the engineering complexity that a production system demands.
