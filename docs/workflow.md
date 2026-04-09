# Nano-vLLM Workflow: Life of a Request

This document describes how nano-vllm processes inference requests end-to-end, from prompt submission through token generation to final output. The design mirrors the core ideas behind [vLLM](https://github.com/vllm-project/vllm): **PagedAttention** for memory-efficient KV caching, **continuous batching** with a scheduler, and **tensor parallelism** for multi-GPU execution.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                      LLMEngine                          │
│                                                         │
│  ┌──────────┐   ┌───────────┐   ┌───────────────────┐   │
│  │Tokenizer │   │ Scheduler │   │   ModelRunner      │  │
│  │(HF)      │   │           │   │                    │  │
│  └──────────┘   │ ┌───────┐ │   │  Qwen3ForCausalLM  │  │
│                 │ │Block  │ │   │  Attention (Flash) │  │
│                 │ │Manager│ │   │  Paged KV Cache    │  │
│                 │ └───────┘ │   │  Sampler           │  │
│                 └───────────┘   └───────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Tensor Parallelism (rank 0 drives, 1..N workers)│   │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

Four subsystems cooperate:

| Subsystem | Responsibility |
|-----------|---------------|
| **Tokenizer** | Encodes string prompts into token IDs and decodes generated IDs back to text. |
| **Scheduler** | Decides which sequences run each step (prefill vs. decode), manages block allocation via `BlockManager`, and handles preemption when memory is tight. |
| **ModelRunner** | Loads the model, manages the paged KV cache, prepares GPU tensors, runs forward passes, and invokes the sampler. |
| **Tensor Parallelism** | Spawns worker processes (ranks 1..N-1) that mirror rank 0's forward pass through NCCL collectives. |

---

## Initialization

When a user creates an `LLM` instance:

```python
llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)
```

The following happens inside `LLMEngine.__init__`:

1. **Config construction** — A `Config` dataclass is built from the model path and keyword arguments. It loads the HuggingFace model config (`AutoConfig`), validates constraints (block size alignment, TP size, max sequence length), and stores parameters like `max_num_batched_tokens`, `gpu_memory_utilization`, and `kvcache_block_size`.

2. **Tensor Parallelism setup** — For TP > 1, worker processes (ranks 1..N-1) are spawned using `torch.multiprocessing` with the `"spawn"` context. Each worker creates its own `ModelRunner` and enters an event-driven loop, waiting for rank 0 to signal operations.

3. **Rank 0 ModelRunner** — The driver process creates the rank-0 `ModelRunner`, which:
   - Initializes the NCCL process group for collective communication.
   - Loads the `Qwen3ForCausalLM` model and distributes weights across TP shards.
   - Allocates the **paged KV cache** — a single large tensor `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]` sized to fill the available GPU memory up to `gpu_memory_utilization`.
   - Optionally captures **CUDA graphs** for decode batches (skipped when `enforce_eager=True`).

4. **Tokenizer** — A HuggingFace `AutoTokenizer` is loaded. The EOS token ID is stored in `Config` so downstream components can detect stop conditions without needing the tokenizer.

5. **Scheduler** — Created with the config. Internally instantiates a `BlockManager` to manage physical KV cache blocks.

---

## Life of a Request

### Phase 1: Submission

```python
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

`generate()` is a convenience wrapper that:

1. Calls `add_request()` for each prompt.
2. Runs the `step()` loop until all sequences finish.
3. Re-orders and decodes the results.

Inside `add_request()`:
- If the prompt is a string, it's tokenized into a list of token IDs.
- A `Sequence` object is created with a unique `seq_id`, the prompt tokens, and the sampling parameters.
- The sequence is appended to the scheduler's **waiting queue** with status `WAITING`.

### Phase 2: Scheduling

Each call to `step()` begins with `scheduler.schedule()`, which decides **what to run and how**.

The scheduler operates in one of two modes each step:

#### Prefill Mode (processing new prompts)

When the waiting queue is non-empty, the scheduler tries to batch as many waiting sequences as possible:

```
for each sequence in waiting queue:
    if adding it would exceed max_num_seqs or max_num_batched_tokens:
        break
    if BlockManager cannot allocate blocks for it:
        break
    allocate physical KV blocks (with prefix caching)
    mark sequence as RUNNING
    move to running queue
```

**Prefix caching**: During block allocation, the `BlockManager` hashes each full block of tokens (using xxhash with chained prefixes). If a matching hash already exists in the cache, that physical block is reused and the sequence's `num_cached_tokens` is incremented. Only the uncached suffix needs to be computed in the forward pass.

The scheduler returns `(scheduled_seqs, is_prefill=True)`.

#### Decode Mode (generating new tokens)

When no waiting sequences can be scheduled, the scheduler runs decode on the running queue:

```
for each sequence in running queue:
    if BlockManager can append a new token:
        reserve the slot
        add to scheduled batch
    else:
        preempt another sequence to free blocks
```

**Preemption**: If memory is exhausted, a running sequence is evicted — its status reverts to `WAITING`, its blocks are deallocated, and it's pushed back to the front of the waiting queue. This implements vLLM's preemptive scheduling to avoid OOM.

The scheduler returns `(scheduled_seqs, is_prefill=False)`.

### Phase 3: Model Execution

`ModelRunner.run()` receives the scheduled sequences and the prefill/decode flag.

#### Input Preparation

**For prefill:**
- Token IDs and positions are flattened across all sequences, excluding cached prefix tokens.
- Variable-length attention metadata is built: `cu_seqlens_q`, `cu_seqlens_k`, `max_seqlen_q`, `max_seqlen_k`.
- A `slot_mapping` tensor maps each token to its physical KV cache slot (block_id × block_size + offset).
- If any prefix cache hits exist, `block_tables` are set so the attention kernel can read cached KV entries.

**For decode:**
- Each sequence contributes exactly **one token** (its last generated token).
- `positions` is set to the current sequence length minus one.
- `context_lens` tracks how many KV entries each sequence has.
- `slot_mapping` points to the single new slot for each sequence.
- `block_tables` maps logical blocks to physical blocks for KV cache lookups.

All metadata is packed into a global **`Context`** object (`set_context()`) so that attention layers and the LM head can access it without explicit parameter passing.

#### Forward Pass

The model (`Qwen3ForCausalLM`) processes the input through:

1. **Embedding** (`VocabParallelEmbedding`) — Token IDs → hidden states. Vocabulary is sharded across TP ranks.

2. **Decoder layers** (×N) — Each layer applies:
   - **RMSNorm** (pre-attention, fused with residual)
   - **Self-attention** (`Qwen3Attention`):
     - QKV projection via `QKVParallelLinear` (TP-sharded)
     - **Rotary positional embedding** (RoPE)
     - Optional Q/K normalization
     - **Paged KV cache write**: A Triton kernel (`store_kvcache`) scatters K and V vectors into their physical slots in the paged cache tensor.
     - **Flash Attention**:
       - *Prefill*: `flash_attn_varlen_func` — variable-length flash attention across the batched sequences. If prefix cache blocks exist, they're passed as paged KV.
       - *Decode*: `flash_attn_with_kvcache` — single-query attention reading from the paged KV cache.
     - Output projection via `RowParallelLinear` (TP all-reduce)
   - **RMSNorm** (pre-MLP, fused with residual)
   - **MLP** (`Qwen3MLP`): gate-up projection (`MergedColumnParallelLinear`) → SwiGLU activation → down projection (`RowParallelLinear`)

3. **Final norm** — RMSNorm on the last hidden states.

4. **LM Head** (`ParallelLMHead`) — Projects hidden states to vocabulary logits. During prefill, only the **last token** of each sequence is projected (using `cu_seqlens_q` offsets) since only that position's logits are needed for next-token prediction. Logits are gathered across TP ranks to form the full vocabulary distribution.

#### CUDA Graph Optimization

For decode steps with small batch sizes (≤512) and when `enforce_eager=False`, the forward pass replays a pre-captured **CUDA graph** instead of running eager execution. The context metadata is copied into pinned graph buffers before replay.

### Phase 4: Sampling

On rank 0 only, the `Sampler` converts logits into token selections:

1. Logits are scaled by `1/temperature` (per-sequence temperatures).
2. Softmax produces a probability distribution.
3. A **Gumbel-max trick** samples from the distribution: divide probabilities by samples from `Exponential(1)`, then take `argmax`. This is equivalent to categorical sampling but is efficiently compiled via `@torch.compile`.

> Note: Greedy decoding (temperature=0) is not supported. `SamplingParams` enforces `temperature > 0`.

### Phase 5: Postprocessing

Back in the scheduler, `postprocess()` handles the sampled tokens:

```
for each sequence, sampled_token in zip(seqs, token_ids):
    sequence.append_token(sampled_token)
    if sampled_token == EOS (and not ignore_eos):
        mark FINISHED, deallocate blocks
    if num_completion_tokens >= max_tokens:
        mark FINISHED, deallocate blocks
```

Finished sequences are removed from the running queue. Their completion token IDs are collected and returned to the caller.

### Phase 6: Output Assembly

In `generate()`, the step loop continues until `scheduler.is_finished()` returns `True` (both waiting and running queues are empty). Results are:

1. Collected in a dict keyed by `seq_id`.
2. Sorted back to the original prompt order.
3. Decoded via the tokenizer into text strings.
4. Returned as a list of `{"text": ..., "token_ids": ...}` dicts.

---

## Continuous Batching

Unlike static batching (where all sequences in a batch must finish before new ones start), nano-vllm uses **continuous batching**:

- Sequences that finish early are immediately removed and their KV blocks freed.
- New sequences from the waiting queue enter prefill in subsequent steps.
- The scheduler alternates between prefill and decode steps based on queue state: if any sequences are waiting **and** blocks are available, prefill runs; otherwise decode runs.

This means the GPU stays busy and memory is recycled efficiently.

---

## PagedAttention and Block Management

PagedAttention is the core memory optimization. Instead of allocating a contiguous KV cache per sequence (wasteful due to variable lengths), nano-vllm manages KV memory in fixed-size **blocks** (default 256 tokens each):

```
Physical KV Cache Tensor
┌──────────────────────────────────────────┐
│ Block 0 │ Block 1 │ Block 2 │ Block 3 │ ...
└──────────────────────────────────────────┘
     ↑          ↑                   ↑
     │          │                   │
  Seq A[0]   Seq A[1]           Seq B[0]     ← logical-to-physical mapping
```

**Key operations:**

| Operation | When | What happens |
|-----------|------|-------------|
| `allocate` | Sequence enters prefill | Assign physical blocks; reuse hash-matched blocks (prefix cache) |
| `may_append` | Decode step | Reserve a slot in the current block, or allocate a new block when the current one is full |
| `deallocate` | Sequence finishes or is preempted | Decrement block ref counts; blocks with zero refs return to the free list |

**Prefix caching** works at block granularity: each full block of tokens gets a content-based hash. If two sequences share the same prefix (down to a block boundary), they share the same physical KV block, avoiding redundant computation and storage.

---

## Tensor Parallelism

For multi-GPU inference, nano-vllm splits the model across devices:

- **Column-parallel linear layers** (QKV projections, gate/up projections, embeddings) shard the output dimension across ranks.
- **Row-parallel linear layers** (attention output projection, MLP down projection) shard the input dimension and perform an **all-reduce** to combine partial results.
- The **LM head** shards the vocabulary and gathers logits on rank 0 for sampling.

Rank 0 acts as the driver: it runs the scheduler, prepares inputs, and signals worker ranks via multiprocessing `Event` objects before each collective operation. Workers execute the same forward pass in lockstep through an RPC-style event loop over shared memory.

---

## Summary: Request Lifecycle

```
User prompt
    │
    ▼
[Tokenize] ──→ token IDs
    │
    ▼
[Sequence created, added to waiting queue]
    │
    ▼
┌─────────── step() loop ───────────┐
│                                    │
│  ┌──────────┐                      │
│  │ Schedule │                      │
│  │ (prefill │◄── waiting queue     │
│  │ or decode)│◄── running queue    │
│  └────┬─────┘                      │
│       │                            │
│       ▼                            │
│  ┌──────────┐                      │
│  │ Prepare  │ build GPU tensors,   │
│  │ inputs   │ set Context          │
│  └────┬─────┘                      │
│       │                            │
│       ▼                            │
│  ┌──────────┐                      │
│  │ Forward  │ Embedding → Layers   │
│  │ pass     │ → Attention (Flash)  │
│  │          │ → KV Cache (Paged)   │
│  │          │ → MLP → LM Head      │
│  └────┬─────┘                      │
│       │                            │
│       ▼                            │
│  ┌──────────┐                      │
│  │ Sample   │ temperature scaling  │
│  │          │ + Gumbel-max trick   │
│  └────┬─────┘                      │
│       │                            │
│       ▼                            │
│  ┌──────────┐                      │
│  │ Post-    │ append token, check  │
│  │ process  │ EOS / max_tokens     │
│  └────┬─────┘                      │
│       │                            │
│       ▼                            │
│  finished? ──no──→ next step()     │
│       │                            │
│      yes                           │
└───────┼────────────────────────────┘
        │
        ▼
[Decode token IDs → text]
        │
        ▼
   Return output
```
