# Block Manager: Paged KV-Cache Memory Management

The `BlockManager` is the memory allocator at the heart of nano-vllm's PagedAttention implementation. It divides GPU KV-cache memory into fixed-size **blocks** and maintains a virtual-to-physical mapping (the **block table**) for every active sequence. It also implements hash-based **prefix caching**, allowing sequences with shared prefixes to reuse previously computed KV entries.

This document covers the data structures, algorithms, and lifecycle of blocks managed by `BlockManager` (defined in `nanovllm/engine/block_manager.py`).

---

## Core Concepts

### Physical Blocks

The GPU KV-cache is a single pre-allocated tensor of shape `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`. Each **block** holds `block_size` tokens worth of K and V tensors (default 256 tokens). The `BlockManager` never touches the tensor directly — it only manages the **logical mapping** of which block IDs belong to which sequences.

### Block Tables

Every active sequence has a **block table**: an ordered list of physical block IDs. The model runner uses this table to scatter/gather KV entries during attention. For example, a 600-token sequence with `block_size=256` needs 3 blocks:

```
Sequence token stream:
┌────────────────┐ ┌────────────────┐ ┌──────────┐
│  tokens 0–255  │ │ tokens 256–511 │ │ 512–599  │
└───────┬────────┘ └───────┬────────┘ └────┬─────┘
        │                  │               │
   block_table[0]=7   block_table[1]=3  block_table[2]=12

Physical block pool:
  ... │ 3 │ ... │ 7 │ ... │ 12 │ ...
```

Blocks don't need to be contiguous in physical memory — that's the key insight of PagedAttention.

### Content Hashing and Prefix Caching

Each **full** block (exactly `block_size` tokens) receives a content hash computed with xxhash-64. The hash is **chained**: it incorporates the hash of the preceding block, so two blocks with identical local tokens but different preceding context produce different hashes. This prevents false cache hits.

```
hash(block_0) = xxh64(token_ids_0)
hash(block_1) = xxh64(hash(block_0) || token_ids_1)
hash(block_2) = xxh64(hash(block_1) || token_ids_2)
```

Partial blocks (the last block of a sequence that hasn't been filled to `block_size`) have a sentinel hash of `-1` and are never entered into the cache.

---

## Data Structures

### `Block`

A `Block` represents a single physical KV-cache slot:

| Field | Type | Description |
|-------|------|-------------|
| `block_id` | `int` | Immutable index into the block pool (0 to `num_blocks - 1`). |
| `ref_count` | `int` | Number of sequences currently sharing this block. Reaches 0 → eligible for deallocation. |
| `hash` | `int` | Content hash. `-1` while the block is still partial (unfilled). |
| `token_ids` | `list[int]` | Token IDs stored in this block, used to verify hash matches during prefix-cache lookups. |

A block transitions through three states:

```
free (ref_count=0, in free list)
  │
  ├── allocate ──▶ used (ref_count ≥ 1, in used set)
  │                  │
  │                  ├── shared (ref_count incremented by prefix cache hit)
  │                  │
  │                  └── deallocate (ref_count decremented)
  │                        │
  └────────────────────────┘  (when ref_count reaches 0)
```

### `BlockManager`

The manager maintains four data structures:

| Field | Type | Description |
|-------|------|-------------|
| `blocks` | `list[Block]` | Flat pool of all physical blocks. |
| `free_block_ids` | `deque[int]` | FIFO queue of available block IDs. New allocations pull from the front. |
| `used_block_ids` | `set[int]` | Block IDs currently held by at least one sequence. |
| `hash_to_block_id` | `dict[int, int]` | Maps content hash → physical block ID for O(1) prefix-cache lookups. |

---

## Operations

### `allocate(seq)` — Assigning Blocks to a New Sequence

Called when a sequence transitions from `WAITING` to `RUNNING` during prefill scheduling. It walks the sequence's tokens block-by-block from left to right:

```
for each logical block i in seq:
    token_ids = seq.block(i)

    if block is full (block_size tokens):
        compute chained hash
    else:
        hash = -1  (partial block, skip caching)

    look up hash in hash_to_block_id:
        HIT  → reuse the existing physical block
        MISS → allocate a fresh block from the free pool

    append physical block ID to seq.block_table
```

**Cache hit behavior** — When a hash matches an existing block *and* the token IDs are identical (guard against hash collisions), the sequence shares that physical block:

- If the block is in `used_block_ids` (actively held by another sequence), its `ref_count` is incremented.
- If the block was deallocated but its hash entry survived in `hash_to_block_id`, it is reclaimed from the free pool.
- The sequence's `num_cached_tokens` is incremented by `block_size`, telling the model runner to skip those positions during prefill.

**Cache miss cascade** — Once the first miss occurs, all subsequent blocks are guaranteed misses too, because the chained hash of any later block depends on the hash of the missed block.

### `deallocate(seq)` — Releasing Blocks

Called when a sequence finishes (hits EOS or `max_tokens`) or is preempted by the scheduler. Iterates through the block table **in reverse** so child blocks are freed before parents, keeping the free list in natural eviction order:

```
for each block_id in reversed(seq.block_table):
    block.ref_count -= 1
    if ref_count == 0:
        move block_id from used_block_ids to free_block_ids
```

The block's `hash` is intentionally **not** cleared. This means a deallocated block remains discoverable in `hash_to_block_id` — if a future sequence happens to need the same prefix, it can reclaim the block from the free pool instead of recomputing from scratch.

### `can_allocate(seq)` — Pre-allocation Check

Returns `True` if the free pool has at least `seq.num_blocks` blocks available. The scheduler calls this before committing to prefill a sequence.

### `can_append(seq)` — Pre-decode Check

During decode, each step appends exactly one token. A new physical block is only needed when that token crosses a block boundary (i.e., `len(seq) % block_size == 1` — the first token of a new block). Otherwise, the token fits in the existing last block.

### `may_append(seq)` — Post-decode Bookkeeping

Called after a token has been appended to a sequence. Handles three cases based on where the new token lands:

| Condition | Meaning | Action |
|-----------|---------|--------|
| `len(seq) % block_size == 1` | First token of a new block | Previous block is now full. Allocate a fresh block for the new token. |
| `len(seq) % block_size == 0` | Last slot fills a block | Block is now complete. Compute its chained hash and register it in `hash_to_block_id` for future prefix-cache lookups. |
| Otherwise | Mid-block | Nothing to do; the block is still partial. |

---

## Prefix Caching Walkthrough

Consider two requests processed sequentially:

**Request A**: `"Translate the following English text to French: ..."` (1000 tokens)

1. `allocate` assigns 4 blocks (with `block_size=256`): blocks 0–3.
2. All 4 blocks get fresh allocations (no cache yet).
3. Blocks 0–2 are full → their chained hashes are computed and stored in `hash_to_block_id`.
4. Block 3 is partial (1000 - 768 = 232 tokens) → hash stays `-1`.

**Request B**: `"Translate the following English text to French: ..."` (different text, but same 800-token system prompt)

1. `allocate` walks B's blocks left to right.
2. Blocks 0–2 (tokens 0–767) match A's hashes exactly → **cache hit**. B shares A's physical blocks, `ref_count` becomes 2 for each. B's `num_cached_tokens` = 768.
3. Block 3 differs (different user text) → **cache miss**. A new block is allocated.
4. During prefill, the model runner skips the first 768 positions and only computes the suffix.

```
Request A blocks:  [ 7 ][ 3 ][ 12 ][ 5 ]
                     ↑     ↑     ↑
Request B blocks:  [ 7 ][ 3 ][ 12 ][ 9 ]   ← blocks 7, 3, 12 shared
                                      ↑ new allocation
```

When A finishes and is deallocated, `ref_count` drops to 1 for the shared blocks (B still holds them). When B also finishes, `ref_count` drops to 0 and the blocks return to the free pool — but their hash entries persist, ready for a future request with the same prefix.

---

## Integration with the Scheduler

The scheduler (`nanovllm/engine/scheduler.py`) orchestrates all block manager calls:

| Scheduler action | Block manager call | Trigger |
|------------------|--------------------|---------|
| Admit sequence to prefill batch | `can_allocate` → `allocate` | Sequence moves from waiting to running |
| Prepare decode step | `can_append` → (after step) `may_append` | Each decode iteration |
| Preempt a sequence | `deallocate` | Memory pressure during decode |
| Finish a sequence | `deallocate` | EOS token or `max_tokens` reached |

The scheduler checks `can_allocate` / `can_append` **before** committing to run a sequence. If neither succeeds, it triggers preemption — evicting the last running sequence back to the waiting queue to free blocks.

---

## Summary

The `BlockManager` provides three key capabilities:

1. **Non-contiguous KV-cache allocation** — Sequences don't need contiguous GPU memory. Blocks are assigned from a shared pool and mapped via block tables, eliminating internal fragmentation.

2. **Prefix caching** — Chained content hashes enable O(1) lookups for reusable KV blocks. Shared blocks avoid redundant computation and reduce memory usage when multiple requests share common prefixes (system prompts, few-shot examples, etc.).

3. **Reference-counted sharing** — Multiple sequences can safely share physical blocks. Blocks are only freed when their last user finishes, and even then their hash entries persist for future reuse.
