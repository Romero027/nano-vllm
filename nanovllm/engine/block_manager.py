"""Paged KV-cache block manager with prefix-caching support.

Physical GPU memory for the KV cache is divided into fixed-size *blocks*
(each holding ``block_size`` tokens worth of K/V tensors).  The BlockManager
maintains a virtual→physical mapping (the *block table*) for every active
sequence and implements hash-based **prefix caching**: blocks whose content
has been seen before can be reused instead of recomputed.

Block lifecycle
───────────────
  free  ──allocate──▶  used (ref_count ≥ 1)  ──deallocate──▶  free
                        ▲            │
                        └─ ref_count ┘  (shared by prefix caching)

A block is identified by a content hash (xxhash-64 over its token IDs
chained with the preceding block's hash).  When a new sequence's prefix
matches an existing block's hash *and* token content, the physical block
is shared (ref_count incremented) and the sequence skips recomputation
of those cached tokens.
"""

from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    """A single physical KV-cache block.

    Attributes
    ----------
    block_id   : int   – Immutable index into the pre-allocated block pool.
    ref_count  : int   – Number of sequences currently using this block.
                         Reaches 0 → eligible for deallocation.
    hash       : int   – Content hash (-1 while the block is still being
                         filled, i.e. it is the last partial block of a
                         sequence).
    token_ids  : list  – Token IDs stored in this block (used to verify
                         hash matches during prefix-cache lookups).
    """

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """Stamp the block with its content hash once fully filled."""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """Prepare the block for fresh use by a new sequence."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:
    """Manages physical KV-cache block allocation and prefix caching.

    The manager owns a flat pool of ``num_blocks`` Blocks and tracks which
    are free, which are in use, and a hash→block_id table that enables
    O(1) prefix-cache lookups.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """Hash a block's tokens, chaining with the previous block's hash.

        Chaining ensures that two blocks with the same local tokens but
        different preceding context produce different hashes, preventing
        false prefix-cache hits.
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """Move a block from the free pool into active use."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """Return a block to the free pool (does NOT clear its hash so it
        remains discoverable for future prefix-cache hits until evicted)."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """Check whether enough free blocks exist to fit the full sequence."""
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Build the block table for a newly scheduled sequence.

        Walks blocks left-to-right.  For each full block, computes the
        chained content hash and attempts a prefix-cache lookup:

        * **Cache hit** – reuses the existing physical block (bumps
          ref_count) and credits ``seq.num_cached_tokens`` so the model
          runner can skip those positions during prefill.
        * **Cache miss** – allocates a fresh block from the free pool.
          Once a miss occurs all subsequent blocks are also misses
          (``cache_miss`` flag), since their chained hashes cannot match.

        The last block may be partial (fewer than ``block_size`` tokens);
        its hash is left as -1 until it is completed during decoding
        (see ``may_append``).
        """
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            # Only compute a hash for fully filled blocks.
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # Block is actively held by another sequence — share it.
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # Block was evicted but its hash entry survived — reclaim.
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """Release all blocks held by a finished sequence.

        Iterates in reverse so that child blocks are freed before parents,
        keeping the free list in a natural eviction order.
        """
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """Check whether a decode step can proceed.

        A new free block is needed only when the latest token spills into
        a brand-new block (i.e. ``len(seq) % block_size == 1`` — one token
        past the boundary).
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """Bookkeeping after a single token has been appended to *seq*.

        Three cases based on where the new token lands within the block:

        1. **First token of a new block** (``len % block_size == 1``):
           The previous block is now complete; allocate a fresh block for
           the new token.
        2. **Last slot fills a block** (``len % block_size == 0``):
           The current block is now full — compute and record its content
           hash so it becomes available for future prefix-cache lookups.
        3. **Mid-block**: Nothing to do; the block is still partial.
        """
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
