"""Scheduler: the central orchestrator of request batching and KV-cache lifecycle.

The scheduler decides *which* sequences run in each engine step and in *what
mode* (prefill vs. decode).  It owns two FIFO queues:

    waiting ──(prefill)──▶ running ──(eos / max_tokens)──▶ finished
                               ▲                                │
                               └──────(preempt)─────────────────┘

Design principles
-----------------
1. **Prefill-first.**  If any waiting sequences can fit, the step is a prefill
   step and *only* those newly admitted sequences are scheduled.  Decode
   sequences continue to occupy their KV-cache blocks but do not participate
   in this step.

2. **Decode only when nothing can be prefilled.**  All currently running
   sequences that still fit in memory are scheduled for decode together.

3. **Preemption via recompute.**  When KV-cache memory is exhausted during the
   decode phase, the scheduler evicts the *youngest* running sequence (LIFO
   from the running deque) back to the waiting queue, freeing its blocks.
   Next time the evicted sequence is re-admitted it will be fully re-prefilled
   (recompute strategy — no swap-to-CPU path).

4. **Continuous batching.**  The engine calls `schedule()` → model forward →
   `postprocess()` in a tight loop.  Finished sequences are removed in
   `postprocess`, and new requests can be admitted in the very next step.

Interaction with BlockManager
-----------------------------
The scheduler never touches physical KV-cache memory directly.  It delegates
all allocation / deallocation decisions to the ``BlockManager``, which tracks
free blocks, reference counts, and prefix-cache hashes.
"""

from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """Batches sequences for execution and manages their lifecycle.

    The scheduler is the single authority that moves sequences between the
    ``waiting`` and ``running`` queues and that triggers KV-cache block
    allocation / deallocation through the ``BlockManager``.

    Parameters (via Config)
    -----------------------
    max_num_seqs : int
        Hard cap on how many sequences can be scheduled in one step.
        Limits GPU parallelism so that per-sequence memory stays bounded.
    max_num_batched_tokens : int
        Maximum total tokens (across all sequences) that the model may
        process in a single forward pass.  This controls peak GPU memory
        for activations during prefill.
    eos : int
        End-of-sequence token id used to detect natural completion.
    num_kvcache_blocks / kvcache_block_size
        Forwarded to the BlockManager to set up the physical block pool.
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)

        # waiting: sequences that have been submitted but not yet prefilled.
        # running: sequences that have been prefilled and are actively decoding
        #          (their KV-cache blocks are allocated).
        # Both are FIFO deques so that oldest requests are served first (FCFS).
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """True when every submitted sequence has been fully generated."""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """Enqueue a new sequence for scheduling.

        The sequence starts in WAITING status and will be admitted (prefilled)
        in a future ``schedule()`` call once there is capacity.
        """
        self.waiting.append(seq)

    # ------------------------------------------------------------------
    # Core scheduling loop
    # ------------------------------------------------------------------

    def schedule(self) -> tuple[list[Sequence], bool]:
        """Select the batch for the next engine step.

        Returns
        -------
        scheduled_seqs : list[Sequence]
            The sequences that should participate in this step's forward pass.
        is_prefill : bool
            ``True`` if this is a prefill step (newly admitted sequences),
            ``False`` if it is a decode step (continuing sequences).

        Algorithm
        ---------
        **Phase 1 — try to prefill waiting sequences.**

        Walk the waiting queue front-to-back and greedily admit sequences as
        long as both constraints are satisfied:

        * ``num_seqs < max_num_seqs`` — batch-size cap.
        * ``num_batched_tokens + tokens_for_seq <= max_num_batched_tokens``
          — activation-memory cap.  Note that ``tokens_for_seq`` subtracts
          ``num_cached_tokens`` because prefix-cached blocks don't need
          recomputation and therefore don't consume activation memory.

        Each admitted sequence gets its KV-cache blocks allocated via the
        BlockManager (which may reuse prefix-cached blocks) and is moved from
        ``waiting`` to ``running``.

        If at least one sequence was admitted, we return immediately — this
        step is a pure prefill step.

        **Phase 2 — decode running sequences.**

        Only reached when *no* waiting sequence could be admitted (either the
        waiting queue is empty or the head-of-line sequence doesn't fit).

        We iterate through running sequences and, for each one, ensure the
        BlockManager can append a new token slot.  Decoding a token may push
        a sequence across a block boundary, requiring one fresh block.  If
        memory is insufficient:

        * **Preempt** the youngest running sequence (``self.running.pop()`` —
          LIFO order) to free its blocks.  Repeat until the current sequence
          fits.
        * If the current sequence is the *only* one left and still can't fit,
          preempt itself (it will be re-prefilled later).

        After the loop, the scheduled sequences are pushed back onto the
        *front* of the running deque so they remain in their original FCFS
        order for the next step.
        """
        # ----------------------------------------------------------
        # Phase 1: Prefill
        # ----------------------------------------------------------
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]

            # Check both the token budget and physical block availability.
            # If the head-of-line request doesn't fit, stop — we don't skip
            # it to preserve FCFS ordering (avoiding starvation of large
            # prompts that would be perpetually skipped).
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break

            num_seqs += 1
            self.block_manager.allocate(seq)

            # Only count tokens that actually need computation.  Prefix-cached
            # tokens already have their KV entries so the model runner skips
            # them, and they don't contribute to activation memory pressure.
            num_batched_tokens += len(seq) - seq.num_cached_tokens

            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True  # is_prefill = True

        # ----------------------------------------------------------
        # Phase 2: Decode
        # ----------------------------------------------------------
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()

            # Ensure there is room for one more token.  can_append() checks
            # whether the next token crosses a block boundary; if so, one
            # free block is needed.
            while not self.block_manager.can_append(seq):
                if self.running:
                    # Evict the youngest sequence (LIFO) to reclaim blocks.
                    self.preempt(self.running.pop())
                else:
                    # No other sequence to evict — preempt ourselves.
                    # This sequence goes back to waiting and will be
                    # re-prefilled in a future step.
                    self.preempt(seq)
                    break
            else:
                # The while-else fires when can_append succeeded (loop
                # condition became False without hitting break).
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)

        # At least one decode sequence must be schedulable (otherwise the
        # system has insufficient memory to make progress on any request).
        assert scheduled_seqs

        # Restore scheduled sequences to the front of the running deque in
        # their original order so that the next step's decode phase sees
        # them first (FCFS fairness).
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False  # is_prefill = False

    # ------------------------------------------------------------------
    # Preemption
    # ------------------------------------------------------------------

    def preempt(self, seq: Sequence):
        """Evict a running sequence to free KV-cache blocks.

        The sequence is moved back to the *front* of the waiting queue (so it
        will be the first to be re-admitted) and all its blocks are released.
        On re-admission the sequence will be fully re-prefilled — this is the
        "recompute" preemption strategy (as opposed to swapping KV-cache to
        CPU memory, which nano-vllm does not implement).
        """
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    # ------------------------------------------------------------------
    # Post-processing (after model forward)
    # ------------------------------------------------------------------

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """Append sampled tokens and retire finished sequences.

        Called by the engine after the model forward + sampling step.  For each
        (sequence, token) pair:

        1. Append the new token to the sequence's token list.
        2. Check the two stopping conditions:
           a. The token is EOS and ``ignore_eos`` is not set.
           b. The sequence has generated ``max_tokens`` completion tokens.
        3. If finished, mark the sequence FINISHED, release its KV-cache
           blocks, and remove it from the running deque.

        The caller uses the updated sequence statuses to decide when to return
        results to the user.
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
