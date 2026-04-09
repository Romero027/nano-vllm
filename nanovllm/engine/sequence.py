"""Sequence: the core unit of work flowing through the engine.

A Sequence represents a single inference request.  It is created from a
tokenized prompt and carries all per-request state needed by the scheduler,
block manager, and model runner:

  * The full (and growing) list of token IDs (prompt + generated tokens).
  * Paged KV-cache bookkeeping (block table, cached-block count).
  * Sampling configuration extracted from SamplingParams.

Lifecycle:  WAITING  ──schedule──▶  RUNNING  ──eos/max_tokens──▶  FINISHED
            (queued)                (actively                     (result
                                    decoding)                     returned)
"""

from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """Tri-state lifecycle of a sequence.

    WAITING  – sitting in the scheduler's waiting queue; not yet allocated
               KV-cache blocks.
    RUNNING  – actively being prefilled or decoded; KV-cache blocks are
               allocated and held.
    FINISHED – generation is complete (hit EOS or max_tokens); blocks are
               eligible for release.
    """
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """Mutable state for one inference request.

    The token list starts as the prompt and grows by one token per decode
    step via `append_token`.  Block-related properties partition the token
    list into fixed-size chunks that map 1-to-1 to KV-cache blocks managed
    by the BlockManager.

    Class attributes
    ----------------
    block_size : int
        Default block size (tokens per KV-cache block).  Overridden at
        engine init to match ``config.kvcache_block_size``.
    counter : itertools.count
        Global auto-incrementing ID generator shared across all sequences.
    """
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """Create a new sequence from a tokenized prompt.

        Parameters
        ----------
        token_ids : list[int]
            Pre-tokenized prompt token IDs.
        sampling_params : SamplingParams
            Per-request sampling knobs (temperature, max_tokens, …).

        Attributes set here
        --------------------
        seq_id            – Unique integer ID (monotonically increasing).
        status            – Starts as WAITING; the scheduler advances it.
        token_ids         – *Copy* of the prompt; grows during generation.
        last_token        – Most-recently appended token (used by the decode
                            input builder so it doesn't have to index into
                            the full list).
        num_tokens        – Current length (prompt + completion).
        num_prompt_tokens – Fixed at creation; never changes.
        num_cached_tokens – Number of leading tokens whose KV entries were
                            reused from a previously cached block (prefix
                            caching).  Set by BlockManager.allocate().
        block_table       – List of physical KV-cache block IDs assigned to
                            this sequence, populated by the BlockManager.
        temperature       – Sampling temperature (flattened from params for
        max_tokens          fast access in the hot loop).
        ignore_eos        –
        """
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """Total number of tokens (prompt + generated so far)."""
        return self.num_tokens

    def __getitem__(self, key):
        """Index or slice directly into the token list."""
        return self.token_ids[key]

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    # ------------------------------------------------------------------
    # Token-count helpers
    # ------------------------------------------------------------------

    @property
    def num_completion_tokens(self):
        """Tokens generated so far (excludes the original prompt)."""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """The original prompt portion of the token list."""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """Only the tokens produced by the model (after the prompt)."""
        return self.token_ids[self.num_prompt_tokens:]

    # ------------------------------------------------------------------
    # Block-level helpers  (paged KV-cache)
    #
    # The token list is logically partitioned into contiguous blocks of
    # `block_size` tokens.  Each block maps to exactly one physical
    # KV-cache block.  The last block may be partially filled.
    #
    #   block 0          block 1          block 2 (partial)
    #  ┌───────────┐   ┌───────────┐   ┌──────┐
    #  │ tok … tok  │   │ tok … tok  │   │ tok … │
    #  └───────────┘   └───────────┘   └──────┘
    #   block_size       block_size      last_block_num_tokens
    # ------------------------------------------------------------------

    @property
    def num_cached_blocks(self):
        """How many leading blocks were fully reused from prefix cache."""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """Total blocks needed to cover all current tokens (ceil division)."""
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        """Number of valid tokens in the final (possibly partial) block.

        Ranges from 1 to block_size.  The model runner uses this to know
        how many slot mappings to emit for the last block.
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        """Return the token IDs belonging to the *i*-th block.

        Used by the BlockManager to compute content hashes for prefix
        caching — two blocks with identical token IDs can share the same
        physical KV-cache block.
        """
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append_token(self, token_id: int):
        """Extend the sequence by one decoded token."""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    # ------------------------------------------------------------------
    # Pickle optimisation (tensor-parallel IPC)
    #
    # When tensor_parallel_size > 1, rank-0 broadcasts the scheduled
    # batch to workers via shared memory using pickle (see ModelRunner.
    # write_shm / read_shm).  The full token_ids list is only needed
    # during prefill (to build input_ids & slot_mapping); during decode
    # the workers only need `last_token`.  So we omit token_ids from
    # the serialised form once generation has started, significantly
    # shrinking the pickled payload on every decode step.
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Minimise pickle size: send token_ids only during prefill,
        otherwise just last_token."""
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        """Restore from the compact representation produced by __getstate__."""
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
