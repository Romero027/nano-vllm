"""ModelRunner: the GPU-side execution engine for inference.

The ModelRunner owns the model, KV-cache, and CUDA graphs.  It is the only
component that touches GPU memory.  One ModelRunner instance is created per
GPU (per tensor-parallel rank).

Execution modes
---------------
* **Prefill** — processes the full prompt (minus any prefix-cached tokens)
  in a single forward pass using variable-length flash-attention
  (cu_seqlens / varlen API).
* **Decode** — generates one token per sequence using paged KV-cache
  attention (block_tables + slot_mapping).

Tensor parallelism
------------------
When ``tensor_parallel_size > 1``, rank 0 is the *driver* and all other
ranks are *workers*.  Communication works as follows:

  1. Rank 0 serialises (method_name, args) into a 1 MiB shared-memory
     segment using pickle, then signals worker ranks via multiprocessing
     Events.
  2. Workers block on their Event, deserialise the payload, and execute
     the same method.  This keeps all ranks in lockstep for the NCCL
     all-reduce calls inside the model.
  3. Only rank 0 performs sampling (since logits are identical after the
     all-reduce of the final hidden states).

CUDA graph acceleration
-----------------------
For decode steps with batch size <= 512, the runner pre-captures a set of
CUDA graphs at discrete batch sizes (1, 2, 4, 8, 16, 32, …) during init.
At runtime it selects the smallest captured graph that fits the current
batch and replays it, eliminating CPU-side kernel launch overhead.  Prefill
steps always run eagerly because their shapes vary widely.
"""

import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """Initialise one ModelRunner on a single GPU.

        Parameters
        ----------
        config : Config
            Global engine configuration (model path, limits, TP size, etc.).
        rank : int
            Tensor-parallel rank (0 = driver, 1+ = workers).
        event : Event | list[Event]
            Synchronisation primitives for the shared-memory IPC channel.
            * Rank 0 receives a **list** of Events (one per worker) and sets
              them all after writing to shared memory.
            * Workers receive a **single** Event and block on it.
        """
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # --- 1. Initialise NCCL process group (one rank per GPU) -----------
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)

        # Temporarily switch the default dtype/device to CUDA so that all
        # model parameters and buffers are created on GPU with the correct
        # precision.  Restored to CPU/default after init.
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # --- 2. Build model, load weights, prepare sampler -----------------
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()

        # --- 3. Warmup → measure peak memory → allocate KV-cache ----------
        # warmup_model runs the largest possible prefill so that PyTorch's
        # caching allocator expands to its peak.  After that, allocate_kv_cache
        # uses the *remaining* GPU memory for paged KV-cache blocks.
        self.warmup_model()
        self.allocate_kv_cache()

        # --- 4. Capture CUDA graphs for decode (optional) ------------------
        if not self.enforce_eager:
            self.capture_cudagraph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # --- 5. Set up shared-memory IPC for tensor parallelism ------------
        # Rank 0 creates the segment; workers wait at a barrier until the
        # segment exists, then attach to it and enter their event-driven loop.
        # Workers never return from __init__ — they spin in self.loop() until
        # they receive an "exit" command.
        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        """Tear down GPU resources and IPC channels.

        Called via ``self.call("exit")`` so that all ranks execute it in
        lockstep.  Cleanup order matters:
          1. Close/unlink shared memory (rank 0 unlinks after all ranks close).
          2. Release captured CUDA graphs so their memory can be freed.
          3. Synchronise the GPU, then destroy the NCCL process group.
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()  # ensure all ranks have closed before unlinking
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    # ------------------------------------------------------------------
    # Tensor-parallel IPC (shared memory + multiprocessing Events)
    #
    # The protocol is simple: rank 0 writes a pickle blob to a 1 MiB
    # shared-memory segment, then signals each worker's Event.  Workers
    # wake up, deserialise, and call the same method on their own runner.
    #
    # Wire format:  [4-byte little-endian length][pickle payload]
    #
    # This avoids torch.distributed broadcast for the control plane
    # (which sequences to run, their block tables, etc.), reserving
    # NCCL exclusively for the data plane (all-reduce of hidden states).
    # ------------------------------------------------------------------

    def loop(self):
        """Worker-only event loop.  Blocks forever until "exit" is received."""
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """Block on the Event, then deserialise (method_name, *args) from shm."""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """Serialise (method_name, *args) into shm and wake all workers."""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        """Invoke a method on this runner, broadcasting to workers if rank 0.

        This is the single entry point for all cross-rank method dispatch.
        Rank 0 first writes the call to shared memory (waking workers), then
        *also* executes the method locally.  Workers call this directly after
        reading from shm (without re-broadcasting).
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """Run the largest possible prefill to measure peak activation memory.

        Why this matters: after warmup we call ``allocate_kv_cache()``, which
        fills *all remaining* GPU memory with KV-cache blocks.  If we didn't
        trigger peak memory first, the KV-cache would over-allocate and the
        first real prefill would OOM.

        The dummy batch is sized to match the scheduler's worst-case prefill:
        ``max_num_batched_tokens`` total tokens split across sequences of
        ``max_model_len`` each, capped at ``max_num_seqs``.

        After the forward pass we clear the cache and reset peak stats so
        ``allocate_kv_cache`` sees a clean baseline.
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """Allocate the paged KV-cache tensor and wire it into the model.

        Memory budget calculation
        -------------------------
        We want to use all GPU memory that is *not* needed for model weights
        or peak activations.  The formula:

            available = total * gpu_memory_utilization
                        - non_torch_overhead   (``used - peak``)
                        - peak_torch_alloc     (``peak``)
                        + currently_live_alloc  (``current``, because these
                          are already counted in ``used``)

        Simplifies to:  ``total * util - used - peak + current``

        KV-cache layout
        ---------------
        The cache is a single contiguous tensor with shape::

            [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]

        Dimension 0 separates K and V.  Each attention layer gets a view
        (``module.k_cache`` / ``module.v_cache``) of shape
        ``[num_blocks, block_size, num_kv_heads, head_dim]``, which it
        indexes via block_tables and slot_mapping during attention.

        block_bytes accounts for K+V (factor of 2) × all layers × tokens
        per block × heads × head_dim × dtype element size.
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # bytes per block = 2 (K+V) × layers × tokens_per_block × heads × head_dim × elem_size
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

        # Wire per-layer views into attention modules.  This avoids passing
        # the cache tensor through the forward call — each attention layer
        # directly reads/writes its own slice.
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    # ------------------------------------------------------------------
    # Input preparation
    #
    # These methods translate the scheduler's high-level "run these
    # sequences" into the low-level tensors consumed by the model:
    #   input_ids, positions  – what to compute
    #   slot_mapping          – where to write new KV entries
    #   block_tables          – where to read existing KV entries
    #   cu_seqlens_*          – flash-attention varlen boundaries
    #   context_lens          – per-sequence context length for decode
    # ------------------------------------------------------------------

    def prepare_block_tables(self, seqs: list[Sequence]):
        """Pad per-sequence block tables to equal length and move to GPU.

        Each sequence may have a different number of allocated blocks.
        We right-pad shorter tables with -1 (an invalid block ID that the
        attention kernel ignores) to form a [batch, max_blocks] tensor.
        Uses pinned memory + non_blocking transfer for overlap with compute.
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """Build flash-attention varlen inputs for a prefill step.

        Flash-attention's varlen API packs multiple variable-length sequences
        into a single 1-D tensor and uses cumulative sequence-length arrays
        (``cu_seqlens_q``, ``cu_seqlens_k``) to locate each sequence.

        Prefix caching
        ~~~~~~~~~~~~~~
        When a sequence has prefix-cached blocks (``num_cached_tokens > 0``),
        those tokens already have valid KV entries in the cache.  We skip them
        in input_ids and positions (they don't need recomputation), so
        ``seqlen_q < seqlen_k``.  The attention kernel uses block_tables to
        read the cached K/V for the prefix portion and the freshly computed
        K/V for the new portion.

        If *no* sequence has cached tokens, ``cu_seqlens_q == cu_seqlens_k``
        everywhere, and block_tables is not needed (pure prefill, Q attends
        only to self).

        Slot mapping
        ~~~~~~~~~~~~
        ``slot_mapping`` tells the attention kernel *where* to write each
        new K/V entry in the flat KV-cache tensor.  For each non-cached
        block, we emit the range ``[block_id * block_size, ...)`` for all
        tokens in that block (the last block may be partial).
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            # Only feed tokens *after* the prefix-cached portion.
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens  # tokens to compute
            seqlen_k = seqlen                           # full context length (including cache)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup (no real blocks allocated)
                continue
            # Emit slot indices only for non-cached blocks.
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))

        # block_tables are only needed when Q ≠ K lengths (prefix cache hit),
        # so the kernel can read cached KV from previous blocks.
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """Build paged-attention inputs for a decode step.

        During decode each sequence contributes exactly **one** new token
        (the most recently sampled one).  The attention kernel reads the
        full KV history via ``block_tables`` and writes the new K/V entry
        to the single slot identified by ``slot_mapping``.

        The slot for the new token is the last occupied position in the
        sequence's final block:
        ``block_table[-1] * block_size + last_block_num_tokens - 1``
        (the scheduler's ``may_append`` has already ensured the block exists).
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """Gather per-sequence sampling temperatures into a GPU tensor.

        Only called on rank 0 (the only rank that performs sampling).
        """
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # ------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """Execute the model forward pass (eager or via CUDA graph).

        Falls back to eager execution when:
        * The step is a prefill (variable-length, not graphable).
        * ``enforce_eager`` is set (debugging / profiling).
        * Batch size exceeds 512 (larger than any captured graph).

        CUDA graph replay path
        ~~~~~~~~~~~~~~~~~~~~~~
        CUDA graphs require fixed tensor addresses — the kernel pointers are
        baked in at capture time.  So we copy the *current* step's data into
        the pre-allocated ``graph_vars`` tensors (which were the ones used
        during capture).  Unused slots are zeroed / set to -1 so the kernels
        produce valid (but ignored) output for padded positions.

        We select the smallest captured batch size >= actual bs to minimise
        wasted work.
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # Pick the smallest graph whose batch size covers the current batch.
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # Copy live data into the fixed graph-captured tensors.
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            # -1 slots are ignored by the write kernel; zeroed context_lens
            # produce zero-length attention for padded sequences.
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """End-to-end single step: prepare → forward → sample.

        Returns the list of newly sampled token IDs (one per sequence).
        Only rank 0 returns actual tokens; workers return None (they only
        need to execute the forward pass for the NCCL all-reduce to complete).
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # ------------------------------------------------------------------
    # CUDA graph capture
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def capture_cudagraph(self):
        """Pre-capture CUDA graphs for decode at discrete batch sizes.

        CUDA graphs record a sequence of GPU operations (kernels, memory
        copies) once and replay them with near-zero CPU overhead.  This is
        especially valuable for decode steps where the model forward pass is
        short and CPU launch latency dominates.

        Capture strategy
        ~~~~~~~~~~~~~~~~
        We capture at batch sizes [1, 2, 4, 8, 16, 32, …, max_bs].  At
        runtime, ``run_model`` picks the smallest graph that fits.  Capturing
        in **reverse** order (largest first) lets all graphs share a single
        memory pool (``graph_pool``), since the largest graph's allocations
        are a superset of smaller ones.

        All captured graphs operate on the **same** pre-allocated tensors
        (``graph_vars``).  At replay time we copy live data into these
        tensors — CUDA graphs require fixed device pointers.

        Each batch size gets a warmup forward pass (to trigger lazy CUDA
        initialisations) followed by the actual capture.
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size

        # Pre-allocate fixed tensors sized for the maximum batch.  These
        # same tensors are used for every graph capture and every replay.
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        # Discrete batch sizes: fine-grained at small sizes (1,2,4,8) to
        # avoid excessive waste, then multiples of 16 for larger batches.
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                # The first (largest) graph creates the shared memory pool;
                # all subsequent graphs reuse it.
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # Store references to the capture tensors so run_model can write
        # live data into them before replaying.
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
