"""
LLMEngine: the central orchestrator of the nano-vllm inference pipeline.

This module ties together the three core subsystems:
  1. ModelRunner  – loads the model weights, manages KV-cache, runs forward
                    passes (optionally with CUDA graphs), and samples tokens.
  2. Scheduler   – decides *which* sequences to run each step (prefill vs.
                    decode), manages KV-cache block allocation, and handles
                    preemption when memory is tight.
  3. Tokenizer   – encodes user prompts into token IDs and decodes generated
                    token IDs back into text.

Architecture overview (tensor-parallel = 2 example):

    ┌─────────────── main process (rank 0) ───────────────┐
    │  LLMEngine                                          │
    │   ├─ Scheduler (request queues + block manager)     │
    │   ├─ Tokenizer                                      │
    │   └─ ModelRunner[rank=0]  ──shared-memory──►        │
    │         NCCL all-reduce  ◄────────────────►         │
    └─────────────────────────────────────────────────────┘
                                                ▲
    ┌──── spawned worker process (rank 1) ──────┤
    │  ModelRunner[rank=1]                      │
    │    (runs in an infinite loop, receiving   │
    │     commands via shared memory + events)  │
    └───────────────────────────────────────────┘

Only rank 0 runs the scheduler and tokenizer; worker ranks are pure
model-execution processes that mirror rank 0's forward passes via NCCL.
"""

import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:
    """
    High-level inference engine that exposes two usage patterns:

    1. **Batch generation** (`generate`): supply a list of prompts and receive
       a list of completed outputs.  Internally this calls `add_request` for
       each prompt and then loops over `step` until every sequence finishes.

    2. **Step-by-step control** (`add_request` / `step` / `is_finished`):
       for callers that need to interleave their own logic between inference
       steps (e.g. streaming, custom stopping criteria, online serving).
    """

    def __init__(self, model, **kwargs):
        # ── Build Config ────────────────────────────────────────────────
        # Only forward kwargs that correspond to actual Config fields so
        # callers can pass extra keyword arguments without triggering errors.
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)

        # ── Tensor-parallel worker processes ────────────────────────────
        # We use the "spawn" start method (not fork) because CUDA contexts
        # are not fork-safe — forking after CUDA initialisation causes
        # undefined behaviour.
        #
        # Ranks 1..N-1 are spawned first.  Each worker's ModelRunner.__init__
        # ends by entering an infinite command loop (see ModelRunner.loop),
        # so the process never returns — it just waits for instructions from
        # rank 0 via shared memory.
        #
        # Each worker gets a single mp.Event used by rank 0 to signal that a
        # new command has been written to shared memory.
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # Rank 0's ModelRunner runs in the main process.  It receives the
        # *list* of all worker events so it can broadcast commands to every
        # worker before executing the command itself (see ModelRunner.call).
        self.model_runner = ModelRunner(config, 0, self.events)

        # ── Tokenizer & scheduler ───────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        # The EOS token ID is written back into the config so the scheduler
        # can detect end-of-sequence without needing its own tokenizer.
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        # Ensure GPU resources and worker processes are cleaned up even if
        # the user forgets to call exit() explicitly.
        atexit.register(self.exit)

    def exit(self):
        """Shut down all workers and release resources.

        The "exit" command propagates through ModelRunner.call → write_shm to
        every worker, causing their command loops to break.  We then join each
        subprocess to wait for clean termination.
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """Enqueue a single generation request.

        Accepts either a raw text string (which will be tokenized) or
        pre-tokenized token IDs.  The resulting Sequence is placed into the
        scheduler's waiting queue and will be picked up by a future `step`.
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        # A Sequence tracks one inference request through its lifecycle
        # (WAITING → RUNNING → FINISHED), holding the growing token list,
        # KV-cache block table, and per-request sampling params.
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        """Execute one scheduler cycle: schedule → forward pass → postprocess.

        Returns:
            outputs:    list of (seq_id, completion_token_ids) for sequences
                        that *finished* on this step (hit EOS or max_tokens).
            num_tokens: an overloaded throughput indicator:
                          • positive  → prefill step; value = total prompt
                            tokens processed this step.
                          • negative  → decode step; |value| = number of
                            sequences that each produced one new token.
                        This sign convention lets `generate` display both
                        prefill tok/s and decode tok/s with a single return.
        """
        # The scheduler decides whether to run a prefill batch (new prompts)
        # or a decode batch (continue running sequences).  Prefill is always
        # prioritised: if any sequences are waiting, they are scheduled first.
        seqs, is_prefill = self.scheduler.schedule()

        # Forward pass + sampling on rank 0 (workers mirror via shared memory).
        token_ids = self.model_runner.call("run", seqs, is_prefill)

        # Append the newly sampled token to each sequence and mark any that
        # have reached a stopping condition (EOS / max_tokens) as finished.
        self.scheduler.postprocess(seqs, token_ids)

        # Collect finished sequences to return to the caller.
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

        # Encode throughput info: positive = prefill tokens, negative = decode
        # batch size (see docstring above for rationale).
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        """True when both the waiting and running queues are empty."""
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """Run offline batch generation over a list of prompts.

        This is the simplest way to use the engine: hand it a batch of
        prompts and get back a list of result dicts with "text" and
        "token_ids" keys, one per prompt, in the original prompt order.

        Args:
            prompts:         list of text strings or pre-tokenized ID lists.
            sampling_params: a single SamplingParams applied to all prompts,
                             or a per-prompt list.
            use_tqdm:        show a progress bar with live throughput stats.

        Returns:
            A list of dicts [{"text": str, "token_ids": list[int]}, ...],
            ordered to match the input `prompts` list.
        """
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)

        # Broadcast a single SamplingParams to every prompt if needed.
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        # Submit all prompts into the scheduler's waiting queue.
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        # Outputs are collected in a dict keyed by seq_id because sequences
        # can finish in any order (shorter prompts complete before longer ones).
        outputs = {}
        prefill_throughput = decode_throughput = 0.

        # ── Main inference loop ─────────────────────────────────────────
        # Each iteration runs one scheduler step.  The scheduler alternates
        # between prefill steps (processing new prompt tokens) and decode
        # steps (generating one new token per running sequence).
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()

            # Update the live throughput display.  The sign of num_tokens
            # tells us which phase the step belonged to (see step() docs).
            if use_tqdm:
                if num_tokens > 0:
                    # Prefill: num_tokens = total prompt tokens processed.
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # Decode: -num_tokens = batch size (each seq produced 1 token).
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })

            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)

        # Re-order results to match the original prompt ordering.  Sequence
        # IDs are assigned by a monotonic counter, so sorting by seq_id
        # restores the original input order.
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
