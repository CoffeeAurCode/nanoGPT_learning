# Module 07 — Distributed Training

**Source files:** `train.py` lines 82–101, 211–212, 292–299, 335–336  
**Estimated time:** 45 minutes  
**Next:** [Module 08 — Sampling & Inference](module_08_sampling_and_inference.md)

---

## What You Are Building

The four places in `train.py` where distributed training (DDP) changes the code. You will understand how `torchrun` launches multiple processes, what `DistributedDataParallel` does to the backward pass, why gradient sync is suppressed during accumulation micro-steps, and how the `master_process` pattern ensures only one process logs and saves.

> **Skip this module if you only have one GPU.** Everything in nanoGPT works on a single GPU. Come back to this module when you want to scale up.

---

## Concept Deep-Dives

### 1. `torchrun` and Process Groups

**Definition:** `torchrun` launches N copies of the training script as separate OS processes. Each process gets a different `RANK` (global index), `LOCAL_RANK` (index on this node), and `WORLD_SIZE` (total processes). The processes then form a "process group" that allows collective communication.

```bash
# Single GPU (what you've been doing):
python train.py config/train_gpt2.py

# 4 GPUs on one machine:
torchrun --standalone --nproc_per_node=4 train.py config/train_gpt2.py

# 8 GPUs across 2 machines:
# On node 0 (master, IP=123.456.123.456):
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# On node 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

**How nanoGPT detects DDP mode:**

```python
ddp = int(os.environ.get('RANK', -1)) != -1
```

`torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` as environment variables before launching each process. If `RANK` is not set (single-GPU run), `os.environ.get('RANK', -1)` returns `-1` and `ddp` is `False`.

**`init_process_group`:**

```python
from torch.distributed import init_process_group, destroy_process_group

if ddp:
    init_process_group(backend=backend)   # backend='nccl' for GPU
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
```

`init_process_group` establishes the communication channel between processes. `nccl` (NVIDIA Collective Communications Library) is the fastest backend for GPU-to-GPU communication via NVLink or InfiniBand.

**What `backend='gloo'` is for:** CPU training or machines without NVLink. Slower than nccl but works everywhere.

**What breaks if you call `init_process_group` twice:** The process group is a global singleton. Calling it again raises a RuntimeError. `destroy_process_group()` at the end of training cleans it up.

---

### 2. DistributedDataParallel (DDP)

**Definition:** DDP wraps the model so that after each backward pass, gradients are averaged across all processes via an all-reduce operation. Every process sees the same gradient and takes the same optimizer step — the models stay in sync.

```python
from torch.nn.parallel import DistributedDataParallel as DDP

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

**How DDP works step by step:**

```
Process 0:                Process 1:
  Forward pass              Forward pass
  loss.backward()           loss.backward()
       ↓                         ↓
  Local gradients           Local gradients
       ↓       ← all-reduce →    ↓
  Averaged grads            Averaged grads (same as P0)
  optimizer.step()          optimizer.step()
  (Same weight update on both)
```

Each process computes gradients on its own micro-batch. All-reduce averages these gradients across all processes. Each process applies the same averaged gradient → same weight update → models stay identical.

**The effective batch size:** If you have 4 GPUs each with `batch_size=12`, the effective batch size is `12 × 4 = 48`. This is automatically handled because the gradient is averaged: each process's gradient represents 12 samples, and averaging across 4 processes gives the same result as computing one gradient over all 48 samples.

**`raw_model = model.module if ddp else model`:** After wrapping in DDP, accessing the original model requires `model.module`. This is used for `state_dict()` (to save without the `module.` prefix) and for `estimate_mfu()` (which is a method of the original GPT, not of DDP).

---

### 3. Gradient Sync Suppression During Accumulation

**Definition:** By default, DDP synchronizes (all-reduces) gradients after every `.backward()` call. During gradient accumulation, you only want one all-reduce per optimizer step, not one per micro-step. Suppressing intermediate syncs saves significant communication overhead.

```python
for micro_step in range(gradient_accumulation_steps):
    if ddp:
        # Suppress gradient sync for all micro-steps except the last
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train')
    scaler.scale(loss).backward()
```

**`model.require_backward_grad_sync`:** This is an internal attribute of DDP that controls whether `backward()` triggers an all-reduce. Setting it to `False` for all but the last micro-step means: accumulate locally, then synchronize once.

**The official alternative `model.no_sync()`:**

```python
# Official way (more verbose):
for micro_step in range(gradient_accumulation_steps - 1):
    with model.no_sync():       # suppress gradient sync
        logits, loss = model(X, Y)
        scaler.scale(loss / gradient_accumulation_steps).backward()
# Last step: sync
logits, loss = model(X, Y)
scaler.scale(loss / gradient_accumulation_steps).backward()
```

nanoGPT uses the direct attribute toggle because `no_sync()` requires duplicating the forward/backward code.

**What breaks without suppression:** For `gradient_accumulation_steps=40`, you'd perform 40 all-reduces per optimizer step instead of 1. On a 4-GPU cluster with NVLink, each all-reduce takes ~10 ms. That's 400 ms of communication overhead per step vs. 10 ms with suppression — a 40× slowdown in communication.

---

### 4. The `master_process` Pattern

**Definition:** In a DDP run, all processes are identical. Most operations (model forward, backward, optimizer step) must run on all processes. But some operations (logging, saving checkpoints, printing) should only run once.

```python
master_process = (ddp_rank == 0)   # True only on rank 0

# This runs on all processes:
for micro_step in range(gradient_accumulation_steps):
    ...backward...

# This runs only on rank 0:
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, ...)
    print(f"step {iter_num}: train loss...")
    if wandb_log:
        wandb.log(...)
```

**Why process 0?** By convention. Any rank could be "master" — the important thing is that exactly one process saves and logs.

**What breaks if you don't use `master_process`:** 4 processes all try to write to the same `ckpt.pt` file simultaneously — race condition, corrupted checkpoint. 4 lines of the same log message. 4 wandb runs logging the same data.

---

### 5. Seed Offset Per Process

```python
seed_offset = ddp_rank   # 0, 1, 2, 3 for 4 GPUs
torch.manual_seed(1337 + seed_offset)
```

**Why?** Each process samples different batches from `train.bin`. If all processes had the same random seed, `torch.randint` would sample the same `ix` positions on every process — they'd all train on the same data. The seed offset ensures each process samples a unique set of positions.

**Why not use a DataLoader with a distributed sampler?** The memmap-based data loader doesn't use PyTorch's DataLoader. With a flat binary file, independent random sampling on each process naturally provides the equivalent of distributed sampling without any coordination.

---

### 6. Gradient Accumulation Steps Scaling in DDP

```python
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

assert gradient_accumulation_steps % ddp_world_size == 0
gradient_accumulation_steps //= ddp_world_size
```

**Why divide `gradient_accumulation_steps` by `ddp_world_size`?**  
With 4 GPUs and `gradient_accumulation_steps=40`, each GPU runs 40 micro-steps before syncing. But each of the 4 GPUs processes 40 micro-batches in parallel — that's 160 micro-batches total, equivalent to a 160× accumulation. To keep the same effective batch size as the single-GPU run, you divide: each GPU runs `40 / 4 = 10` micro-steps.

**The `assert`:** Ensures `gradient_accumulation_steps` is exactly divisible by `ddp_world_size`. If not, different processes would run a different number of micro-steps, causing a gradient mismatch.

---

## Reading the Source File

### The four DDP-specific code regions in `train.py`

**Region 1 — Initialization (lines 82–101):**
```python
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
```

**Region 2 — DDP wrap (line 212):**
```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```
This comes after `torch.compile` (if enabled). Wrapping a compiled model in DDP still works.

**Region 3 — Gradient sync suppression (lines 292–295):**
```python
if ddp:
    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
```

**Region 4 — Cleanup (lines 335–336):**
```python
if ddp:
    destroy_process_group()
```
Cleanly shuts down the NCCL communication group. Without this, processes may hang waiting for each other after training completes.

---

## Why This Design

**Why DDP over DataParallel (DP)?**  
PyTorch's older `DataParallel` runs all work on one GPU (the "master"), with the others just computing gradients. This creates a memory imbalance (master GPU needs to hold the model + all gradients) and a communication bottleneck. DDP gives each GPU an independent process with its own memory space — scales linearly with GPU count.

**Why DDP over FSDP (Fully Sharded Data Parallel)?**  
FSDP shards the model weights across GPUs, enabling training of models larger than a single GPU's memory. For GPT-2 at 124M parameters, the model easily fits on one GPU. DDP is simpler and faster when the model fits. Use FSDP for models >1B parameters.

**Why not DeepSpeed?**  
DeepSpeed is a more powerful distributed training library with additional optimizations (ZeRO stages, activation checkpointing, CPU offloading). For nanoGPT's scale, it adds complexity without much benefit. For training 7B+ parameter models, DeepSpeed or FSDP are necessary.

---

## Running the Tests

There are no unit tests for the DDP code itself — distributed training requires actually launching multiple processes, which does not work in pytest's single-process environment. The DDP code is tested end-to-end by running:

```bash
# Test DDP on a 2-GPU machine:
torchrun --standalone --nproc_per_node=2 train.py config/train_shakespeare_char.py --max_iters=100
```

Verify that the checkpoint saves correctly and that training runs without hanging.

---

## Checkpoint ✓

Verify you understand the DDP detection pattern (works on any machine):

```python
import os
import torch

# Simulate what happens in train.py without torchrun
os.environ.pop('RANK', None)   # ensure not set
ddp = int(os.environ.get('RANK', -1)) != -1
print(f"ddp={ddp}, master_process={not ddp or True}")

# Simulate what happens with torchrun (RANK=0)
os.environ['RANK'] = '0'
ddp = int(os.environ.get('RANK', -1)) != -1
master_process = int(os.environ['RANK']) == 0
print(f"ddp={ddp}, master_process={master_process}")
os.environ.pop('RANK')
```

**Expected:**
```
ddp=False, master_process=True
ddp=True, master_process=True
```

---

## Exercises

**1 (Easy) — Trace the DDP code path:**  
Read `train.py` and mark every line that is DDP-specific (only runs when `ddp=True`). Count them. Verify that the single-GPU path is a strict subset of the DDP path — the DDP code adds to the single-GPU code but does not replace it.

**Success condition:** A list of line numbers that are DDP-only. The single-GPU path produces the same checkpoint format as the DDP path.

**2 (Medium) — Simulate gradient accumulation with explicit averaging:**  
In a single-GPU setting, implement gradient accumulation manually without using the loop in `train.py` — use `torch.autograd.grad()` to compute per-batch gradients and average them manually. Verify the result is numerically identical to the loop-based approach.

**Success condition:** The manually averaged gradient matches the loop-based accumulated gradient to within 1e-6.

**3 (Hard) — Implement data-parallel training with `multiprocessing.spawn`:**  
Rewrite the DDP initialization to use `torch.multiprocessing.spawn` instead of `torchrun`. The key difference: `spawn` launches processes from within Python instead of via a shell command. This requires passing the rank explicitly and calling `init_process_group` with `rank` and `world_size` arguments.

**Success condition:** Training runs on 2 GPUs (or 2 CPU processes for testing), both processes see different batches, and the final checkpoint is identical to a single-GPU run for the same random seed.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| DDP tutorial | [PyTorch DDP tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) | Official tutorial; covers initialization, wrapping, and teardown |
| NCCL operations | [NCCL API (NVIDIA)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html) | Explains all-reduce, all-gather, and broadcast — the operations DDP uses internally |
| Gradient synchronization | [PyTorch no_sync docs](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync) | Official API for suppressing gradient sync during accumulation |

---

## What's Next

[Module 08 — Sampling & Inference](module_08_sampling_and_inference.md): load a trained checkpoint and generate text with temperature and top-k sampling.
