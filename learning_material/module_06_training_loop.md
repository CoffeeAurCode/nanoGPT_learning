# Module 06 — The Training Loop

**Source files:** `train.py` lines 116–336  
**Estimated time:** 60 minutes  
**Next:** [Module 07 — Distributed Training](module_07_distributed_training.md)

---

## What You Are Building

The main training loop in `train.py` — the code that actually updates the model weights. By the end of this module you will understand every line of the loop: how batches are sampled from the binary file, how gradient accumulation simulates a larger batch size, how the cosine learning rate schedule works, and how to checkpoint correctly.

---

## Concept Deep-Dives

### 1. Batch Sampling from a Memory-Mapped File

```python
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```

**`torch.randint(len(data) - block_size, (batch_size,))`:** Picks `batch_size` random start positions. The upper bound is `len(data) - block_size` so that `data[i:i+block_size+1]` never reads past the end of the file.

**`data[i:i+block_size].astype(np.int64)`:** The memmap dtype is `uint16` (2 bytes). PyTorch embedding layers require `int64` (8 bytes) for index tensors. The `.astype()` conversion happens in numpy before the tensor is created.

**`x` and `y` are shifted by 1:** `y[j]` is always `x[j+1]` — the next token. This is the language modeling target.

**`pin_memory().to(device, non_blocking=True)`:** `pin_memory()` allocates the tensor in page-locked host memory, which allows the CPU→GPU transfer to happen asynchronously (the CPU can fetch the next batch while the GPU processes the current one). This is one of the few cases where the CPU and GPU are doing useful work simultaneously.

**Why recreate memmap each call:** Holding a persistent memmap reference causes a slow memory leak because Python's reference count prevents the OS from reclaiming the file mapping. See the linked Stack Overflow answer in the code.

---

### 2. Gradient Accumulation

**Definition:** Instead of one large batch, run N smaller micro-batches and sum their gradients before taking an optimizer step. The effective batch size is `batch_size × gradient_accumulation_steps`.

```python
# What gradient accumulation means mathematically:
# Loss for big batch = mean(losses for all samples)
# = sum(losses) / (batch_size * accum_steps)
# = mean over accum_steps of (loss for one micro-batch / accum_steps)
#
# So: scale each micro-batch loss by 1/accum_steps, then sum

gradient_accumulation_steps = 40
for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps   # ← scale before backward
    X, Y = get_batch('train')                       # prefetch next batch
    scaler.scale(loss).backward()                   # accumulate gradients

scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

**Why divide by `gradient_accumulation_steps`?** Each `.backward()` call *adds* to the existing `.grad` tensors. After all micro-steps, `.grad` contains the sum of gradients. We want the mean. If you don't divide, the effective learning rate is `accum_steps × lr`, causing instability.

**Why prefetch the next batch inside the accumulation loop?**  
```python
with ctx:
    logits, loss = model(X, Y)   # GPU is computing forward pass
    loss = loss / ...
X, Y = get_batch('train')        # CPU is reading disk — happens in parallel with GPU backward
scaler.scale(loss).backward()
```
The batch for the next micro-step is loaded from disk while the GPU is running the backward pass. On a GPU, the backward pass takes 2-3× longer than the forward pass, so there is time for the disk read.

**What breaks without the division:** A training run with `gradient_accumulation_steps=40` and `learning_rate=6e-4` would behave as if `lr=0.024`. Training would diverge on the first step.

---

### 3. Cosine Learning Rate Schedule with Linear Warmup

```python
def get_lr(it):
    # Phase 1: linear warmup
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # Phase 2: floor
    if it > lr_decay_iters:
        return min_lr
    # Phase 3: cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff: 1.0 → 0.0
    return min_lr + coeff * (learning_rate - min_lr)
```

**Phase 1 — warmup:** Learning rate rises linearly from ~0 to `learning_rate` over `warmup_iters` steps. Without warmup, large gradients at initialization (when the model is far from a good solution) can cause irreversible damage to the optimizer's moment estimates.

**Phase 2 — cosine decay:** The cosine schedule is smooth — it decays slowly at first (near the top, where the cosine curve is flat) and faster in the middle, which matches the dynamics of loss landscape exploration. Compared to a linear decay, cosine decay spends more time at higher learning rates.

**Phase 3 — floor:** After `lr_decay_iters`, the learning rate is floored at `min_lr ≈ learning_rate / 10`. Without a floor, the LR drops to zero and training stalls.

**Why `(it + 1) / (warmup_iters + 1)` instead of `it / warmup_iters`?**  
This ensures LR > 0 at iteration 0. If you used `it / warmup_iters`, iteration 0 would have LR = 0, and no learning would happen on the very first step.

---

### 4. Gradient Clipping

```python
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**Definition:** Gradient clipping rescales the gradient vector if its L2 norm exceeds a threshold: `g = g * (clip / ||g||)` if `||g|| > clip`.

**Why `scaler.unscale_` first?** `GradScaler` multiplied the loss (and therefore the gradients) by a large scale factor (e.g., 65536) during `backward()`. Before clipping, the gradients must be divided back to their true magnitude. `unscale_()` does this.

**What breaks without clipping:** Occasional large gradient spikes (common early in training or when the model encounters rare token combinations) can take the weights far from their optimal region in a single step — a "gradient explosion". Training loss spikes and may never recover.

**Why `grad_clip=1.0`?** This is the threshold from the GPT-3 paper. The gradient norm of a healthy transformer training run stays well below 1.0 most of the time; setting the clip at 1.0 only activates during anomalous spikes.

---

### 5. Loss Estimation Without Gradients

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

**`model.eval()`:** Switches Dropout off (outputs deterministic activations). Without this, the validation loss estimate would vary due to random dropout, making it unreliable.

**`@torch.no_grad()`:** Disables gradient computation during the entire function. This prevents the computation graph from being built, saving ~50% memory and ~30% time compared to running with gradients.

**`loss.item()`:** Transfers the loss scalar from GPU to CPU memory. This is a CPU-GPU synchronization point — the CPU blocks until the GPU finishes computing the loss. It is called `eval_iters` times, so minimizing eval frequency is important for training throughput.

**Why average over `eval_iters` batches?** One batch gives a noisy estimate of the true loss (high variance). Averaging over 200 batches gives a much more reliable number for deciding whether to save a checkpoint.

---

### 6. Checkpointing

```python
if losses['val'] < best_val_loss or always_save_checkpoint:
    best_val_loss = losses['val']
    if iter_num > 0:
        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_args': model_args,
            'iter_num': iter_num,
            'best_val_loss': best_val_loss,
            'config': config,
        }
        torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
```

**What to save:**
- `model`: weights (the only thing needed for inference)
- `optimizer`: momentum buffers (needed to resume training smoothly — without it, the optimizer starts "cold" and the first few steps after resuming are suboptimal)
- `model_args`: the architecture hyperparameters (needed to reconstruct the model object)
- `iter_num` and `best_val_loss`: training state
- `config`: the full set of hyperparameters (for reproducibility and logging)

**Why `raw_model.state_dict()` and not `model.state_dict()`?** In DDP, `model` is wrapped in a `DistributedDataParallel` container. `model.state_dict()` includes a `module.` prefix on every key. `raw_model = model.module if ddp else model` strips the container, giving a clean state dict that can be loaded without DDP.

**Why `if iter_num > 0`?** Skip saving at iter 0 — the model is randomly initialized and saving it wastes time. The check `losses['val'] < best_val_loss or always_save_checkpoint` uses short-circuit evaluation: if `always_save_checkpoint` is True, the loss comparison is skipped entirely.

---

## Reading the Source File

### `train.py` — the main training loop (lines 250–333)

```python
X, Y = get_batch('train')   # fetch the very first batch before the loop
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```
PyTorch does not have a built-in "set LR for this iteration" API. Instead, you directly mutate the `lr` key in each param_group dict. This is the standard pattern for custom LR schedules.

```python
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        ...
        if losses['val'] < best_val_loss or always_save_checkpoint:
            ...checkpoint...
```
Only `master_process` (rank 0 in DDP, or always in single-GPU) saves checkpoints and prints logs. This prevents N processes all trying to write the same file simultaneously.

```python
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
```
`require_backward_grad_sync` is set to `True` only on the last micro-step. This tells DDP to skip the all-reduce gradient synchronization until all micro-steps are done — see Module 07 for details.

```python
    scaler.unscale_(optimizer)
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
```
`optimizer.zero_grad(set_to_none=True)` sets gradient tensors to `None` instead of filling them with zeros. This is faster (avoids a memset) and uses slightly less memory (no zero-filled tensor allocated).

```python
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if local_iter_num >= 5:
        mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
    print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
```
`running_mfu` is an exponential moving average (α=0.9) of the per-iteration MFU. The first 5 iterations are discarded because CUDA kernel compilation and warmup make them unrepresentative.

---

## Why This Design

**Why a `while True` loop with a break, rather than `for i in range(max_iters)`?**  
The while loop allows resuming from a checkpoint: `iter_num` is loaded from the checkpoint and the loop runs until `iter_num > max_iters`. A `for` loop would restart from 0 on resume unless you manually implemented an offset.

**Why `always_save_checkpoint=True` by default for large models?**  
If the run crashes at iteration 599,999 (out of 600,000), you want to resume from 598,000 (the last eval), not from 0. For small models (shakespeare_char), you only save when val loss improves (`always_save_checkpoint=False`), since resuming would overfit.

**Why estimate loss over 200 batches, not 1?**  
One batch of 12 × 1024 = 12,288 tokens is too small to give a stable loss estimate. 200 batches = 2.4M tokens provides a mean with standard deviation ~0.01 nats — small enough to detect real improvement.

---

## Running the Tests

```bash
pytest tests/test_train_utils.py -v
```

All tests verify the `get_lr` function's mathematical contracts:
- `test_get_lr_warmup_is_monotonically_increasing`: LR rises during warmup
- `test_get_lr_at_end_of_warmup_approaches_max_lr`: warmup approaches but doesn't exceed max
- `test_get_lr_decay_is_monotonically_decreasing`: LR falls during cosine decay
- `test_get_lr_after_decay_iters_returns_min_lr`: floor is enforced
- `test_get_lr_never_exceeds_learning_rate`: hard upper bound

---

## Checkpoint ✓

```python
import math

def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

LR, WARMUP, DECAY, MIN_LR = 6e-4, 100, 5000, 6e-5

# Print LR at key milestones
for it in [0, 50, 100, 500, 2500, 5000, 6000]:
    print(f"iter {it:5d}: lr = {get_lr(it, LR, WARMUP, DECAY, MIN_LR):.6f}")
```

**Expected output (approximate):**
```
iter     0: lr = 0.000006
iter    50: lr = 0.000297
iter   100: lr = 0.000572
iter   500: lr = 0.000543
iter  2500: lr = 0.000330
iter  5000: lr = 0.000060
iter  6000: lr = 0.000060
```

---

## Exercises

**1 (Easy) — Plot the LR schedule:**  
Use matplotlib to plot `get_lr(it)` for `it` from 0 to 6000. Label the three phases (warmup, cosine, floor) with vertical lines. Verify the curve is smooth and never drops below `min_lr`.

**Success condition:** A clean plot with three distinct phases, no discontinuities.

**2 (Medium) — Measure gradient accumulation overhead:**  
Modify the training loop to time just the gradient accumulation steps (excluding eval). Compare wall-clock time for `gradient_accumulation_steps=1` vs. `gradient_accumulation_steps=8` with the same total number of tokens. Are they the same speed? Why or why not?

**Success condition:** Report throughput in tokens/second for both settings. Explain whether accumulation adds overhead (it should not on a GPU — the operations are identical).

**3 (Hard) — Implement learning rate finder:**  
Add a "LR range test" mode: run 100 training steps with LR increasing exponentially from `1e-7` to `1e-1`. Plot loss vs. LR. Find the LR that gives the steepest loss decrease — this is typically a good learning rate. Compare to nanoGPT's default of `6e-4` for the shakespeare_char config.

**Success condition:** A plot showing loss vs. LR, with the optimal region identified. Your suggested LR should be within 3× of `6e-4`.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| Cosine LR schedule | [Cyclical Learning Rates (Smith, 2017)](https://arxiv.org/abs/1506.01186) | Original paper on cycling LR; cosine schedule is a simplified version |
| Gradient clipping | [On the difficulty of training RNNs (Pascanu et al.)](https://arxiv.org/abs/1211.5063) | Section 5 explains gradient explosion and proves clipping is the right fix |
| Gradient accumulation | [PyTorch gradient accumulation tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html) | Official docs on backward() accumulation behavior |
| Checkpointing best practices | [PyTorch save and load tutorial](https://pytorch.org/tutorials/beginner/saving_loading_models.html) | What state_dict contains and how to resume training |

---

## What's Next

[Module 07 — Distributed Training](module_07_distributed_training.md): scale to multiple GPUs with DistributedDataParallel.
