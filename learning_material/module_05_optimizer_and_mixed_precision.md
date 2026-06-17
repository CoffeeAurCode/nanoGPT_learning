# Module 05 — Optimizer & Mixed Precision

**Source files:** `model.py` (`configure_optimizers`, `estimate_mfu`) · `train.py` lines 196–208  
**Estimated time:** 45 minutes  
**Next:** [Module 06 — Training Loop](module_06_training_loop.md)

---

## What You Are Building

This module covers everything between "model is built" and "training loop starts": configuring the AdamW optimizer with selective weight decay, setting up automatic mixed precision (AMP) with `torch.amp`, and understanding the Model FLOPs Utilization metric that tells you how efficiently you're using your GPU.

---

## Concept Deep-Dives

### 1. AdamW vs Adam: What Weight Decay Actually Does

**Definition:** Weight decay adds a penalty proportional to the L2 norm of each parameter to the loss: `loss_total = loss + λ/2 * ||θ||²`. In practice this shrinks weights toward zero on each step. AdamW implements this as a direct weight multiplication rather than a gradient modification (which is what `Adam` with `weight_decay` incorrectly does).

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 4)

# Adam with weight_decay (incorrect — mixes L2 into the gradient normalization)
optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

# AdamW (correct — applies weight decay separately from adaptive gradient scaling)
optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# Both have the same API, but AdamW is mathematically correct for adaptive optimizers
# Reference: Decoupled Weight Decay Regularization (Loshchilov & Hutter, 2019)
```

**How nanoGPT uses it:** AdamW with `weight_decay=0.1`, `beta1=0.9`, `beta2=0.95` (slightly higher β2 than the PyTorch default of 0.999, which GPT-3 found works better for LM pre-training).

**What breaks naively:** Using `Adam` with `weight_decay` applies weight decay through the adaptive moment normalization, which means the effective regularization is inconsistent across parameters with different gradient magnitudes. AdamW keeps weight decay clean and independent.

---

### 2. Selective Weight Decay: Which Parameters Should Decay?

**Definition:** Weight decay on biases and LayerNorm scale/shift parameters actively hurts convergence — these parameters don't represent magnitude-scalable features and should not be penalized toward zero.

**The rule nanoGPT uses:** Parameters with `ndim >= 2` (weight matrices, embedding tables) get decayed. Parameters with `ndim < 2` (biases, LayerNorm γ and β) do not.

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 4),    # .weight: (4,4) ndim=2 → decay; .bias: (4,) ndim=1 → no decay
    nn.LayerNorm(4),    # .weight: (4,) ndim=1 → no decay; .bias: (4,) ndim=1 → no decay
    nn.Embedding(10, 4) # .weight: (10,4) ndim=2 → decay
)

decay_params   = [p for p in model.parameters() if p.dim() >= 2]
nodecay_params = [p for p in model.parameters() if p.dim() < 2]

optimizer = torch.optim.AdamW([
    {'params': decay_params,   'weight_decay': 0.1},
    {'params': nodecay_params, 'weight_decay': 0.0},
], lr=1e-3)
```

**How nanoGPT uses it:** `configure_optimizers` does exactly this split, but iterates `named_parameters()` to print helpful diagnostics:

```python
decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
num_decay_params   = sum(p.numel() for p in decay_params)
num_nodecay_params = sum(p.numel() for p in nodecay_params)
print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
```

**What breaks naively:** Applying weight decay to LayerNorm scale (γ) shrinks it toward zero. At γ ≈ 0, LayerNorm outputs near-zero values everywhere — the model collapses. Training diverges or converges to a degenerate solution.

---

### 3. Fused AdamW: A CUDA Kernel Optimization

**Definition:** The fused AdamW kernel combines the parameter update (moment update + weight update) into a single CUDA kernel launch instead of multiple separate operations. This reduces GPU kernel launch overhead and memory bandwidth.

```python
import inspect
import torch

# Check if the fused version is available (PyTorch >= 2.0, CUDA only)
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
extra_args = dict(fused=True) if use_fused else dict()

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
```

**The `inspect.signature` trick:** Rather than try/except, nanoGPT checks the optimizer's signature to see if `fused` is an accepted keyword. This is more robust than a version check because it works correctly if PyTorch adds the parameter in a patch version.

**What breaks without it:** Training still works. On a GPU, the fused kernel provides ~20-30% speedup on the optimizer step (which is a small fraction of total training time). On CPU, the fused kernel is not available, so the check is necessary.

---

### 4. Mixed Precision Training: bfloat16 vs float16

**Definition:** Mixed precision uses a lower-precision data type (16-bit) for the forward and backward passes, but keeps weights in full precision (32-bit). This halves memory usage and speeds up GPU matrix multiplications.

```python
import torch
# float32: 32 bits, exponent range ±38, mantissa 23 bits (7 decimal digits)
# float16: 16 bits, exponent range ±4.5, mantissa 10 bits (3 decimal digits)
# bfloat16: 16 bits, exponent range ±38, mantissa 7 bits (2 decimal digits)

f32 = torch.tensor(1e-4, dtype=torch.float32)
f16 = f32.to(torch.float16)       # fine: 1e-4 is in range
bf16 = f32.to(torch.bfloat16)     # fine: same exponent range as f32

small = torch.tensor(1e-5, dtype=torch.float32)
print(small.to(torch.float16))    # 0.0   ← underflow (below f16 minimum)
print(small.to(torch.bfloat16))   # 1e-05 ← correct (bfloat16 keeps f32 range)
```

**bfloat16 vs float16:**
- float16 has more mantissa bits (higher precision per value) but a tiny exponent range — gradients can underflow to zero during backprop.
- bfloat16 has the same exponent range as float32 (no underflow risk) but fewer mantissa bits. For language modeling, the reduced precision is acceptable.
- **Recommendation:** Use bfloat16 on Ampere+ GPUs (A100, RTX 3090+). Use float16 on older GPUs.

**How nanoGPT auto-selects:**
```python
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
```

---

### 5. `torch.amp.autocast` and `GradScaler`

**Definition:** `autocast` is a context manager that automatically casts eligible operations to the lower-precision dtype. `GradScaler` compensates for float16's narrow exponent range by scaling up the loss before backprop (so gradients don't underflow) and then unscaling before the optimizer step.

```python
from contextlib import nullcontext
import torch

device_type = 'cuda'
ptdtype = torch.bfloat16
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# GradScaler is only needed for float16 (bfloat16 doesn't underflow)
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# In the training loop:
with ctx:
    logits, loss = model(X, Y)     # forward pass in bfloat16
    loss = loss / gradient_accumulation_steps

scaler.scale(loss).backward()      # scales loss × scale_factor, then backprop
scaler.unscale_(optimizer)         # divides gradients by scale_factor
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # clip after unscale
scaler.step(optimizer)             # checks for NaN/inf, skips step if found
scaler.update()                    # adjusts scale_factor for next iteration
```

**What `autocast` does:** Inside `with ctx:`, operations like matrix multiplication run in bfloat16 for speed. Certain operations (softmax, layer norm) that need higher precision are kept in float32 automatically by PyTorch.

**What `GradScaler` does:** Multiplies the loss by a large factor (e.g., 65536) before backward, so small gradients that would underflow to 0 in float16 become representable. After backward, divides all gradients back by the same factor before the optimizer step.

**What breaks with `nullcontext` on CPU:** Running `autocast` with `device_type='cpu'` works but is slow because CPUs don't have native bfloat16 hardware acceleration. nanoGPT uses `nullcontext()` on CPU (which does nothing) to avoid the overhead.

---

### 6. Model FLOPs Utilization (MFU)

**Definition:** MFU = (actual FLOPs per second) / (theoretical peak FLOPs per second). It measures how efficiently your training loop utilizes the GPU's floating-point capability.

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    N = self.get_num_params()
    cfg = self.config
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size

    # FLOPs per token: 6N (matmuls) + 12LHQT (attention)
    flops_per_token  = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter   = flops_per_fwdbwd * fwdbwd_per_iter

    flops_achieved = flops_per_iter / dt     # per second
    flops_promised = 312e12                  # A100 bfloat16 peak
    return flops_achieved / flops_promised
```

The `6N` comes from the PaLM paper (Appendix B): each parameter is touched 6 times per forward+backward pass (2 for forward matmul, 4 for backward). A good MFU for transformer training is 40–60%. Above 50% is excellent.

**What a low MFU indicates:** Data loading is the bottleneck, or the batch size is too small to keep GPU cores busy, or gradient accumulation overhead is high.

---

## Reading the Source File

### `model.py` — `configure_optimizers`

```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
```
Filter to only trainable parameters. Frozen parameters (e.g., from a partially frozen fine-tune) are excluded.

```python
    decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
```
The dimension check: weight matrices and embedding tables are ≥ 2D; biases and LayerNorm parameters are 1D.

```python
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                  **(dict(fused=True) if use_fused else dict()))
```
Using `**dict()` pattern instead of `if/else` is a clean way to conditionally add keyword arguments.

---

## Why This Design

**Why β2 = 0.95 instead of the PyTorch default of 0.999?**  
β2 controls how quickly the second moment estimate adapts to new gradient magnitudes. Higher β2 = slower adaptation = more stable but less responsive. For language model pre-training on diverse text, the gradient distribution changes slowly, so 0.999 works. For smaller or more specialized models, 0.95 or 0.99 can converge faster. nanoGPT uses 0.95 following GPT-3.

**Why `weight_decay=0.1`?**  
A stronger regularizer than the typical 0.01. For large language models, higher weight decay helps prevent overfitting on the more frequent tokens. Chinchilla used 0.1.

**Why not just always use bfloat16?**  
Older GPUs (anything before Ampere, e.g., V100, RTX 20-series) don't have native bfloat16 hardware support. On those GPUs, bfloat16 operations fall back to software emulation, which is slower than float32. The auto-detection in nanoGPT handles this correctly.

---

## Running the Tests

```bash
pytest tests/test_model.py -v -k "optimizer or param"
```

Key tests to look at:
- `test_gpt_loss_decreases_after_one_step`: the optimizer is working if loss decreases

There are no dedicated optimizer unit tests because `configure_optimizers` wraps PyTorch's optimizer directly. The meaningful test is the end-to-end training test.

---

## Checkpoint ✓

```python
import torch
from model import GPTConfig, GPT

cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=8,
                vocab_size=64, dropout=0.0, bias=False)
model = GPT(cfg)

optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=1e-3,
    betas=(0.9, 0.95), device_type='cpu'
)

# Verify two parameter groups
assert len(optimizer.param_groups) == 2
decay_group   = optimizer.param_groups[0]
nodecay_group = optimizer.param_groups[1]
assert decay_group['weight_decay'] == 0.1
assert nodecay_group['weight_decay'] == 0.0

print(f"decayed tensors: {len(decay_group['params'])}")
print(f"non-decayed tensors: {len(nodecay_group['params'])}")
```

**Expected:** Two groups. All biases and LayerNorm parameters in the non-decayed group.

---

## Exercises

**1 (Easy) — Identify every parameter in each decay group:**  
Print the name of every parameter in `param_dict` and whether it goes in the decay or no-decay group. List all LayerNorm parameters and confirm they are in no-decay.

**Success condition:** You can explain the group membership of every parameter by name.

**2 (Medium) — Compare AdamW vs SGD:**  
Train the `shakespeare_char` config for 1000 steps with AdamW (default) and with SGD + momentum. Plot both val loss curves. Write two sentences explaining why AdamW converges faster for transformers.

**Success condition:** AdamW reaches lower val loss faster. Your explanation mentions adaptive learning rates and the importance of per-parameter scaling for embedding matrices.

**3 (Hard) — Implement gradient checkpointing:**  
`torch.utils.checkpoint.checkpoint` trades recomputation for memory: it does not store intermediate activations during the forward pass, instead recomputing them during backward. Modify `Block.forward()` to optionally use gradient checkpointing. Measure memory usage with and without it using `torch.cuda.max_memory_allocated()`.

**Success condition:** Memory usage drops by ~30-50% with checkpointing enabled. Training still converges to the same loss. Throughput decreases slightly due to recomputation.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| AdamW | [Decoupled Weight Decay Regularization (Loshchilov & Hutter)](https://arxiv.org/abs/1711.05101) | 4-page paper explaining why Adam's weight decay is wrong and AdamW fixes it |
| Mixed precision training | [PyTorch AMP tutorial](https://pytorch.org/docs/stable/amp.html) | Official docs with runnable examples of autocast and GradScaler |
| FLOPs estimation for transformers | [PaLM paper Appendix B](https://arxiv.org/abs/2204.02311) | Derivation of the 6N formula used in `estimate_mfu` |

---

## What's Next

[Module 06 — Training Loop](module_06_training_loop.md): the full training loop — batch sampling, gradient accumulation, cosine LR schedule, and checkpointing.
