# Module 04 — The GPT Model

**Source files:** `model.py` lines 108–330 (`GPTConfig`, `GPT`)  
**Estimated time:** 60 minutes  
**Next:** [Module 05 — Optimizer & Mixed Precision](module_05_optimizer_and_mixed_precision.md)

---

## What You Are Building

The `GPT` class assembles the building blocks from Module 03 into the complete language model: token embeddings, positional embeddings, the stacked transformer blocks, a final LayerNorm, and the language model head. You will also understand weight tying, weight initialization, and the autoregressive generation algorithm.

---

## Concept Deep-Dives

### 1. `@dataclass` for Configuration

**Definition:** `@dataclass` is a Python decorator that auto-generates `__init__`, `__repr__`, and `__eq__` from annotated class attributes. It produces a clean, typed container for hyperparameters.

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

cfg = GPTConfig(n_layer=6, n_head=6, n_embd=384)
print(cfg)  # GPTConfig(block_size=1024, vocab_size=50304, n_layer=6, ...)
print(cfg.n_embd)   # 384
```

**How nanoGPT uses it:** Every layer receives `config` and reads `config.n_embd`, `config.n_head`, etc. This is the **single source of truth** for all dimensions — change one value in `GPTConfig` and it propagates everywhere.

**What breaks naively:** Without a config object, you would pass `n_embd`, `n_head`, `n_layer` as separate arguments to each layer. Adding a new hyperparameter would require updating every function signature.

---

### 2. Token and Positional Embeddings

**Definition:** An embedding table maps each integer token ID to a dense vector. Positional embeddings map each position (0, 1, 2, ..., T-1) to a dense vector. Both are learned.

```python
import torch
import torch.nn as nn

vocab_size, n_embd = 50304, 768
block_size = 1024

wte = nn.Embedding(vocab_size, n_embd)   # Token embedding table
wpe = nn.Embedding(block_size, n_embd)  # Positional embedding table

# For a batch of token indices:
idx = torch.tensor([[15496, 11, 995, 0]])   # shape (1, 4)
pos = torch.arange(4)                       # [0, 1, 2, 3]

tok_emb = wte(idx)   # (1, 4, 768) — look up each token's vector
pos_emb = wpe(pos)   # (4, 768)    — look up each position's vector
x = tok_emb + pos_emb  # (1, 4, 768) — token meaning + position information
```

**Why both?** Attention is permutation-invariant — it doesn't know if token 3 comes before token 5. Without positional embeddings, the model would produce the same output for any permutation of the same tokens. Positional embeddings give the model an inductive bias about order.

**Why learned positional embeddings?** The original Transformer tested both fixed (sinusoidal) and learned positional embeddings and found learned ones work just as well. GPT-2 used learned embeddings. There is no principled reason to prefer sinusoidal for a fixed context window.

**What breaks if you add instead of concatenate:** Summing embeddings keeps the dimension at `n_embd`, which is efficient. If you concatenated, the model dimension would become `2 * n_embd` and every subsequent layer would need adjustment.

---

### 3. `nn.ModuleDict` and `nn.ModuleList`

**Definition:** `nn.ModuleDict` and `nn.ModuleList` are container classes that register sub-modules so PyTorch can find their parameters. Plain Python dicts and lists are invisible to `model.parameters()`.

```python
import torch.nn as nn

# Wrong: parameters in a plain dict are NOT registered
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = {'a': nn.Linear(4, 4)}   # invisible to optimizer!

# Right: use ModuleDict
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleDict({'a': nn.Linear(4, 4)})

good = GoodModel()
print(list(good.parameters()))  # includes Linear parameters
bad = BadModel()
print(list(bad.parameters()))   # empty!
```

**How nanoGPT uses it:**

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    drop = nn.Dropout(config.dropout),
    h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
```

The `h` (for "hidden") key holds all transformer blocks in a `ModuleList`. Iterating `self.transformer.h` applies them in order.

**What breaks with a plain list:** If you wrote `self.h = [Block(config) for _ in range(n_layer)]`, `model.to(device)` would not move the blocks to GPU, `model.state_dict()` would not include their weights, and the optimizer would not update them.

---

### 4. Weight Tying

**Definition:** Weight tying makes the token embedding table `wte` and the output projection (`lm_head`) share the same weight matrix.

```python
# After building both layers:
self.transformer.wte.weight = self.lm_head.weight
# They now point to the same tensor object in memory.
```

**Why this works:**
- `wte` maps token ID → embedding vector (vocab_size × n_embd lookup table)
- `lm_head` maps final hidden state → vocabulary logits (n_embd × vocab_size linear layer)
- These are transposes of each other. The embedding "encodes" a token; the lm_head "decodes" a hidden state into token space. Tying them enforces consistency.

**The benefit:** Saves `vocab_size × n_embd` parameters — for GPT-2 (50257 × 768), that is ~38.6 million parameters — without hurting performance. Press & Wolf (2017) showed tying improves perplexity compared to separate matrices.

**What breaks naively:** Using separate `wte` and `lm_head` works fine but uses more memory and parameters. There is no correctness issue — just an efficiency one.

---

### 5. Weight Initialization

**Definition:** The initial values of weights affect convergence speed and stability. GPT-2 uses normal distribution with std=0.02 for all weights, and a special scaled initialization for residual projections.

```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```

The `self.apply(_init_weights)` call applies this function recursively to every sub-module.

**Residual scaling:** After the default initialization, residual projections (`c_proj.weight`) are scaled down:

```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

**Why scale down residual projections?** At initialization, each transformer block adds `f(x)` to `x`. If all `f(x)` have similar magnitude, the residual stream grows as O(√n_layer) across the stack. Scaling the output projection by `1/√(2 * n_layer)` keeps the residual stream variance constant at initialization, regardless of depth. This is from the GPT-2 paper.

**What breaks with the wrong initialization:** std=0.02 was tuned for GPT-2's architecture. Too large: activations explode, NaN gradients on step 1. Too small: training is slow because weights start far from useful values. Without the residual scaling: deep networks are less stable at initialization.

---

### 6. The Forward Pass

```python
def forward(self, idx, targets=None):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)

    tok_emb = self.transformer.wte(idx)   # (b, t, n_embd)
    pos_emb = self.transformer.wpe(pos)   # (t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)
    for block in self.transformer.h:
        x = block(x)
    x = self.transformer.ln_f(x)

    if targets is not None:
        logits = self.lm_head(x)                    # (b, t, vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        logits = self.lm_head(x[:, [-1], :])        # (b, 1, vocab_size)
        loss = None

    return logits, loss
```

**Training path** (`targets is not None`): Run lm_head on all positions. Reshape logits to `(b*t, vocab_size)` and targets to `(b*t,)` for cross_entropy. `ignore_index=-1` allows masking padding positions.

**Inference path** (`targets is None`): Only forward lm_head on the **last position** — `x[:, [-1], :]`. The `[-1]` (list, not int) preserves the time dimension. This saves compute because we only need the next-token distribution.

**What breaks naively:** If you call `self.lm_head(x[:, -1, :])` (int index instead of list), the shape becomes `(b, vocab_size)` and the generate loop breaks because it expects a time dimension.

---

### 7. Autoregressive Generation

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

**Temperature:** Dividing logits by `temperature` before softmax sharpens (T < 1) or flattens (T > 1) the distribution. At T → 0, always picks the most probable token (greedy). At T = 1, samples from the model's exact distribution.

**Top-k:** Keeps only the k most probable tokens, setting all others to -∞. This prevents the model from accidentally sampling very low-probability tokens (typos, gibberish).

**`@torch.no_grad()`:** Disables gradient tracking. During generation we don't need gradients, and tracking them would waste memory proportional to sequence length.

**Context cropping:** `idx[:, -self.config.block_size:]` ensures we never feed more tokens than the model was trained on. Without this, the positional embedding lookup would request position > block_size and raise an index error.

---

## Reading the Source File

### `model.py` lines 108–330 — key sections

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304   # 50257 padded to nearest multiple of 64
    ...
```
`50304 = 50257 + 47`. The padding to a multiple of 64 aligns the embedding matrix rows to GPU memory boundaries, allowing vectorized operations to be more efficient (no partial cache lines).

```python
class GPT(nn.Module):
    def __init__(self, config):
        ...
        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(...),
            wpe  = nn.Embedding(...),
            drop = nn.Dropout(...),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(...),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
```
Note `lm_head` is defined *outside* the `transformer` `ModuleDict`. This is intentional — the weight tying line works by reference; the lm_head's weight becomes the same tensor as wte's weight. If lm_head were inside `transformer`, state_dict operations might double-save or double-load the shared weight.

```python
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
```
`self.apply()` traverses every sub-module depth-first and calls `_init_weights` on each. The second loop re-initializes only `c_proj` weights (output projections of attention and MLP) with the scaled std.

---

## Why This Design

**Why not use `nn.Transformer`?**  
PyTorch's built-in `nn.Transformer` is encoder-decoder and uses Post-LN. A decoder-only Pre-LN transformer requires building it from scratch anyway. The 300 lines of `model.py` are more readable than adapting `nn.TransformerDecoder`.

**Why `GPTConfig` as a dataclass and not a dict?**  
Type annotations. `config.n_embd` raises `AttributeError` if misspelled; `config['n_embd']` raises `KeyError`. Both catch bugs, but the dot-notation dataclass reads more naturally in deeply nested code. The `@dataclass` decorator also provides free `__repr__` for debugging.

**Why `vocab_size=50304` in the default instead of `50257`?**  
The 64-alignment improves CUDA kernel efficiency. The extra 47 tokens (IDs 50257–50303) are never seen in the training data, so the model learns to assign them near-zero probability. The lm_head weight matrix is slightly larger but the speedup is worth it on A100/H100 GPUs.

---

## Running the Tests

```bash
pytest tests/test_model.py -v -k "gpt"
```

Key tests:
- `test_gpt_forward_no_targets_returns_last_position_logits`: verifies the inference optimization
- `test_gpt_weight_tying`: verifies `wte.weight is lm_head.weight` (same object)
- `test_gpt_loss_decreases_after_one_step`: end-to-end sanity check
- `test_gpt_generate_extends_sequence_by_max_new_tokens`: generation length
- `test_gpt_generate_temperature_zero_is_deterministic`: verify low temperature = greedy

---

## Checkpoint ✓

```python
import torch
from model import GPTConfig, GPT

cfg = GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=8,
                vocab_size=64, dropout=0.0, bias=False)
model = GPT(cfg)
model.eval()

# 1. Forward pass with loss
idx     = torch.tensor([[10, 20, 30, 40]])
targets = torch.tensor([[20, 30, 40, 50]])
logits, loss = model(idx, targets)
print(f"loss: {loss.item():.4f}")   # ~4.1 (≈ log(64), random init)

# 2. Weight tying
print(model.transformer.wte.weight is model.lm_head.weight)  # True

# 3. Generate 5 tokens
out = model.generate(torch.tensor([[5]]), max_new_tokens=5, temperature=0.8)
print(out.shape)   # torch.Size([1, 6])
```

---

## Exercises

**1 (Easy) — Verify weight tying saves parameters:**  
Count the parameters with and without weight tying and confirm the difference equals `vocab_size × n_embd`.

```python
model_tied = GPT(cfg)
# Manually un-tie
model_untied = GPT(cfg)
model_untied.lm_head = torch.nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
model_untied.lm_head.weight.data = model_untied.transformer.wte.weight.data.clone()

tied_params   = sum(p.numel() for p in model_tied.parameters())
untied_params = sum(p.numel() for p in model_untied.parameters())
print(untied_params - tied_params)    # should equal vocab_size * n_embd
print(cfg.vocab_size * cfg.n_embd)
```

**Success condition:** Both print statements give the same number.

**2 (Medium) — Inspect the attention at different temperatures:**  
Generate 20 tokens from a pretrained (or freshly initialized) model at temperatures 0.1, 0.8, 1.5, and 2.0. Print the output for each. Describe in one sentence what you observe about diversity and coherence as temperature changes.

**Success condition:** You can articulate the temperature trade-off in your own words, with examples from the actual outputs.

**3 (Hard) — Add sinusoidal positional embeddings:**  
Replace the learned `wpe` embedding with fixed sinusoidal positional encodings (from the original Vaswani et al. 2017 paper). Run training for 1000 iterations on `shakespeare_char` and compare val loss to the learned embedding version. Which converges faster? Hypothesize why.

**Success condition:** Both models train successfully (no NaN). You have a loss comparison and a written hypothesis.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| Weight tying | [Using the Output Embedding to Improve Language Models (Press & Wolf)](https://arxiv.org/abs/1608.05859) | Original paper proving weight tying improves perplexity |
| GPT-2 architecture | [Language Models are Unsupervised Multitask Learners (Radford et al.)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | The GPT-2 paper; Appendix A describes the initialization nanoGPT replicates |
| Autoregressive generation | [The Illustrated GPT-2 (Jay Alammar)](https://jalammar.github.io/illustrated-gpt2/) | Best visual explanation of the generation loop and attention mechanisms |
| `nn.Module` internals | [PyTorch Module source](https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/module.py) | Understand how `apply()`, `parameters()`, and `state_dict()` work |

---

## What's Next

[Module 05 — Optimizer & Mixed Precision](module_05_optimizer_and_mixed_precision.md): configure AdamW with selective weight decay and set up mixed-precision training.
