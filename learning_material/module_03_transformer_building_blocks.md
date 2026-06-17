# Module 03 — Transformer Building Blocks

**Source files:** `model.py` lines 1–106 (`LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`)  
**Estimated time:** 60 minutes  
**Next:** [Module 04 — The GPT Model](module_04_the_gpt_model.md)

---

## What You Are Building

One transformer block — the repeated unit that makes up the GPT stack. By the end of this module you will be able to write `LayerNorm`, `CausalSelfAttention`, `MLP`, and `Block` from scratch, explain what each line does, and describe what would break if you removed or simplified any part.

---

## Concept Deep-Dives

### 1. Layer Normalization

**Definition:** Layer Normalization computes the mean and variance across the feature dimension of each token independently, then rescales with learnable parameters γ (weight) and β (bias). This keeps activations in a healthy range throughout training.

**Minimal standalone example:**

```python
import torch
import torch.nn.functional as F

# Manual Layer Norm (what F.layer_norm does internally):
x = torch.tensor([[2.0, 4.0, 6.0, 8.0]])  # shape (1, 4)
mean = x.mean(dim=-1, keepdim=True)        # 5.0
var  = x.var(dim=-1, keepdim=True, unbiased=False)  # 5.0
x_norm = (x - mean) / (var + 1e-5).sqrt() # ~[-1.34, -0.45, 0.45, 1.34]

# With learnable scale (gamma) and shift (beta):
gamma = torch.ones(4)
beta  = torch.zeros(4)
out = gamma * x_norm + beta
print(out)   # tensor([[-1.3416, -0.4472,  0.4472,  1.3416]])

# PyTorch's built-in:
print(F.layer_norm(x, normalized_shape=(4,), weight=gamma, bias=beta, eps=1e-5))
```

**How nanoGPT uses it:** `LayerNorm` in `model.py` wraps `F.layer_norm` with one difference: the `bias` parameter can be `None`. PyTorch's `nn.LayerNorm` always creates a bias tensor, which slightly harms performance in large models.

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

**What breaks without LayerNorm:** Without normalization, activations in deep networks grow exponentially through residual connections, causing NaN gradients. Training becomes impossible beyond a few layers.

---

### 2. Multi-Head Causal Self-Attention

**Definition:** Self-attention allows each token to aggregate information from other tokens in the sequence. "Multi-head" means this is done in parallel with multiple learned projections ("heads"). "Causal" means token i can only attend to tokens 0…i — not future tokens.

#### Step-by-step construction:

**Step 1: The QKV projection**

```python
# For a sequence of length T with embedding dimension C:
# One linear layer produces Q, K, V for all heads simultaneously.
import torch
import torch.nn as nn

C, n_head = 8, 2
hs = C // n_head   # head size = 4
c_attn = nn.Linear(C, 3 * C, bias=False)

x = torch.randn(1, 4, C)   # (batch=1, T=4, C=8)
qkv = c_attn(x)             # (1, 4, 24)
q, k, v = qkv.split(C, dim=2)  # each (1, 4, 8)

# Reshape into (B, n_head, T, hs)
q = q.view(1, 4, n_head, hs).transpose(1, 2)  # (1, 2, 4, 4)
k = k.view(1, 4, n_head, hs).transpose(1, 2)
v = v.view(1, 4, n_head, hs).transpose(1, 2)
```

**Step 2: Scaled dot-product attention (manual)**

```python
import math

# Attention scores: (B, n_head, T, T)
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))

# Causal mask: set upper triangle to -inf so softmax → 0
mask = torch.tril(torch.ones(4, 4))
att = att.masked_fill(mask == 0, float('-inf'))
att = torch.softmax(att, dim=-1)

# Weighted sum of values: (B, n_head, T, hs)
y = att @ v

# Merge heads back: (B, T, C)
y = y.transpose(1, 2).contiguous().view(1, 4, C)
print(y.shape)   # torch.Size([1, 4, 8])
```

**Step 3: Flash Attention (what PyTorch 2.0 provides)**

```python
# Equivalent to steps 2 above, but:
# - Never materializes the (T, T) attention matrix in HBM (GPU high-bandwidth memory)
# - Uses tiled SRAM computation → ~10x lower memory, ~3x faster for T > 512
y = torch.nn.functional.scaled_dot_product_attention(
    q, k, v,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,    # handles the causal mask internally
)
```

**How nanoGPT uses it:**

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
```

Notice `register_buffer`: the causal mask is stored as a non-trainable buffer (not a parameter). It moves to GPU with `model.to(device)` and is saved in checkpoints, but gradient computation never touches it.

**What breaks without the causal mask:** During training, every token can attend to future tokens. The model can "cheat" — predict token 5 by looking at tokens 6, 7, 8. Training loss drops to near zero, but at inference time (when the future doesn't exist), the model generates garbage.

**What breaks if n_embd is not divisible by n_head:** The `view(B, T, n_head, C // n_head)` reshape would fail because the integer division would not evenly split the embedding. The `assert config.n_embd % config.n_head == 0` catches this immediately.

---

### 3. The MLP (Feed-Forward Network)

**Definition:** The position-wise MLP applies two linear transformations with a non-linear activation in between. It runs independently on each token position.

```python
# What the MLP does conceptually:
# Project up: n_embd → 4 * n_embd
# Activate with GELU
# Project down: 4 * n_embd → n_embd

import torch
import torch.nn as nn

class MinimalMLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.up   = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.down = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.down(self.gelu(self.up(x)))
```

**Why GELU instead of ReLU?**  
GELU (Gaussian Error Linear Unit) multiplies the input by its Gaussian CDF: `x * Φ(x)`. For large positive inputs it is ≈ x (like ReLU); for negative inputs it is ≈ 0 (like ReLU). The difference: GELU is smooth and has small but non-zero gradient for negative inputs. Empirically, transformers trained with GELU outperform ReLU on language tasks — used in GPT-2, BERT, and nearly every large LM since.

```python
import torch
x = torch.linspace(-3, 3, 7)
relu = torch.relu(x)
gelu = torch.nn.functional.gelu(x)
print("x:   ", x.tolist())    # [-3, -2, -1,  0,  1,  2,  3]
print("relu:", relu.tolist())  # [ 0,  0,  0,  0,  1,  2,  3]
print("gelu:", [f"{v:.2f}" for v in gelu.tolist()])
# [-0.00, -0.05, -0.16, 0.00, 0.84, 1.95, 3.00]
```

**Why 4× expansion?** The factor 4 is from the original Transformer paper (Vaswani et al., 2017). It was a pragmatic choice that worked well. GPT-2 kept it. It gives the MLP enough capacity to act as a "key-value memory" that stores factual associations.

**What breaks without the MLP:** Self-attention alone can only compute weighted sums of value vectors — it is a linear operation over the value space. The MLP introduces the non-linearity required to learn complex functions. A transformer with no MLP converges to a much higher loss.

---

### 4. The Transformer Block with Residual Connections and Pre-LN

**Definition:** A residual connection adds the input of a sub-layer directly to its output: `output = x + f(x)`. Pre-LN (Pre-Layer Normalization) applies LayerNorm *before* the sub-layer, not after.

```python
# Post-LN (original Transformer, Vaswani 2017):
def post_ln_block(x):
    x = layer_norm(x + attention(x))  # norm after residual
    x = layer_norm(x + mlp(x))
    return x

# Pre-LN (GPT-2, nanoGPT):
def pre_ln_block(x):
    x = x + attention(layer_norm(x))  # norm before sub-layer
    x = x + mlp(layer_norm(x))
    return x
```

**Why residual connections?** Without them, gradients must flow through all layer transformations to reach early layers. In a 12-layer network, multiplying by 12 near-zero Jacobians causes the gradient to vanish. With residual connections, gradients have a "highway" that bypasses each layer.

**Why Pre-LN?** Post-LN requires very careful learning rate warmup — the gradients at early training are very large and can destabilize training. Pre-LN is more stable: the gradient scale at initialization is O(1) regardless of depth. GPT-2, GPT-3, and most modern LLMs use Pre-LN.

```python
# nanoGPT's Block:
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp  = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # Pre-LN attention
        x = x + self.mlp(self.ln_2(x))    # Pre-LN MLP
        return x
```

**What breaks with Post-LN in a deep model without warmup:** The training loss shoots to NaN on the first iteration because gradients are unbounded at initialization. You can recover with a very small learning rate and long warmup, but Pre-LN is strictly easier to use.

---

## Reading the Source File

### `model.py` lines 18–106 — complete walkthrough

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
```
`nn.Parameter` wraps a tensor and registers it as a trainable parameter. `torch.ones` initializes γ to 1 (identity scale) and `torch.zeros` initializes β to 0 (no shift) — the standard initialization for LayerNorm.

```python
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```
`self.weight.shape` is the `normalized_shape` argument — tells PyTorch to normalize over the last `ndim` dimensions. `1e-5` is the epsilon added to the variance to prevent division by zero.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        ...
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
```
A single `Linear` with output size `3 * n_embd` instead of three separate `Linear` layers for Q, K, V. This is more efficient: one matrix multiplication vs. three, and the GPU can execute the larger GEMM with better utilization. The output is later split with `.split(n_embd, dim=2)`.

```python
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(...))
```
`hasattr()` detects whether Flash Attention is available (PyTorch ≥ 2.0). The causal mask buffer is only allocated if we need the manual fallback — saves memory on PyTorch 2.0+.

```python
    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
```
After the QKV projection, each of Q/K/V has shape `(B, T, C)`. The `.view()` splits the `C` dimension into `n_head` heads each of size `C // n_head`, giving shape `(B, T, n_head, hs)`. The `.transpose(1, 2)` swaps T and n_head so the shape becomes `(B, n_head, T, hs)` — the format required for batched matrix multiplication across heads.

```python
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
```
The `1.0 / math.sqrt(k.size(-1))` factor scales the dot products before softmax. Without it, the dot products grow with head size, pushing softmax into regions with near-zero gradients (the "saturation" problem).

```python
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
```
`.contiguous()` is required before `.view()` because `.transpose()` creates a non-contiguous tensor (the memory layout doesn't match the logical shape). `.view()` requires contiguous memory.

---

## Why This Design

**Why one c_attn instead of three separate Q/K/V projections?**  
One large GEMM is faster than three small GEMMs on GPU. The results are identical — the weight matrices are equivalent to three stacked matrices, and `.split()` separates them after the fact.

**Why `register_buffer` for the causal mask?**  
The mask is not a trainable parameter — it is a fixed triangular matrix. Using `register_buffer` ensures it: (1) moves to GPU with `model.to(device)`, (2) is included in `state_dict()` (saved in checkpoints), (3) is not passed to the optimizer. Using a plain `torch.tensor` would lose it during device transfer.

**Why merge heads with `.contiguous().view()` instead of `.reshape()`?**  
`.view()` is zero-copy when the tensor is contiguous — it just reinterprets the same memory. `.reshape()` sometimes needs to copy. Since this runs billions of times during training, the copy would be significant. The `.contiguous()` call makes the copy explicit when needed.

---

## Running the Tests

```bash
pytest tests/test_model.py -v -k "LayerNorm or Attention or MLP or Block"
```

Key tests:
- `test_layer_norm_normalizes_input`: verifies near-zero mean and near-unit variance
- `test_causal_self_attention_is_causal`: the core correctness test — changing the last token must not affect earlier positions
- `test_mlp_hidden_dimension_is_4x_embd`: verifies the 4× expansion factor
- `test_block_output_shape`: verifies the residual connection preserves shape

---

## Checkpoint ✓

Build and run a single transformer block from scratch in a REPL:

```python
import torch
from model import GPTConfig, Block

cfg = GPTConfig(block_size=8, vocab_size=64, n_layer=1, n_head=2,
                n_embd=16, dropout=0.0, bias=True)
block = Block(cfg)
block.eval()

x = torch.randn(1, 6, 16)   # batch=1, seq_len=6, n_embd=16
out = block(x)
print(out.shape)   # torch.Size([1, 6, 16])

# Verify causal masking:
x2 = x.clone()
x2[0, -1] = torch.randn(16)  # change last token
out2 = block(x2)
print(torch.allclose(out[0, :-1], out2[0, :-1], atol=1e-5))  # True
```

If both assertions hold, your block is correctly shaped and causal.

---

## Exercises

**1 (Easy) — Visualize attention weights:**  
Temporarily modify `CausalSelfAttention.forward()` to return `att` (the attention matrix) alongside `y`. Create a random input, run a single forward pass, and plot the attention matrix as a heatmap using matplotlib. Verify the upper triangle is zero (causal masking).

**Success condition:** You produce a plot where every cell above the diagonal is exactly 0, and each row sums to 1.

**2 (Medium) — Implement attention without Flash Attention:**  
Set `self.flash = False` in `CausalSelfAttention.__init__` and verify the manual attention path produces the same output as Flash Attention for the same Q/K/V matrices (they should agree to within 1e-4 due to floating-point order differences).

**Success condition:** `torch.allclose(flash_output, manual_output, atol=1e-4)` returns `True` for a random input.

**3 (Hard) — Replace GELU with ReLU and measure the difference:**  
Replace `nn.GELU()` with `nn.ReLU()` in `MLP`. Train both versions on `shakespeare_char` for 500 iterations. Plot both validation loss curves. Calculate the parameter count difference (it is zero — same architecture, different activation). Explain why GELU might outperform ReLU in this setting.

**Success condition:** You have two loss curves and a written explanation of the GELU vs. ReLU trade-off for language models.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| Attention mechanism | [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/abs/1706.03762) | The original paper; equations 1–3 are exactly what `CausalSelfAttention` implements |
| Flash Attention | [Flash Attention paper (Dao et al.)](https://arxiv.org/abs/2205.14135) | Explains why memory bandwidth (not FLOPs) is the bottleneck, and how tiling solves it |
| GELU activation | [GELU paper (Hendrycks & Gimpel)](https://arxiv.org/abs/1606.08415) | Short paper explaining the Gaussian CDF formulation |
| Layer Normalization | [Layer Normalization paper (Ba et al.)](https://arxiv.org/abs/1607.06450) | Original paper; contrast with Batch Normalization in the intro |
| Pre-LN vs Post-LN | [On Layer Normalization in the Transformer (Xiong et al.)](https://arxiv.org/abs/2002.04745) | Proves Pre-LN has better gradient flow; explains why GPT-2 switched |

---

## What's Next

[Module 04 — The GPT Model](module_04_the_gpt_model.md): assemble the building blocks into the full GPT class, add embeddings, weight tying, and autoregressive generation.
