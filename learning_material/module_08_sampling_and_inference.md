# Module 08 — Sampling & Inference

**Source files:** `sample.py` · `model.py` (`generate`)  
**Estimated time:** 45 minutes  
**Next:** [Module 09 — Pretrained Weights & Fine-tuning](module_09_pretrained_weights_and_finetuning.md)

---

## What You Are Building

`sample.py` loads a trained checkpoint (or a pretrained GPT-2 model), encodes a text prompt, and runs the `GPT.generate()` loop to produce new text. By the end of this module you will understand every parameter in `sample.py`, implement temperature scaling and top-k filtering from scratch, and know why `model.eval()` and `@torch.no_grad()` are not optional.

---

## Concept Deep-Dives

### 1. `model.eval()` vs. `model.train()`

**Definition:** `model.eval()` switches the model into inference mode. The key effects: Dropout layers output their full input (no masking); BatchNorm layers use running statistics instead of batch statistics. `model.train()` re-enables training behavior.

```python
import torch
import torch.nn as nn

dropout = nn.Dropout(p=0.5)
x = torch.ones(1, 10)

dropout.train()
print(dropout(x))   # roughly half the values are 0.0 (randomly)

dropout.eval()
print(dropout(x))   # tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
                    # all values pass through — no dropout
```

**How nanoGPT uses it:**
```python
model.eval()
model.to(device)
```
Called immediately after loading the checkpoint. Every subsequent `model(x)` call runs in eval mode.

**What breaks without it:** With `dropout > 0`, each call to `model.generate()` produces a different output for the same input — the generation is non-deterministic due to random dropout masking. This makes it impossible to reproduce results or debug generation quality.

---

### 2. `@torch.no_grad()`

**Definition:** A decorator (or context manager) that disables gradient tracking for all operations inside it. PyTorch normally builds a computation graph for every operation so it can compute gradients via backpropagation. During inference, this graph is unnecessary and wastes both memory and time.

```python
import torch

x = torch.tensor([2.0], requires_grad=True)

# With gradients (default):
y = x * 3
print(y.grad_fn)   # <MulBackward0> — graph exists

# Without gradients:
with torch.no_grad():
    y = x * 3
    print(y.grad_fn)   # None — no graph, less memory

# As a decorator:
@torch.no_grad()
def fast_forward(model, x):
    return model(x)
```

**Memory savings:** For a sequence of length T, the computation graph for attention requires O(T²) memory. For long generation runs, `torch.no_grad()` makes the difference between generating 1000 tokens and running out of memory.

**What breaks without it:** The computation graph grows with every generated token. For 500 tokens, the graph is 500 forward passes deep. Memory usage grows linearly until OOM.

---

### 3. Temperature Scaling

**Definition:** Temperature T divides the logits before softmax. T < 1 sharpens the distribution (higher probability to the most likely tokens). T > 1 flattens it (more uniform, more surprising outputs).

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.5, -1.0])   # raw model output

print("T=0.5 (sharp):", F.softmax(logits / 0.5, dim=-1).tolist())
# [0.843, 0.114, 0.038, 0.005]  ← top token gets 84%

print("T=1.0 (neutral):", F.softmax(logits / 1.0, dim=-1).tolist())
# [0.636, 0.234, 0.142, 0.018]  ← top token gets 64%

print("T=1.5 (flat):", F.softmax(logits / 1.5, dim=-1).tolist())
# [0.521, 0.264, 0.176, 0.040]  ← top token gets 52%
```

**As T → 0:** All probability mass concentrates on the single highest-logit token (greedy decoding). Outputs are deterministic but repetitive.

**As T → ∞:** Distribution approaches uniform. Every token is equally likely. Output is random noise.

**How nanoGPT uses it:**
```python
logits = logits[:, -1, :] / temperature
```
Divides only the last-position logits. One line, zero overhead.

**Good default:** `temperature=0.8` gives slightly sharper outputs than raw sampling. For creative text, 0.7–1.0. For factual/constrained tasks, 0.2–0.5.

---

### 4. Top-k Filtering

**Definition:** Before sampling, zero out (set to -∞) all tokens except the k most probable. This prevents sampling from the long tail of unlikely tokens (typos, rare characters, garbage).

```python
import torch

def top_k_filter(logits, k):
    # Find the k-th highest logit value
    top_k_values, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = top_k_values[:, [-1]]   # shape (batch, 1)
    # Set everything below threshold to -inf
    logits[logits < threshold] = float('-inf')
    return logits

logits = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0]])
filtered = top_k_filter(logits.clone(), k=3)
print(filtered)
# tensor([[3.0, 2.0, 1.0, -inf, -inf]])
# Only top 3 tokens remain; softmax gives them the full probability mass
```

**How nanoGPT uses it:**
```python
if top_k is not None:
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
```
`v[:, [-1]]` is the k-th largest value (using list `[-1]` to preserve the batch dimension). Everything below it gets -∞.

**Why `min(top_k, logits.size(-1))`?** If `top_k > vocab_size`, `torch.topk` would raise an error. The `min` clamps it to the vocabulary size.

**Good default:** `top_k=200`. This is large enough to allow diversity but small enough to exclude obviously bad tokens. For a 50,000 token vocabulary, top-200 covers the top 0.4%.

---

### 5. `torch.multinomial` for Sampling

**Definition:** `torch.multinomial(probs, num_samples)` draws `num_samples` token indices from a categorical distribution defined by `probs`. Each draw is independent.

```python
import torch

probs = torch.tensor([[0.7, 0.2, 0.1]])   # token 0 is most likely
# Draw 1 sample (as in language generation):
sample = torch.multinomial(probs, num_samples=1)
print(sample)   # usually tensor([[0]]), sometimes [[1]] or [[2]]

# Verify the distribution over many draws:
counts = torch.zeros(3)
for _ in range(10000):
    counts[torch.multinomial(probs, 1).item()] += 1
print(counts / 10000)   # approximately [0.70, 0.20, 0.10]
```

**What breaks with `torch.argmax` instead:** `argmax` always picks the most probable token — deterministic greedy decoding. Language models trained with cross-entropy produce well-calibrated next-token distributions. Greedy decoding ignores this calibration and often produces degenerate, repetitive text ("The the the the the...").

---

## Reading the Source File

### `sample.py` — complete walkthrough

```python
init_from = 'resume'   # 'resume' loads from out_dir/ckpt.pt
out_dir = 'out'
start = "\n"           # start character (can also be "FILE:prompt.txt")
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 1337
device = 'cuda'
dtype = 'bfloat16' if ...
compile = False        # no compile needed for inference
exec(open('configurator.py').read())
```
All these are overridable from the CLI: `python sample.py --temperature=0.5 --top_k=50 --num_samples=3`.

```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
```
Setting the seed makes generation reproducible. `torch.cuda.manual_seed` seeds the CUDA random number generator (used by `torch.multinomial` on GPU).

```python
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
```

**`map_location=device`:** Loads tensors directly onto the target device. Without this, they always load to CPU first, then must be moved with `model.to(device)`.

**`_orig_mod.` prefix:** When `torch.compile()` is used during training, it sometimes wraps the model in an internal `_orig_mod` container. The state dict then has keys like `_orig_mod.transformer.wte.weight`. The prefix-stripping loop removes this so the checkpoint is loadable with or without `compile`.

```python
if init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))
```
This path loads GPT-2 weights directly from HuggingFace — see Module 09 for how.

```python
model.eval()
model.to(device)
if compile:
    model = torch.compile(model)
```
`torch.compile` is optional for inference. It provides ~30% speedup after a one-time compilation cost of ~60 seconds. For sampling a few hundred tokens, the overhead exceeds the benefit.

```python
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
```
The two-path tokenizer: if `meta.pkl` exists (character-level or custom tokenizer), use it. Otherwise, fall back to GPT-2 BPE. The `encode`/`decode` lambdas provide a uniform interface regardless of which path was taken.

```python
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
```
`[None, ...]` adds a batch dimension: shape goes from `(T,)` to `(1, T)`.

```python
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```
`torch.no_grad()` is the outer context; `ctx` (autocast) is the inner. Both are necessary: `no_grad` disables the computation graph; autocast enables mixed-precision computation.

`y[0].tolist()` extracts the first (and only, since batch=1) sample as a Python list of ints, which `decode()` converts back to a string.

---

## Why This Design

**Why `num_samples` samples from the same prompt?**  
Language model outputs are stochastic — each sample is a different completion. Generating 10 samples and reading them all gives a better intuition for the model's distribution than reading one.

**Why not beam search?**  
Beam search keeps the k most probable sequences at each step, then returns the globally most likely sequence. For language modeling, beam search often produces repetitive, generic text ("The best way to do this is to..."). Temperature + top-k sampling produces more diverse and interesting outputs. Beam search is better for constrained generation tasks (translation, summarization).

**Why default `compile=False` in `sample.py` when `compile=True` in `train.py`?**  
Compilation takes ~60 seconds. For a training run of 600,000 steps, amortizing 60 seconds is trivial. For generating 10 text samples (a few seconds of work), paying 60 seconds of compilation is a bad trade.

---

## Running the Tests

There are no dedicated unit tests for `sample.py` because it is a script that orchestrates model loading and generation. The model's `generate()` method is tested in `test_model.py`:

```bash
pytest tests/test_model.py -v -k "generate"
```

To test the full `sample.py` script manually:
```bash
python data/shakespeare_char/prepare.py   # if not done yet
python train.py config/train_shakespeare_char.py --max_iters=100 --eval_interval=100
python sample.py --out_dir=out-shakespeare-char --device=cpu --max_new_tokens=50 --num_samples=2
```

---

## Checkpoint ✓

Run generation from a random model (before training) using the generate loop directly:

```python
import torch
from model import GPTConfig, GPT

cfg = GPTConfig(n_layer=2, n_head=2, n_embd=32, block_size=16,
                vocab_size=65, dropout=0.0, bias=False)
model = GPT(cfg)
model.eval()

# Encode "Hello" as arbitrary token IDs
prompt_ids = [1, 5, 12, 12, 15]   # arbitrary
x = torch.tensor([prompt_ids])     # (1, 5)

with torch.no_grad():
    out = model.generate(x, max_new_tokens=10, temperature=0.8, top_k=20)

print(f"Input length: {x.shape[1]}")
print(f"Output length: {out.shape[1]}")   # should be 15
print(f"Generated tokens: {out[0, 5:].tolist()}")   # the 10 new tokens
```

**Expected:** Output has shape `(1, 15)`. The 10 new tokens are in the vocabulary range [0, 65).

---

## Exercises

**1 (Easy) — Compare temperature effects:**  
Generate 5 samples at each of T=0.2, T=0.8, T=1.5 from a trained `shakespeare_char` model. Copy the best and worst outputs from each temperature. Write one sentence describing the quality vs. diversity trade-off.

**Success condition:** You have 15 samples (5 × 3 temperatures) and a written observation.

**2 (Medium) — Implement top-p (nucleus) sampling:**  
Top-p (or nucleus) sampling selects the smallest set of tokens whose cumulative probability exceeds p, then samples from that set. Add `top_p` as a parameter to `GPT.generate()` alongside `top_k`.

```python
# Hint:
probs_sorted, sorted_indices = torch.sort(probs, descending=True, dim=-1)
cumulative = torch.cumsum(probs_sorted, dim=-1)
# Remove tokens where cumulative probability exceeds top_p
sorted_indices_to_remove = cumulative > top_p
# Keep at least one token
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0
```

**Success condition:** `top_p=0.9` generates coherent text. `top_p=0.01` degenerates to near-greedy. Add a test in `test_model.py` that verifies the output shape is still `(1, 1 + max_new_tokens)`.

**3 (Hard) — Implement speculative decoding (conceptually):**  
Speculative decoding uses a small "draft" model to generate k tokens quickly, then uses the large "target" model to verify all k tokens in a single forward pass. Read [Chen et al. 2023](https://arxiv.org/abs/2302.01318) and sketch the implementation in pseudocode. Identify which parts of nanoGPT you would need to modify.

**Success condition:** A pseudocode implementation with line-by-line comments explaining the verification step. Identify that `model(idx_cond)` needs to return logits at all positions (training mode) rather than just the last position.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| Temperature & sampling | [The Curious Case of Neural Text Degeneration (Holtzman et al.)](https://arxiv.org/abs/1904.09751) | Introduces nucleus (top-p) sampling; explains why greedy/beam search produces degenerate text |
| `torch.multinomial` | [PyTorch multinomial docs](https://pytorch.org/docs/stable/generated/torch.multinomial.html) | Official reference; note the `replacement` parameter |
| torch.compile | [torch.compile tutorial](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) | Explains what compilation does and why it's fast |

---

## What's Next

[Module 09 — Pretrained Weights & Fine-tuning](module_09_pretrained_weights_and_finetuning.md): load GPT-2 weights from HuggingFace and fine-tune on Shakespeare.
