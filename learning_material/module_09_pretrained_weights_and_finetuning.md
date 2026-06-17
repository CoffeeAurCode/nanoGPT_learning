# Module 09 — Pretrained Weights & Fine-tuning

**Source files:** `model.py` (`from_pretrained`, `crop_block_size`) · `config/finetune_shakespeare.py`  
**Estimated time:** 45 minutes

---

## What You Are Building

Loading GPT-2's pretrained weights into nanoGPT's architecture. The OpenAI checkpoint uses a different weight layout than nanoGPT (Conv1D vs. Linear), which requires weight transposition. You will also understand `crop_block_size` (model surgery) and the fine-tuning hyperparameter strategy.

---

## Concept Deep-Dives

### 1. HuggingFace `GPT2LMHeadModel` and State Dicts

**Definition:** A state dict is an `OrderedDict` mapping parameter names (strings) to tensors. PyTorch models can save and load state dicts independently of their architecture.

```python
import torch
import torch.nn as nn

model = nn.Linear(4, 4)
sd = model.state_dict()
print(dict(sd))
# {'weight': tensor([[...]]), 'bias': tensor([...])}

# Load into a fresh model:
model2 = nn.Linear(4, 4)
model2.load_state_dict(sd)
```

**How nanoGPT uses it:**

```python
from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
sd_hf = model_hf.state_dict()

# nanoGPT's state dict keys (sample):
# transformer.wte.weight
# transformer.wpe.weight
# transformer.h.0.ln_1.weight
# transformer.h.0.attn.c_attn.weight
# ...

# HuggingFace's state dict keys (sample):
# transformer.wte.weight
# transformer.wpe.weight
# transformer.h.0.ln_1.weight
# transformer.h.0.attn.c_attn.weight    ← same name!
# ...
```

The key names are almost identical because nanoGPT was deliberately written to mirror the HuggingFace naming convention (which itself mirrors OpenAI's TensorFlow checkpoint). The only required transformation is the weight transposition.

---

### 2. Conv1D vs. Linear: Why Transposition Is Needed

**Definition:** OpenAI's original GPT-2 implementation used `Conv1D` instead of `nn.Linear`. A `Conv1D(in, out)` has a weight matrix of shape `(in, out)`, while `nn.Linear(in, out)` has shape `(out, in)`. They compute the same operation (a linear transformation), but their weight tensors are transposed relative to each other.

```python
import torch
import torch.nn as nn

in_f, out_f = 4, 3

# nn.Linear: weight is (out, in)
linear = nn.Linear(in_f, out_f, bias=False)
print(linear.weight.shape)   # torch.Size([3, 4]) ← (out, in)

# What Conv1D does (conceptually):
# conv1d_weight.shape == (in, out) = (4, 3)
# forward: x @ conv1d_weight  (works because (batch, in) @ (in, out) = (batch, out))

# To copy Conv1D weights into nn.Linear:
conv1d_weight = torch.randn(in_f, out_f)   # (4, 3)
linear.weight.data.copy_(conv1d_weight.t()) # transpose: (3, 4)
```

**How nanoGPT uses it:**

```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']

for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k].t())   # ← transpose
    else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k])
```

`sd_hf[k].shape[::-1] == sd[k].shape` verifies that the shapes are transposed versions of each other before copying. If they're not, the model architectures have diverged.

**What breaks without transposition:** The weight matrices are numerically inverted. The model output is garbage (random noise). There is no error — the shapes still match — making this a silent correctness bug.

---

### 3. `@classmethod` as a Factory Constructor

**Definition:** A `@classmethod` is a method that receives the class (`cls`) as its first argument instead of an instance (`self`). It can create and return new instances — a "factory constructor" pattern.

```python
class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    @classmethod
    def from_description(cls, description):
        # Parse "Cat meows" → Animal("Cat", "meows")
        parts = description.split()
        return cls(parts[0], parts[1])

cat = Animal.from_description("Cat meows")
print(cat.name, cat.sound)   # Cat meows
```

**How nanoGPT uses it:**

```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        ...
    }[model_type]
    config = GPTConfig(**config_args)
    model = cls(config)                    # creates a new GPT instance
    ...
    return model
```

`cls(config)` is the same as `GPT(config)` — but because it uses `cls`, subclasses of `GPT` would get their own type back, not a `GPT`. This is good factory pattern practice.

---

### 4. `crop_block_size`: Model Surgery

**Definition:** Shrinking a trained model's context window by truncating its positional embedding table and (if present) its attention causal masks.

```python
def crop_block_size(self, block_size):
    assert block_size <= self.config.block_size
    self.config.block_size = block_size
    # Truncate the positional embedding table
    self.transformer.wpe.weight = nn.Parameter(
        self.transformer.wpe.weight[:block_size]
    )
    # Truncate the causal mask buffers (only present in non-Flash-Attention path)
    for block in self.transformer.h:
        if hasattr(block.attn, 'bias'):
            block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]
```

**When you use this:** You download GPT-2 (trained with `block_size=1024`) but want to fine-tune on a dataset where sequences are at most 256 tokens. The truncated positional embeddings keep the first 256 position vectors; positions 256–1023 are discarded. This saves memory and speeds up training.

```python
# In train.py:
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
```

**What breaks if you don't crop:** No error — the model works. But: (1) position embeddings for positions 256–1023 are allocated but never used, wasting memory; (2) if you feed a sequence longer than your actual data allows, you'd get the old GPT-2 embeddings for those positions, which may be nonsensical for your fine-tuning domain.

**What breaks if you try to expand (crop to a larger size):** The `assert block_size <= self.config.block_size` prevents this. Expanding would require learning new positional embeddings — you can't just truncate; you'd need to add rows and train them.

---

### 5. Fine-tuning Hyperparameter Strategy

```python
# config/finetune_shakespeare.py

init_from = 'gpt2'       # start from pretrained weights
max_iters = 20            # very few iterations (GPT-2 already has language knowledge)
decay_lr = False          # no LR decay for such a short run
learning_rate = 3e-5      # ~20x lower than from-scratch LR (6e-4)
dropout = 0.1             # small amount of regularization
batch_size = 1
gradient_accumulation_steps = 32
```

**Why lower LR?** GPT-2's weights encode useful representations of language. A high learning rate would overwrite them too aggressively. Fine-tuning with `lr ≈ 3e-5` nudges the model toward the target domain without destroying general language understanding.

**Why fewer iterations?** The Shakespeare corpus is ~1M tokens. At batch_size=32 and block_size=1024, each effective step covers 32,768 tokens. 20 steps × 32,768 = 655,360 tokens — nearly the entire corpus once. For a 1M token dataset, 1–3 epochs is enough.

**Why `decay_lr=False`?** With only 20 iterations, a warmup + cosine schedule would never reach a useful learning rate. Constant LR is simpler and works fine for such short runs.

---

## Reading the Source File

### `model.py` — `from_pretrained` walkthrough

```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
    assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
    override_args = override_args or {}
    assert all(k == 'dropout' for k in override_args)
```
Only `dropout` can be overridden when loading pretrained weights — changing `n_layer` or `n_embd` would make the architectures incompatible.

```python
    config_args = {
        'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),
        'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024),
        'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280),
        'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600),
    }[model_type]
    config_args['vocab_size'] = 50257    # must match OpenAI's vocab
    config_args['block_size'] = 1024     # must match OpenAI's block_size
    config_args['bias'] = True           # must match — GPT-2 has biases
```
These are hardcoded to match OpenAI's exact architecture. The default `bias=False` in `GPTConfig` would produce mismatched state dict shapes if used here.

```python
    config = GPTConfig(**config_args)
    model = cls(config)
    sd = model.state_dict()
    sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')]
```
`'.attn.bias'` is the causal mask buffer (registered with `register_buffer`). It's in the state dict but not a parameter — exclude it from the key-matching check.

```python
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    sd_keys_hf = [k for k in sd_hf.keys()
                  if not k.endswith('.attn.masked_bias')
                  and not k.endswith('.attn.bias')]
```
HuggingFace's GPT-2 has both `.attn.masked_bias` (a buffer used for masking in the manual attention path) and `.attn.bias` (another buffer). Neither is a trained weight; both are excluded.

```python
    assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: ..."
```
If this assertion fails, the architecture or the key filtering logic is wrong. This is a fast sanity check before the slower weight copy loop.

---

## Why This Design

**Why not just `model.load_state_dict(sd_hf)`?**  
The key names don't match (HuggingFace uses `h.0.attn.c_attn.weight`, nanoGPT uses the same — actually they do match). But the weight shapes don't match for the 4 Conv1D layers. Direct loading would raise a `RuntimeError`.

**Why `with torch.no_grad(): sd[k].copy_(sd_hf[k].t())`?**  
`copy_` is an in-place operation. Inside `torch.no_grad()`, the copy does not create autograd graph nodes — it's purely a data transfer. Without `no_grad`, every copy would register as an operation requiring a gradient, wasting memory.

**Why does nanoGPT use `vocab_size=50304` for from-scratch but `50257` for pretrained?**  
From scratch: pad to multiple of 64 for efficiency. From pretrained: must exactly match OpenAI's vocabulary. Changing vocab_size would require a different lm_head shape, making the checkpoint incompatible.

---

## Running the Tests

```bash
# Unit tests (no network):
pytest tests/test_model.py -v -k "crop"

# Integration test (downloads ~500 MB GPT-2 weights):
INTEGRATION=1 pytest tests/test_model.py -v -k "pretrained"
```

To run a full fine-tuning experiment:
```bash
python data/shakespeare/prepare.py   # BPE tokenization (requires tiktoken)
python train.py config/finetune_shakespeare.py
python sample.py --out_dir=out --start="ROMEO:" --num_samples=3
```

---

## Checkpoint ✓

```python
import torch
from model import GPTConfig, GPT

cfg = GPTConfig(block_size=16, vocab_size=64, n_layer=2, n_head=2,
                n_embd=16, dropout=0.0, bias=True)
model = GPT(cfg)

# Crop block_size from 16 to 8
model.crop_block_size(8)
assert model.config.block_size == 8
assert model.transformer.wpe.weight.shape[0] == 8

# Verify we can still run a forward pass with the cropped model
idx = torch.randint(0, 64, (1, 8))
logits, _ = model(idx)
print(logits.shape)   # torch.Size([1, 1, 64])
print("crop_block_size works correctly")
```

---

## Exercises

**1 (Easy) — Compare parameter counts across GPT-2 variants:**  
For each of `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`, compute the parameter count using `model.get_num_params()` without actually downloading the weights (just use `GPTConfig` with the right values). Verify against the documented sizes: 124M, 350M, 774M, 1558M.

**Success condition:** Your computed counts match the paper numbers to within 1%.

**2 (Medium) — Write `from_pretrained` without HuggingFace:**  
OpenAI's original GPT-2 weights are available as TensorFlow checkpoints. Modify `from_pretrained` to load directly from the OpenAI TF format using `tensorflow` or the raw `.npz` files. Hint: the weight matrices are in `tf.nn.Variable` format and also need to be transposed.

**Success condition:** A `from_pretrained_tf` classmethod that loads `gpt2` weights without importing `transformers`.

**3 (Hard) — Implement LoRA fine-tuning:**  
LoRA (Low-Rank Adaptation) freezes the pretrained weights and adds a small pair of trainable matrices (A and B, with rank r ≪ n_embd) to each attention projection. The effective weight is `W + AB`. Modify `CausalSelfAttention` to support LoRA. Compare the trainable parameter count and fine-tuning loss convergence to full fine-tuning.

**Success condition:** LoRA fine-tuning with rank=4 converges within 10% of full fine-tuning's final loss, using 100× fewer trainable parameters. Read [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al.)](https://arxiv.org/abs/2106.09685).

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| HuggingFace Transformers | [GPT2 model docs](https://huggingface.co/docs/transformers/model_doc/gpt2) | State dict layout and all configuration options |
| Conv1D vs Linear | [OpenAI GPT-2 source (TF)](https://github.com/openai/gpt-2/blob/master/src/model.py) | The original TF code where Conv1D was used; compare to nanoGPT's Linear |
| Fine-tuning strategies | [A survey on instruction tuning (Zhang et al.)](https://arxiv.org/abs/2308.10792) | Overview of when and how to fine-tune; puts nanoGPT's approach in context |
| LoRA | [LoRA: Low-Rank Adaptation (Hu et al.)](https://arxiv.org/abs/2106.09685) | The paper behind most modern fine-tuning — relevant for Exercise 3 |
