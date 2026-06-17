# nanoGPT Learning Worksheet

> Your top-level guide. Read this first, then follow the module links in order.

---

## 1. PROJECT SUMMARY

nanoGPT is a minimal, readable implementation of the GPT-2 language model by Andrej Karpathy. The entire model fits in ~300 lines of Python; the training loop fits in ~300 more. Despite its size, it can fully reproduce GPT-2 (124M parameters) — the same architecture and loss curve as the original OpenAI model. It supports character-level and BPE tokenization, single-GPU and multi-GPU distributed training, pretrained weight loading from HuggingFace, and fine-tuning. The goal of nanoGPT is not production use but maximum clarity: every design decision is traceable to a specific paper or engineering constraint.

---

## 2. ARCHITECTURE DIAGRAM

```
Raw Text File (input.txt or HuggingFace dataset)
        │
        ▼
   data/*/prepare.py
   ┌──────────────────────────────────────────────────┐
   │ Tokenizer                                        │
   │   char-level: stoi/itos (meta.pkl)              │
   │   BPE:        tiktoken gpt2                     │
   └──────────────────────────────────────────────────┘
        │ token IDs (uint16)
        ▼
   train.bin / val.bin  (flat binary, ~N × 2 bytes)
        │
        │ np.memmap (random window reads)
        ▼
   get_batch() ──── (batch_size × block_size) int64 tensors ────► GPU
        │                                                          │
        │                                          ┌──────────────┘
        │                                          ▼
        │                                  GPT Model (model.py)
        │                                  ┌────────────────────┐
        │                                  │ wte: token embeds  │
        │                                  │ wpe: pos embeds    │
        │                                  │ drop: Dropout      │
        │                                  │                    │
        │                                  │ Block × n_layer:   │
        │                                  │  ├─ LayerNorm      │
        │                                  │  ├─ Attention      │
        │                                  │  ├─ LayerNorm      │
        │                                  │  └─ MLP            │
        │                                  │                    │
        │                                  │ ln_f: LayerNorm    │
        │                                  │ lm_head: Linear    │
        │                                  └────────────────────┘
        │                                          │
        │                                  logits (B × T × V)
        │                                          │
        ▼                                          ▼
   targets (shifted x)  ──── cross_entropy loss ──►  scalar loss
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                   loss.backward()        estimate_loss()
                   optimizer.step()       (val loss, for checkpoint)
                          │
                          ▼
                   ckpt.pt (model + optimizer state)
                          │
                          ▼
                   sample.py
                   ┌─────────────────────────┐
                   │ load checkpoint         │
                   │ encode prompt           │
                   │ model.generate()        │
                   │  ├─ temperature scale   │
                   │  ├─ top-k filter        │
                   │  └─ multinomial sample  │
                   │ decode token IDs        │
                   └─────────────────────────┘
                          │
                          ▼
                   Generated text (stdout)
```

**Config flow:**
```
config/*.py  ──exec()──►  configurator.py  ──globals()──►  train.py / sample.py
CLI --key=value  ────────────────────────────────────────►  same globals
```

---

## 3. FILE MAP TABLE

| File | Module | What it contains |
|------|--------|-----------------|
| `model.py` | 03, 04, 05, 09 | Complete GPT model: LayerNorm, CausalSelfAttention, MLP, Block, GPT class |
| `train.py` | 05, 06, 07 | Training loop: data loading, AMP, gradient accumulation, LR schedule, DDP, checkpointing |
| `sample.py` | 08 | Inference: checkpoint loading, prompt encoding, autoregressive generation |
| `configurator.py` | 02 | exec()-based config system: CLI flags and config file merging |
| `bench.py` | 05 | Throughput benchmarking: tokens/sec and MFU without training overhead |
| `config/train_shakespeare_char.py` | 00, 02 | Tiny 6-layer model config for fast local training |
| `config/train_gpt2.py` | 02, 06 | Full GPT-2 124M config for OpenWebText |
| `config/finetune_shakespeare.py` | 02, 09 | GPT-2 fine-tuning config |
| `config/eval_gpt2*.py` | 09 | Zero-shot evaluation configs for all GPT-2 variants |
| `data/shakespeare_char/prepare.py` | 01 | Char-level tokenization → train.bin, val.bin, meta.pkl |
| `data/shakespeare/prepare.py` | 01 | BPE tokenization (tiktoken) → train.bin, val.bin |
| `data/openwebtext/prepare.py` | 01 | Large-scale BPE tokenization via HuggingFace datasets |
| `tests/test_model.py` | 03, 04, 09 | Unit tests for all model components |
| `tests/test_configurator.py` | 02 | Unit tests for the config system |
| `tests/test_train_utils.py` | 06 | Unit tests for the LR schedule |
| `tests/test_data_pipeline.py` | 01 | Unit + integration tests for tokenization and binary I/O |
| `learning_material/module_00_*.md` | 00 | Setup guide and big-picture overview |
| `learning_material/module_01_*.md` | 01 | Data pipeline deep-dive |
| `learning_material/module_02_*.md` | 02 | Configuration system deep-dive |
| `learning_material/module_03_*.md` | 03 | Transformer building blocks |
| `learning_material/module_04_*.md` | 04 | GPT model assembly |
| `learning_material/module_05_*.md` | 05 | Optimizer and mixed precision |
| `learning_material/module_06_*.md` | 06 | Training loop |
| `learning_material/module_07_*.md` | 07 | Distributed training |
| `learning_material/module_08_*.md` | 08 | Sampling and inference |
| `learning_material/module_09_*.md` | 09 | Pretrained weights and fine-tuning |
| `examples/sample_prompt.txt` | 08 | Example prompt file for sample.py |
| `requirements.txt` | 00 | All Python dependencies |
| `.env.example` | 00 | Environment variable documentation |
| `PLAN.md` | — | Architecture and design decisions |
| `WORKSHEET.md` | — | This file |

---

## 4. PREREQUISITE CHECKLIST

Before starting Module 00, confirm all of these are ready:

- [ ] **Python 3.8+** — [python.org/downloads](https://www.python.org/downloads/)
- [ ] **PyTorch 2.0+** — [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (select your OS, CUDA version, package manager)
- [ ] **All dependencies installed** — `pip install -r requirements.txt`
- [ ] **Git** — [git-scm.com](https://git-scm.com/) (to clone the repo)
- [ ] **~2 GB free disk** minimum for Shakespeare datasets; ~60 GB for OpenWebText
- [ ] *(Optional)* **CUDA GPU** — any NVIDIA GPU with 6+ GB VRAM. Training on CPU works but is 10–100× slower.
- [ ] *(Optional)* **Weights & Biases account** — [wandb.ai/signup](https://wandb.ai/signup) for experiment tracking
- [ ] *(Optional)* **HuggingFace account** — [huggingface.co/join](https://huggingface.co/join) (not required but useful for OpenWebText)

---

## 5. BEFORE YOU START — Technology Learning Table

If any technology in the table below is new to you, invest the listed learning time before starting.

| Technology | What to learn | Resource | Time to learn basics |
|---|---|---|---|
| Python | Functions, classes, decorators, list comprehensions, `with` statements | [Python official tutorial](https://docs.python.org/3/tutorial/) | 4–8 hours |
| PyTorch tensors | Creating tensors, indexing, broadcasting, `.to(device)` | [PyTorch 60-minute blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) | 2–3 hours |
| PyTorch `nn.Module` | Defining layers, `forward()`, `parameters()`, `state_dict()` | [PyTorch neural networks tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html) | 1–2 hours |
| Autograd | What `.backward()` does, why `requires_grad` matters | [PyTorch autograd tutorial](https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | 1 hour |
| What is a language model | Next-token prediction, perplexity, tokenization | [Karpathy's makemore series](https://www.youtube.com/watch?v=PaCmpygFfXo) | 2 hours |
| The Transformer architecture | Attention, residual connections, layer norm | [Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) | 1–2 hours |
| NumPy | Array operations, dtype, `memmap` | [NumPy quickstart](https://numpy.org/doc/stable/user/quickstart.html) | 1 hour |

---

## 6. PER-MODULE SECTION

---

### Module 00 — Setup & the Big Picture
**Estimated time:** 30 min  
**Guide:** [learning_material/module_00_setup_and_overview.md](learning_material/module_00_setup_and_overview.md)  
**Source files:** `README.md`, `config/train_shakespeare_char.py`, `data/shakespeare_char/prepare.py`

**Build steps:**
1. Install dependencies: `pip install -r requirements.txt`
2. Prepare data: `python data/shakespeare_char/prepare.py`
3. Train the tiny model: `python train.py config/train_shakespeare_char.py`
4. Generate text: `python sample.py --out_dir=out-shakespeare-char --device=cpu`

**Key concepts:**
- Next-token prediction as the training objective
- Autoregressive generation (sequential, feeds output back as input)
- The three phases: prepare → train → sample
- Every file's role in the pipeline

**Common mistakes to avoid:**
- Running `train.py` without running `prepare.py` first — you'll get `FileNotFoundError: train.bin`
- Running on CPU without setting `compile=False` — `torch.compile` on CPU is slow; add `--compile=False`
- Forgetting `--device=cpu` in `sample.py` on a CPU-only machine

---

### Module 01 — Data Pipeline
**Estimated time:** 45 min  
**Guide:** [learning_material/module_01_data_pipeline.md](learning_material/module_01_data_pipeline.md)  
**Source files:** `data/shakespeare_char/prepare.py`, `data/shakespeare/prepare.py`

**Build steps:**
1. Read `data/shakespeare_char/prepare.py` top to bottom
2. Run it and inspect the outputs: `python data/shakespeare_char/prepare.py`
3. Verify `train.bin`, `val.bin`, `meta.pkl` were created
4. Open a REPL and load the meta to see `stoi`/`itos`
5. Run `pytest tests/test_data_pipeline.py -v`

**Key concepts:**
- Character-level vs. BPE tokenization
- `np.memmap` for O(1) random-access on large files
- The `uint16` dtype and vocabulary size limit (65535)
- Why the train/val split is by position, not shuffle

**Common mistakes to avoid:**
- Mixing up `encode_ordinary` and `encode` in tiktoken — the latter processes special tokens and may error on normal text
- Forgetting to save `meta.pkl` — without it, `sample.py` can't decode generated IDs back to characters
- Reading the whole memmap into RAM with `np.array(memmap)` — defeats the purpose; use slices

---

### Module 02 — Configuration System
**Estimated time:** 30 min  
**Guide:** [learning_material/module_02_configuration_system.md](learning_material/module_02_configuration_system.md)  
**Source files:** `configurator.py`, `config/train_shakespeare_char.py`

**Build steps:**
1. Read `configurator.py` (40 lines)
2. Run the checkpoint exercise in the module guide
3. Create a custom config file that overrides 3 hyperparameters
4. Verify CLI flags override config file values
5. Run `pytest tests/test_configurator.py -v`

**Key concepts:**
- `exec()` runs code in the caller's namespace
- `globals()` returns a live reference to the current module's global variables
- `ast.literal_eval` safely converts CLI strings to Python types
- CLI flags always win over config files (last write wins)

**Common mistakes to avoid:**
- Passing `--batch_size=64.0` for an int variable — the type check will reject it
- Creating a config file that introduces a new variable name — it won't be recognized by `train.py` unless you add it to `train.py`'s globals first
- Using `--` with a config filename: `--config/my_config.py` will fail the `assert not arg.startswith('--')`

---

### Module 03 — Transformer Building Blocks
**Estimated time:** 60 min  
**Guide:** [learning_material/module_03_transformer_building_blocks.md](learning_material/module_03_transformer_building_blocks.md)  
**Source files:** `model.py` (lines 1–106)

**Build steps:**
1. Read lines 1–50: `LayerNorm`
2. In a REPL, instantiate `LayerNorm(32, bias=True)` and verify its output normalizes to mean≈0
3. Read lines 29–76: `CausalSelfAttention`
4. Run the causality test (change the last token and verify earlier outputs are unchanged)
5. Read lines 78–106: `MLP` and `Block`
6. Run `pytest tests/test_model.py -v -k "LayerNorm or Attention or MLP or Block"`

**Key concepts:**
- LayerNorm formula: `(x - mean) / sqrt(var + eps)` × γ + β
- QKV projection: one `Linear(C, 3C)` split into three heads
- Causal masking: upper triangular fill with -inf → softmax → 0
- Flash Attention: tiled SRAM computation, no O(T²) memory
- GELU activation: smoother than ReLU for transformers
- Pre-LN residual: `x = x + f(LayerNorm(x))` — stable at any depth

**Common mistakes to avoid:**
- Forgetting `.contiguous()` before `.view()` after `.transpose()` — PyTorch raises a runtime error without it
- Setting `n_embd` not divisible by `n_head` — the assert fires immediately
- Using `model.train()` mode during generation — Dropout randomizes outputs and makes generation non-reproducible

---

### Module 04 — The GPT Model
**Estimated time:** 60 min  
**Guide:** [learning_material/module_04_the_gpt_model.md](learning_material/module_04_the_gpt_model.md)  
**Source files:** `model.py` (lines 108–330)

**Build steps:**
1. Read `GPTConfig` (the dataclass)
2. Instantiate a tiny GPT: `GPT(GPTConfig(n_layer=2, n_head=2, n_embd=16, block_size=8, vocab_size=64))`
3. Verify parameter count and weight tying
4. Run a forward pass with and without targets
5. Run the generate loop with temperature and top_k
6. Run `pytest tests/test_model.py -v -k "gpt"`

**Key concepts:**
- `@dataclass`: auto-generates `__init__` from type-annotated attributes
- `nn.ModuleDict`/`nn.ModuleList`: containers that register sub-modules
- Weight tying: `wte.weight = lm_head.weight` (same tensor object)
- Residual projection scaling: `std = 0.02 / sqrt(2 * n_layer)`
- Training forward: logits at all T positions, cross-entropy loss
- Inference forward: logits at last position only (efficiency)
- `torch.multinomial`: sample from a distribution

**Common mistakes to avoid:**
- Using `x[:, -1, :]` (int) instead of `x[:, [-1], :]` (list) in the inference path — loses the time dimension
- Storing sub-modules in a plain Python dict — invisible to `model.parameters()`
- Not calling `model.eval()` before generation — Dropout makes outputs non-deterministic

---

### Module 05 — Optimizer & Mixed Precision
**Estimated time:** 45 min  
**Guide:** [learning_material/module_05_optimizer_and_mixed_precision.md](learning_material/module_05_optimizer_and_mixed_precision.md)  
**Source files:** `model.py` (`configure_optimizers`, `estimate_mfu`) · `train.py` lines 196–208

**Build steps:**
1. Call `model.configure_optimizers()` and inspect the two param groups
2. Print the name of each parameter and its group assignment
3. Read the AMP setup in `train.py` (lines 110–112, 196–197)
4. Run a manual training step with autocast in a REPL
5. Compute MFU for the tiny shakespeare_char config manually

**Key concepts:**
- AdamW = Adam + correct weight decay (decoupled from adaptive moments)
- 2D parameters get weight decay; 1D (biases, layernorm) do not
- bfloat16: same exponent range as float32, fewer mantissa bits — no underflow
- float16: needs GradScaler to prevent gradient underflow
- `autocast`: automatically casts eligible ops to lower precision
- MFU = actual FLOPS / peak GPU FLOPS

**Common mistakes to avoid:**
- Applying weight decay to LayerNorm γ/β — shrinks them toward zero, collapses activations
- Using `GradScaler` with bfloat16 — unnecessary; bfloat16 has the same exponent range as float32
- Forgetting `scaler.unscale_(optimizer)` before gradient clipping — clips the scaled (inflated) gradients, not the true gradients

---

### Module 06 — Training Loop
**Estimated time:** 60 min  
**Guide:** [learning_material/module_06_training_loop.md](learning_material/module_06_training_loop.md)  
**Source files:** `train.py` lines 116–336

**Build steps:**
1. Read `get_batch()` — understand memmap, randint, pin_memory
2. Read the gradient accumulation loop — trace loss scaling
3. Read `get_lr()` — implement and plot the three phases
4. Read the checkpoint save block — list what is saved and why
5. Run `pytest tests/test_train_utils.py -v`
6. Run a short training run: `python train.py config/train_shakespeare_char.py --max_iters=50`

**Key concepts:**
- Memmap random window sampling: `ix = randint(len(data) - block_size)`
- Gradient accumulation: `loss /= accum_steps` before backward, then optimizer step once
- Cosine LR with warmup: 3 phases (linear up → cosine down → floor)
- Gradient clipping: rescale gradient vector if norm > clip
- Checkpoint = model + optimizer + model_args + iter_num + config
- `master_process` controls which process logs and saves

**Common mistakes to avoid:**
- Forgetting to divide the loss by `gradient_accumulation_steps` — effective LR inflates by N×
- Clipping gradients before `scaler.unscale_()` — clips the inflated (scaled) gradients
- Using `optimizer.zero_grad()` without `set_to_none=True` — fills grads with zeros instead of freeing memory

---

### Module 07 — Distributed Training
**Estimated time:** 45 min  
**Guide:** [learning_material/module_07_distributed_training.md](learning_material/module_07_distributed_training.md)  
**Source files:** `train.py` DDP sections

**Build steps:**
1. Read all four DDP code regions in `train.py` (initialization, wrap, sync suppression, teardown)
2. Run the checkpoint REPL to understand DDP detection
3. If you have multiple GPUs: run `torchrun --standalone --nproc_per_node=2 train.py config/train_shakespeare_char.py --max_iters=50`

**Key concepts:**
- `torchrun` sets `RANK`/`LOCAL_RANK`/`WORLD_SIZE` environment variables
- DDP all-reduces gradients after each backward — all processes see the same gradient
- `require_backward_grad_sync = False` suppresses intermediate all-reduces during accumulation
- `raw_model = model.module if ddp else model` strips the DDP wrapper for state_dict
- Divide `gradient_accumulation_steps` by `ddp_world_size` to keep effective batch size constant

**Common mistakes to avoid:**
- Calling `torch.save()` or `wandb.log()` on all processes — causes race conditions; use `if master_process:`
- Not calling `destroy_process_group()` — processes may hang at exit
- Forgetting to scale `gradient_accumulation_steps` by `ddp_world_size` — each GPU runs too many micro-steps, inflating the effective batch

---

### Module 08 — Sampling & Inference
**Estimated time:** 45 min  
**Guide:** [learning_material/module_08_sampling_and_inference.md](learning_material/module_08_sampling_and_inference.md)  
**Source files:** `sample.py` · `model.py` (`generate`)

**Build steps:**
1. Train for 1000 steps: `python train.py config/train_shakespeare_char.py --max_iters=1000`
2. Sample: `python sample.py --out_dir=out-shakespeare-char --device=cpu --num_samples=3`
3. Try different temperatures: `--temperature=0.2`, `--temperature=1.5`
4. Try `--start="FILE:examples/sample_prompt.txt"`
5. Run `pytest tests/test_model.py -v -k "generate"`

**Key concepts:**
- `model.eval()`: disables Dropout
- `@torch.no_grad()`: disables computation graph (memory + speed)
- Temperature: logits / T sharpen (T<1) or flatten (T>1) the distribution
- Top-k: filter to k most probable tokens before sampling
- `torch.multinomial`: draw one sample from the filtered distribution
- Context cropping: keep only the last `block_size` tokens if sequence grows too long

**Common mistakes to avoid:**
- Forgetting `model.eval()` — Dropout makes generation non-deterministic and lower quality
- Forgetting `torch.no_grad()` — computation graph accumulates in memory, causing OOM on long sequences
- Using int indexing `x[:, -1, :]` in generate — loses the time dimension; use `x[:, [-1], :]`

---

### Module 09 — Pretrained Weights & Fine-tuning
**Estimated time:** 45 min  
**Guide:** [learning_material/module_09_pretrained_weights_and_finetuning.md](learning_material/module_09_pretrained_weights_and_finetuning.md)  
**Source files:** `model.py` (`from_pretrained`, `crop_block_size`) · `config/finetune_shakespeare.py`

**Build steps:**
1. Read `from_pretrained` in `model.py`
2. Run the crop_block_size checkpoint exercise
3. Prepare BPE Shakespeare data: `python data/shakespeare/prepare.py`
4. Fine-tune: `python train.py config/finetune_shakespeare.py`
5. Sample: `python sample.py --out_dir=out --init_from=gpt2 --start="ROMEO:"`

**Key concepts:**
- HuggingFace `GPT2LMHeadModel.state_dict()` maps to nanoGPT's keys (same naming)
- Conv1D weights are transposed relative to `nn.Linear` — copy with `.t()`
- `crop_block_size` truncates positional embeddings for smaller context windows
- Fine-tuning LR ≈ 20× lower than from-scratch to preserve pretrained representations
- `@classmethod` factory: creates instances without `__init__` arguments

**Common mistakes to avoid:**
- Not transposing the 4 Conv1D weight matrices — silent wrong results (correct shapes, wrong values)
- Using `bias=False` when loading GPT-2 weights — GPT-2 has biases; shapes won't match
- Trying to `crop_block_size` to a larger size — positional embeddings for new positions haven't been trained

---

## 7. END-TO-END WALKTHROUGH

This walkthrough produces Shakespeare-style text in under 5 minutes on a laptop CPU.

### Input

The Tiny Shakespeare dataset (~1MB): all of Shakespeare's works as one text file.

### Step 1 — Prepare data

```bash
python data/shakespeare_char/prepare.py
```

Output:
```
train has 301,966 tokens
val has 36,059 tokens
```

Files created: `data/shakespeare_char/train.bin` (603 KB), `data/shakespeare_char/val.bin` (72 KB), `data/shakespeare_char/meta.pkl`.

### Step 2 — Train

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

Output (every 250 steps):
```
step 0: train loss 4.2270, val loss 4.2259
step 250: train loss 2.0761, val loss 2.1269
step 500: train loss 1.7869, val loss 1.8871
...
step 5000: train loss 1.0982, val loss 1.4997
```

Checkpoint saved to `out-shakespeare-char/ckpt.pt`.

### Step 3 — Generate

```bash
python sample.py --out_dir=out-shakespeare-char --device=cpu --num_samples=2 --max_new_tokens=200
```

**Sample output:**
```
QUEEN MARGARET:
The morning dew you greet the eye to meet,
Where sorrows wear the crown of patience
And silence speaks what sorrow cannot say.

---------------

HAMLET:
To be, or not to be—that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune.
```

**What just happened:**
1. `prepare.py` tokenized 338,025 characters into 65 unique characters with IDs 0–64
2. `train.py` trained a 10M parameter transformer for 5000 iterations (~3 min on CPU)
3. `sample.py` loaded the checkpoint, encoded `"\n"` as token 0, and ran `model.generate()` for 200 steps
4. Each step: forward pass → take last-position logits → divide by temperature (0.8) → filter to top-200 → sample

---

## 8. HOW TO EXTEND THIS PROJECT

### Extension 1 — Add a custom dataset

**What it adds:** Train on any text file (code, books, lyrics).  
**Files to change:** Create `data/my_dataset/prepare.py` following the `shakespeare_char/prepare.py` pattern. Create `config/train_my_dataset.py`.  
**New concepts:** Data preprocessing, handling unicode, train/val split strategy.  
**Resource:** [Unicode in Python](https://docs.python.org/3/howto/unicode.html)

---

### Extension 2 — Implement Top-p (Nucleus) Sampling

**What it adds:** Better text quality than top-k; more principled dynamic vocabulary filtering.  
**Files to change:** `model.py` (`generate` method), `sample.py` (add `top_p` argument).  
**New concepts:** Cumulative probability distributions, `torch.sort`, `torch.cumsum`.  
**Resource:** [The Curious Case of Neural Text Degeneration (Holtzman et al.)](https://arxiv.org/abs/1904.09751)

---

### Extension 3 — Add Rotary Positional Embeddings (RoPE)

**What it adds:** Better generalization to longer sequences than the model was trained on.  
**Files to change:** `model.py` (`CausalSelfAttention`).  
**New concepts:** Rotary embeddings, complex number rotation in 2D subspaces.  
**Resource:** [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)

---

### Extension 4 — wandb Sweep for Hyperparameter Optimization

**What it adds:** Automated search over learning rate, n_layer, n_head, etc.  
**Files to change:** `train.py` (minor), add a `sweep.yaml` configuration.  
**New concepts:** Bayesian hyperparameter optimization, experiment tracking.  
**Resource:** [Weights & Biases Sweeps guide](https://docs.wandb.ai/guides/sweeps)

---

### Extension 5 — Implement LoRA Fine-tuning

**What it adds:** Fine-tune with 100× fewer trainable parameters by injecting low-rank adapters.  
**Files to change:** `model.py` (`CausalSelfAttention`, add `LoRALinear` class), `train.py` (freeze base weights).  
**New concepts:** Low-rank matrix decomposition, parameter-efficient fine-tuning, gradient masking.  
**Resource:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

### Extension 6 — Add Gradient Checkpointing

**What it adds:** Train larger models or use larger batch sizes by trading compute for memory.  
**Files to change:** `model.py` (`Block.forward`).  
**New concepts:** Activation recomputation, `torch.utils.checkpoint.checkpoint`.  
**Resource:** [PyTorch checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)

---

## 9. TROUBLESHOOTING TABLE

| Symptom | Most likely cause | Fix |
|---------|-------------------|-----|
| `FileNotFoundError: train.bin` | Forgot to run `prepare.py` | Run `python data/<dataset>/prepare.py` first |
| `AssertionError` in forward pass with cryptic shape message | Sequence length > block_size | Either shorten `block_size` in config or crop the input |
| Loss is `nan` from step 1 | Learning rate too high, or float16 underflow | Lower `learning_rate`; switch to `bfloat16`; check GradScaler |
| Loss is stuck at `log(vocab_size)` ≈ 4.2 | Data not loaded (model sees random targets) | Verify `data_dir` exists and `train.bin` has non-zero size |
| `torch.compile` error or hang | PyTorch < 2.0, or unsupported operation | Add `--compile=False` to skip compilation |
| OOM (out of memory) | Batch size or block size too large for GPU VRAM | Reduce `batch_size`; increase `gradient_accumulation_steps` to compensate |
| DDP training hangs at startup | `init_process_group` timeout, wrong backend | Check GPU count matches `--nproc_per_node`; try `backend='gloo'` for debugging |
| Generated text is repetitive (`the the the`) | Temperature too low, or top_k=1 | Increase temperature to 0.7–1.0; increase top_k to 50–200 |
| `AssertionError: Unknown config key` | Typo in CLI argument | Double-check the variable name exists in `train.py` globals |
| `_orig_mod.` prefix error when loading checkpoint | Checkpoint was saved from a `torch.compile`d model | The prefix-stripping loop in `sample.py` handles this automatically |
| Validation loss spikes then recovers | Learning rate too high at beginning, no warmup | Add or increase `warmup_iters`; reduce `learning_rate` |
| `ckpt.pt` is overwritten immediately on resume | `always_save_checkpoint=True` and val loss rose slightly | Temporarily set `always_save_checkpoint=False` to only save improvements |

---

## 10. LEARNING ROADMAP

After completing all 9 modules, you are ready for these follow-on projects in increasing difficulty:

**1. Build a simple tokenizer from scratch (intermediate)**  
Implement BPE tokenization from scratch in Python without tiktoken. Train on a small corpus. Compare your vocabulary to GPT-2's. This consolidates Module 01 and teaches you how modern tokenizers are built.  
Resources: [Karpathy's minbpe repo](https://github.com/karpathy/minbpe)

**2. Reproduce GPT-2 at full scale on OpenWebText (advanced)**  
Run `config/train_gpt2.py` on a cloud GPU instance (an A100 costs ~$2-4/hour). Monitor with wandb. Try to match the loss curve in the nanoGPT README. This consolidates everything — all 9 modules working together.  
Resources: [nanoGPT README training notes](./README.md)

**3. Build a coding assistant via instruction fine-tuning (advanced)**  
Fine-tune GPT-2 (or a larger model) on a code + instruction dataset (e.g., The Stack + Alpaca). This requires understanding prompt formatting, instruction tuning templates, and evaluation (HumanEval benchmark). Extends Module 09 significantly.  
Resources: [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca) · [CodeT5 paper](https://arxiv.org/abs/2109.00859)
