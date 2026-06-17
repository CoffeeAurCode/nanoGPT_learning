# PLAN.md — nanoGPT Learning Guide

---

## 1. PROJECT OVERVIEW

nanoGPT is Andrej Karpathy's minimal, readable implementation of the GPT-2 language model in pure PyTorch. It solves the problem of the "GPT implementation gap": every open-source GPT codebase is either a toy tutorial that doesn't scale, or a production system (HuggingFace Transformers, Megatron-LM) so wrapped in abstraction layers that the core algorithm is invisible. nanoGPT fits in ~300 lines of model code and ~300 lines of training code, yet can fully reproduce GPT-2 (124M parameters) on OpenWebText — the same dataset and loss curve as the original OpenAI paper. The entire system: tokenizes raw text into a flat binary file, reads it via memory-mapped arrays, builds a transformer decoder with causal self-attention, trains it with AdamW + mixed-precision + gradient accumulation + optional distributed data parallelism, and generates text autoregressively. It also loads pretrained GPT-2 weights from HuggingFace for fine-tuning.

---

## 2. TECHNOLOGY STACK TABLE

| Layer | Tool / Library | Why this choice over alternatives |
|---|---|---|
| Deep learning framework | **PyTorch 2.x** | Native `torch.compile`, Flash Attention via `scaled_dot_product_attention`, and the most active research ecosystem. JAX would be faster on TPUs but has a steeper learning curve and worse Windows support. TensorFlow is legacy at this point. |
| Tokenizer | **tiktoken** (OpenAI) | Byte-pair encoding (BPE) that exactly matches GPT-2's vocabulary — required to load pretrained weights. HuggingFace `tokenizers` also supports GPT-2 BPE but tiktoken is ~5× faster and has a simpler API for this use case. |
| Dataset loading (large scale) | **HuggingFace `datasets`** | The only practical way to stream and multiprocess-tokenize OpenWebText (8M documents). A manual downloader would be hundreds of lines and much slower. |
| Pretrained weight loading | **HuggingFace `transformers`** | GPT-2 weights live on the HuggingFace Hub; `GPT2LMHeadModel` provides the only convenient public interface to them. |
| Numerical arrays / binary I/O | **NumPy** | The `np.memmap` class enables memory-mapped file I/O — reading a 17 GB token file without loading it into RAM. PyTorch has no built-in memmap equivalent. |
| Experiment tracking | **Weights & Biases (wandb)** | Optional but zero-config remote dashboard for loss curves. TensorBoard is local-only; MLflow requires a server; wandb is one `pip install` + `wandb.init()`. |
| Distributed training | **PyTorch DDP via `torchrun`** | Native to PyTorch, no extra install. Horovod and DeepSpeed are heavier alternatives needed only for very large models. |
| Configuration | **Custom `configurator.py`** | A 40-line `exec()`-based system that merges a config file and `--key=value` CLI flags directly into the caller's global namespace. Zero boilerplate vs. argparse, Hydra, or OmegaConf. Trade-off: non-standard and slightly magical (see Module 2). |

---

## 3. MODULE BREAKDOWN

### Module 0 — Setup & the Big Picture
**Source files:** `README.md`, `config/train_shakespeare_char.py`, `data/shakespeare_char/prepare.py`  
**Completion time:** ~30 minutes  
**What you will be able to do:** Install all dependencies, run the Shakespeare character-level model end-to-end, and explain what a language model is in one sentence.  
**Concepts taught:**
- What a language model is (next-token prediction)
- The train/val split and why it matters
- How character-level vs. BPE tokenization differs
- How to read a nanoGPT config file

---

### Module 1 — Data Pipeline
**Source files:** `data/shakespeare/prepare.py`, `data/shakespeare_char/prepare.py`, `data/openwebtext/prepare.py`  
**Completion time:** ~45 minutes  
**What you will be able to do:** Write a data preparation script that downloads text, tokenizes it, and saves it as a flat binary file readable via memmap.  
**Concepts taught:**
- Byte-pair encoding (BPE) with tiktoken
- Character-level tokenization (manual `stoi`/`itos` maps)
- `np.memmap` and why random-access binary beats in-memory arrays for large datasets
- The `uint16` dtype choice and its limits (max vocab 65535)
- Train/val split by position (why shuffling would cause data leakage here)

---

### Module 2 — Configuration System
**Source files:** `configurator.py`, `config/train_shakespeare_char.py`, `config/finetune_shakespeare.py`  
**Completion time:** ~30 minutes  
**What you will be able to do:** Explain the `exec()`-based config pattern, write a new config file, and override any hyperparameter from the CLI.  
**Concepts taught:**
- Python's `exec()` and `globals()` — how code can modify its own namespace
- `ast.literal_eval` for safe type-coercing of CLI strings
- The trade-off: simplicity vs. security/readability
- Alternative approaches: argparse, dataclasses, Hydra

---

### Module 3 — Transformer Building Blocks
**Source files:** `model.py` (lines 1–106: `LayerNorm`, `CausalSelfAttention`, `MLP`, `Block`)  
**Completion time:** ~60 minutes  
**What you will be able to do:** Implement each layer from scratch: a bias-optional LayerNorm, multi-head causal self-attention with Flash Attention fallback, a position-wise MLP with GELU, and a residual transformer block.  
**Concepts taught:**
- Layer Normalization: definition, formula, why bias is optional
- Scaled dot-product attention: the QKV projection, head splitting, causal mask
- Flash Attention: what it optimizes (memory bandwidth, not FLOPs) and why `scaled_dot_product_attention` is a drop-in
- GELU activation: why it outperforms ReLU for transformers
- Residual connections: why `x = x + f(x)` prevents vanishing gradients
- Pre-LN vs. Post-LN architecture and why GPT-2 uses Pre-LN

---

### Module 4 — The GPT Model
**Source files:** `model.py` (lines 108–330: `GPTConfig`, `GPT`)  
**Completion time:** ~60 minutes  
**What you will be able to do:** Build the full `GPT` class: token + positional embeddings, stacked transformer blocks, final LayerNorm, language model head, weight initialization, and autoregressive generation.  
**Concepts taught:**
- `@dataclass` for configuration objects
- Token embeddings vs. positional embeddings and why both are learned
- Weight tying: sharing `wte` and `lm_head` weights (reduces parameters, improves perplexity)
- The `_init_weights` strategy: normal(0, 0.02) for linear/embedding, zeros for bias — and why residual projections are scaled by `1/sqrt(2*n_layer)`
- `nn.ModuleDict` / `nn.ModuleList` and why plain Python dicts don't work here
- Cross-entropy loss on vocabulary logits
- Autoregressive generation: temperature scaling, top-k filtering, multinomial sampling

---

### Module 5 — Optimizer & Mixed Precision
**Source files:** `model.py` (`configure_optimizers`, `estimate_mfu`), `train.py` (lines 196–208)  
**Completion time:** ~45 minutes  
**What you will be able to do:** Explain why AdamW decays some parameters but not others, set up mixed-precision training with `torch.amp`, and interpret the MFU metric.  
**Concepts taught:**
- AdamW vs. Adam: what weight decay actually does and why biases/LayerNorm params should not be decayed
- Fused AdamW: a CUDA kernel that fuses the optimizer step (available in PyTorch 2.x on CUDA)
- `torch.amp.autocast`: how automatic mixed precision works (bfloat16 vs. float16)
- `GradScaler`: why float16 needs loss scaling but bfloat16 does not
- Model FLOPs Utilization (MFU): how to estimate GPU efficiency from model size and timing

---

### Module 6 — The Training Loop
**Source files:** `train.py` (lines 116–336)  
**Completion time:** ~60 minutes  
**What you will be able to do:** Implement the full training loop: batch sampling, gradient accumulation, gradient clipping, cosine learning rate schedule with warmup, checkpointing, and loss estimation.  
**Concepts taught:**
- `np.memmap`-based data loader: why it's re-created each batch (memory leak prevention)
- Gradient accumulation: simulating batch size `B * accum_steps` with GPU memory for `B`
- Cosine decay with linear warmup: the three-phase LR schedule (warmup → cosine decay → floor)
- `grad_clip`: why gradient norm clipping stabilizes transformer training
- `torch.no_grad()` for eval: what happens to the computation graph if omitted
- Checkpointing: what to save (`model`, `optimizer`, `iter_num`, `best_val_loss`, `config`) and why
- `master_process` pattern for multi-GPU: only rank 0 should log and save

---

### Module 7 — Distributed Training
**Source files:** `train.py` (lines 82–101, 211–212, 293–299, 335–336)  
**Completion time:** ~45 minutes  
**What you will be able to do:** Explain how DistributedDataParallel works and identify the four places in `train.py` where DDP changes the code.  
**Concepts taught:**
- `torchrun` and environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`)
- `init_process_group` / `destroy_process_group` lifecycle
- `DistributedDataParallel`: gradient all-reduce, why gradients sync only at the last micro-step
- `model.require_backward_grad_sync`: the `no_sync()` shortcut
- Seed offset per rank: why each GPU needs a different random seed for data sampling

---

### Module 8 — Sampling & Inference
**Source files:** `sample.py`, `model.py` (`generate`)  
**Completion time:** ~45 minutes  
**What you will be able to do:** Load a trained checkpoint (or a pretrained GPT-2), encode a prompt, run autoregressive generation, and explain every hyperparameter in `sample.py`.  
**Concepts taught:**
- `model.eval()` mode: what it changes (Dropout, BatchNorm) and why it matters
- The `@torch.no_grad()` decorator: disabling the autograd graph for inference
- Temperature scaling: how dividing logits by T < 1 sharpens, T > 1 flattens the distribution
- Top-k filtering: masking all but the k highest-probability tokens
- `torch.multinomial`: sampling without replacement from a probability distribution
- The `FILE:` prompt prefix pattern for reading prompts from disk

---

### Module 9 — Pretrained Weights & Fine-tuning
**Source files:** `model.py` (`from_pretrained`, `crop_block_size`), `config/finetune_shakespeare.py`  
**Completion time:** ~45 minutes  
**What you will be able to do:** Load any GPT-2 variant's weights into nanoGPT, explain the Conv1D→Linear transposition, and run a fine-tuning experiment on Shakespeare.  
**Concepts taught:**
- HuggingFace `GPT2LMHeadModel` state dict layout vs. nanoGPT's layout
- Why OpenAI used Conv1D (and why it's equivalent to Linear with a transposed weight)
- `model surgery`: `crop_block_size` — truncating positional embeddings after loading
- Fine-tuning strategy: higher dropout, lower LR, fewer iterations, `init_from='gpt2'`
- `@classmethod` pattern: factory constructors in Python

---

## 4. FINAL FILE STRUCTURE

```
nanoGPT_learning/
│
├── model.py                    # Entire GPT model: LayerNorm, Attention, MLP, Block, GPT class
├── train.py                    # Training loop: data loading, optimizer, LR schedule, DDP, checkpointing
├── sample.py                   # Inference: loads checkpoint or GPT-2, generates text autoregressively
├── configurator.py             # exec()-based config override: merges config files and --key=value CLI args
├── bench.py                    # Throughput benchmark: measures tokens/sec and MFU without training overhead
│
├── config/
│   ├── train_gpt2.py           # Hyperparameters to reproduce GPT-2 124M on OpenWebText
│   ├── train_shakespeare_char.py  # Tiny 6-layer char-level model for fast local experimentation
│   ├── finetune_shakespeare.py    # Fine-tune pretrained GPT-2 on Shakespeare (BPE tokens)
│   ├── eval_gpt2.py            # Evaluate base GPT-2 (124M) zero-shot
│   ├── eval_gpt2_medium.py     # Evaluate GPT-2 medium (350M) zero-shot
│   ├── eval_gpt2_large.py      # Evaluate GPT-2 large (774M) zero-shot
│   └── eval_gpt2_xl.py         # Evaluate GPT-2 XL (1558M) zero-shot
│
├── data/
│   ├── shakespeare/
│   │   ├── prepare.py          # Downloads Tiny Shakespeare, tokenizes with tiktoken BPE → train.bin, val.bin
│   │   └── readme.md           # Dataset stats and notes
│   ├── shakespeare_char/
│   │   ├── prepare.py          # Downloads Tiny Shakespeare, char-level tokenize → train.bin, val.bin, meta.pkl
│   │   └── readme.md           # Dataset stats and notes
│   └── openwebtext/
│       ├── prepare.py          # Tokenizes OpenWebText (~8M docs) via HuggingFace datasets → 17GB train.bin
│       └── readme.md           # Dataset stats and notes
│
├── assets/
│   ├── nanogpt.jpg             # Architecture diagram used in README
│   └── gpt2_124M_loss.png      # Training loss curve for GPT-2 124M reproduction
│
├── scaling_laws.ipynb          # Notebook: Chinchilla scaling law analysis
├── transformer_sizing.ipynb    # Notebook: parameter count and FLOP estimation for various model sizes
│
├── README.md                   # Project overview, quick-start commands, and results table
└── prompt.md                   # This learning prompt
```

---

## 5. KEY DESIGN DECISIONS TABLE

| Decision | Why | What you would do differently without this constraint |
|---|---|---|
| **Single-file model** (`model.py` is self-contained) | Maximizes readability — you can read the entire architecture top-to-bottom without jumping files. Karpathy's explicit goal. | Split into `layers/attention.py`, `layers/mlp.py`, `layers/norm.py` for a production codebase to enable per-layer testing and reuse. |
| **`exec()`-based configuration** | Eliminates argparse boilerplate and the awkward `config.learning_rate` prefix on every variable. Config files are plain Python, so they can have comments and arithmetic. | Use Hydra or simple dataclasses + argparse. The `exec()` pattern is a security risk if config files come from untrusted sources. |
| **Raw binary memmap for data** (uint16 flat array) | Zero-copy random-access reads into a 17 GB file without loading it into RAM. Achieves near-disk-bandwidth data loading. | Use HuggingFace `datasets` with Arrow format for richer metadata, better shuffling, and built-in multiprocessing — but at the cost of a larger on-disk footprint and less transparency. |
| **Gradient accumulation instead of large batches** | Simulates the effective batch size needed for stable GPT-2 training (B=480 in the paper) on GPUs with limited VRAM. | Simply increase `batch_size` if you have a GPU with enough memory (e.g., 80 GB A100). |
| **Flash Attention with manual fallback** | Flash Attention (`scaled_dot_product_attention`) uses tiled SRAM computation to avoid materializing the full O(T²) attention matrix, saving ~10× memory for long sequences. | Always use the manual implementation for strict reproducibility; always use Flash Attention if you only target PyTorch ≥ 2.0. |
| **Weight tying** (`wte.weight = lm_head.weight`) | Reduces 38M parameters (vocab_size × n_embd) to zero extra cost. Empirically improves perplexity per Press & Wolf (2017). | Use separate embedding and output projection matrices for maximum flexibility (e.g., if you want to use different embedding dimensions). |
| **Selective weight decay** (2D params only) | Weight decay on bias terms and LayerNorm scale/shift parameters hurts convergence — these params have no "magnitude to regularize." The 2D heuristic (all weight matrices, all embeddings) correctly targets matmul weights. | Use a per-parameter-name allowlist instead of dimensionality heuristic; more explicit but more brittle as model architecture changes. |
| **Pre-LN transformer blocks** (`x = x + attn(ln(x))`) | Post-LN (original Transformer, Vaswani 2017) requires careful warmup or it diverges. Pre-LN trains stably with larger learning rates. GPT-2 used Pre-LN. | Use Post-LN with a very conservative LR warmup if strict architectural equivalence to the original Transformer paper is needed. |
| **DDP via `torchrun` env variables** | The simplest supported DDP launch method. `RANK`/`LOCAL_RANK`/`WORLD_SIZE` are set automatically. | Use `torch.multiprocessing.spawn` for programmatic launch from a Python script, which avoids the external `torchrun` command. |
| **No dedicated `DataLoader` class** | `get_batch()` is 10 lines that read from a memmap. A `DataLoader` object would add abstraction with no benefit when data is already pre-tokenized and randomly accessible. | Use `torch.utils.data.DataLoader` with a custom `Dataset` for variable-length sequences, streaming data, or on-the-fly augmentation. |

---

## 6. OPEN QUESTIONS FOR THE USER

Answer these before building starts — each affects which path through the code you will take first.

1. **Hardware**: Do you have a CUDA GPU available? If not (CPU-only or Apple Silicon MPS), you will need to add `device='cpu'` and `compile=False` to every config. The benchmarking numbers in the learning material assume CUDA.

2. **Starting dataset**: Which dataset do you want to run first?
   - `shakespeare_char` — downloads in seconds, trains in ~3 minutes on a single GPU, visible results fast. Best for first learning pass.
   - `shakespeare` (BPE) — requires tiktoken, trains in ~15 minutes, produces better quality text.
   - `openwebtext` — requires ~54 GB disk and ~3–5 days of A100 time to reproduce GPT-2. For reference only unless you have cloud compute.

3. **Train from scratch or fine-tune?** Fine-tuning GPT-2 on Shakespeare (`config/finetune_shakespeare.py`) takes ~10 minutes and produces impressive results immediately, but skips the training-from-scratch experience. Recommend: run `shakespeare_char` from scratch first, then fine-tune GPT-2.

4. **Distributed training**: Do you have access to multiple GPUs? If not, the DDP sections (Module 7) can be read-only — the code is clearly marked and skippable without losing any other concept.

5. **Experiment tracking**: Do you have a Weights & Biases account? It is entirely optional (`wandb_log = False` by default), but useful if you want to visualize loss curves remotely. Alternative: the `log_interval` prints to stdout and is sufficient for learning.

6. **Python version & PyTorch version**: The code requires Python ≥ 3.8 and PyTorch ≥ 2.0 for `torch.compile` and Flash Attention. If you are on PyTorch 1.x, `compile=False` and Flash Attention will be disabled automatically — everything still works, just slower.

---

Review PLAN.md and reply **"approved"** when ready to build.
