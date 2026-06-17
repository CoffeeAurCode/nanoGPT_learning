# Module 01 — Data Pipeline

**Source files:** `data/shakespeare_char/prepare.py` · `data/shakespeare/prepare.py` · `data/openwebtext/prepare.py`  
**Estimated time:** 45 minutes  
**Next:** [Module 02 — Configuration System](module_02_configuration_system.md)

---

## What You Are Building

The model never sees raw text. Before training can begin, text must be converted into a flat sequence of integers (token IDs) and written to a binary file that training can read efficiently. This module covers the two tokenization strategies nanoGPT supports (character-level and BPE), why data is stored as raw binary rather than in a database or HuggingFace cache, and how `np.memmap` enables reading a 17 GB file without loading it into RAM.

---

## Concept Deep-Dives

### 1. Tokenization: Converting Text to Integers

**Definition:** Tokenization maps a string to a sequence of integer IDs drawn from a fixed vocabulary. The reverse mapping (IDs → string) is called decoding.

#### Strategy A: Character-level

Every unique character in the training corpus gets an ID. Simple but produces long sequences — one character = one token.

```python
text = "hello"
chars = sorted(set(text))        # ['e', 'h', 'l', 'o']
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

ids = encode("hello")    # [1, 0, 2, 2, 3]
print(decode(ids))       # "hello"
```

**How nanoGPT uses it:** `data/shakespeare_char/prepare.py` builds `stoi` and `itos` from the Shakespeare corpus (65 unique characters). It saves them in `meta.pkl` so `sample.py` can decode generated IDs.

**What breaks without decoding map:** Generated IDs are meaningless integers. You cannot print the model's output as text.

---

#### Strategy B: Byte-Pair Encoding (BPE)

**Definition:** BPE starts with individual bytes and iteratively merges the most frequent adjacent pair into a new token. The result is a vocabulary of ~50,000 subword pieces that balances compression and coverage.

```python
# You do not implement BPE from scratch — tiktoken handles it.
import tiktoken
enc = tiktoken.get_encoding("gpt2")   # GPT-2's exact vocabulary

ids = enc.encode_ordinary("Hello, world!")
print(ids)                     # [15496, 11, 995, 0]
print(enc.decode(ids))         # "Hello, world!"
print(enc.n_vocab)             # 50257
```

**How nanoGPT uses it:** `data/shakespeare/prepare.py` uses `enc.encode_ordinary()` (which ignores special tokens) to tokenize the Shakespeare text. The result is a shorter sequence than character-level, so the model can cover more context per `block_size`.

**What breaks naively:** If you use `enc.encode()` instead of `enc.encode_ordinary()`, it will try to encode the `<|endoftext|>` special token in mid-text and may raise an error on documents that contain that substring literally.

---

### 2. NumPy Memory-Mapped Files (`np.memmap`)

**Definition:** A memory-mapped file is a file that the OS maps directly into virtual address space. Reading a slice of a mmap'd array reads that byte range from disk without loading the whole file into RAM first.

```python
import numpy as np
import tempfile, os

# Create a large array and write it to disk
arr = np.arange(1_000_000, dtype=np.uint16)
path = "/tmp/tokens.bin"
arr.tofile(path)

# Open it as a memory-mapped array — no RAM used yet
mmap = np.memmap(path, dtype=np.uint16, mode='r')

# Only bytes 200–208 are read from disk here
print(mmap[100:104])    # [100, 101, 102, 103]

# Slicing a random window (as the data loader does):
import torch
i = 42
chunk = torch.from_numpy(mmap[i : i + 8].astype(np.int64))
print(chunk)            # tensor([42, 43, 44, 45, 46, 47, 48, 49])

os.unlink(path)
```

**How nanoGPT uses it:** `get_batch()` in `train.py` opens the binary file as a memmap on every call:

```python
data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
ix = torch.randint(len(data) - block_size, (batch_size,))
x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
```

**Why recreate memmap every batch:** If you hold one memmap object across many batches, Python's reference counting keeps the OS mapping alive and accumulates a subtle memory leak. Creating a new memmap object each time lets the OS release the mapping after the batch is consumed. See the [Stack Overflow explanation cited in train.py](https://stackoverflow.com/a/61472122).

**What breaks without memmap:** For a 17 GB `train.bin`, loading the whole file with `np.load()` or `torch.load()` would require 17 GB of RAM before training even starts.

---

### 3. The `uint16` dtype and vocabulary limit

**Definition:** `uint16` stores integers 0–65535 in 2 bytes each. GPT-2's vocabulary has 50,257 tokens (all ≤ 65535), so uint16 is sufficient and halves storage vs. `int32`.

```python
import numpy as np
ids = [0, 100, 50256, 50257]   # 50257 is out of range!
arr16 = np.array(ids, dtype=np.uint16)
print(arr16)   # [    0   100 50256     1]  ← 50257 wraps around!

# Safe: vocab_size must be <= 65535
assert max(ids[:-1]) < 2**16
```

**How nanoGPT uses it:** Both `prepare.py` scripts use `dtype=np.uint16`. GPT-2's vocabulary (50,257 tokens) fits within uint16. The comment in `model.py` pads `vocab_size` to 50,304 (nearest multiple of 64) for CUDA efficiency — still well within uint16.

**What breaks naively:** If you trained a model with vocab_size > 65535 (e.g., a Chinese BPE tokenizer) and stored IDs as uint16, token IDs would silently wrap around and corrupt your training data.

---

### 4. Train/Val Split by Position

**Definition:** Split the corpus at a fixed position (e.g., 90% train, 10% val). Do not shuffle.

```python
data = open('input.txt').read()
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]
```

**Why no shuffling:** The model learns from contiguous token sequences. Shuffling at the character/token level would break all grammatical structure. Shuffling at the document level (as OpenWebText does) is fine — but nanoGPT keeps it simple.

**What breaks if you shuffle character-level:** You get sequences like `"e thr wol"` — meaningless garbage. The model would still learn some statistics but the task becomes much harder.

---

## Reading the Source Files

### `data/shakespeare_char/prepare.py` — line by line

```python
import os
import requests
import pickle
import numpy as np
```
Only standard library plus numpy. No tiktoken dependency — character-level needs nothing external.

```python
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        f.write(requests.get(data_url).text)
```
Lazy download: only fetches from the network if the file isn't already present. `os.path.dirname(__file__)` ensures the file is saved next to this script regardless of where you run it from.

```python
chars = sorted(set(data))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
```
`sorted(set(...))` is deterministic — the vocab is always in the same order regardless of which Python version or platform you run on.

```python
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
```
`np.array(...).tofile()` writes raw bytes with no header. The resulting `.bin` file is just a flat sequence of 2-byte unsigned integers. This is the simplest possible binary format — no compression, no metadata, instant reads.

```python
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
```
`meta.pkl` is the decoder ring. `train.py` checks for it to get `vocab_size`, and `sample.py` loads `stoi`/`itos` to decode generated token IDs back to text.

### `data/shakespeare/prepare.py` — key differences from char version

```python
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
```
Uses GPT-2's BPE tokenizer instead of character-level. No `meta.pkl` is saved — `sample.py` will fall back to the GPT-2 tiktoken encoding by default. Vocab size is 50,257 instead of 65.

### `data/openwebtext/prepare.py` — large-scale version

This script handles a dataset 10,000× larger. Key differences:

```python
from datasets import load_dataset
dataset = load_dataset("openwebtext", num_proc=8)
```
Uses HuggingFace `datasets` to stream and multiprocess the download. The `num_proc=8` parallelizes both downloading and tokenization across CPU cores.

```python
arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
for batch_idx in tqdm(range(total_batches)):
    batch = dset.shard(num_shards=1024, index=batch_idx, ...).with_format('numpy')
    arr_batch = np.concatenate(batch['ids'])
    arr[idx : idx + len(arr_batch)] = arr_batch
    idx += len(arr_batch)
arr.flush()
```
Writes to the output file in 1024 batches to avoid OOM. `mode='w+'` creates a new writable memmap. `arr.flush()` ensures all OS write buffers are committed to disk.

---

## Why This Design

**Why flat binary instead of a database or pickle?**  
A flat `uint16` binary file supports O(1) random access by index: to read tokens at position `i`, seek to byte `2*i` and read `2*block_size` bytes. A pickle or parquet file would require decompression. A database adds a network layer. Flat binary is the lowest-latency option for sequential reads with random offsets.

**Why not store a `torch.Tensor` directly?**  
PyTorch tensors saved with `torch.save()` include metadata and require loading the entire file. `np.memmap` lets the OS page tokens in on demand — the training loop never holds more than one batch in memory at a time.

**Why 90%/10% and not 80%/20%?**  
Shakespeare is ~1M characters. 10% ≈ 100K characters is ample for estimating generalization loss. For OpenWebText (9B tokens), even 0.05% gives 4.5M tokens for validation — more than enough. The split ratio matters less than having enough validation tokens for a statistically meaningful loss estimate.

---

## Running the Tests

```bash
# Unit tests only (no network, no tiktoken required)
pytest tests/test_data_pipeline.py -v -k "not INTEGRATION"

# All tests including integration (downloads Shakespeare ~1 MB)
INTEGRATION=1 pytest tests/test_data_pipeline.py -v
```

Key tests to understand:
- `test_char_encode_decode_roundtrip`: verifies that `encode(decode(text)) == text`
- `test_uint16_binary_write_and_read_roundtrip`: verifies the binary file format
- `test_memmap_does_not_load_full_array_into_ram`: confirms memmap works on large data

---

## Checkpoint ✓

Run this in a Python REPL after `python data/shakespeare_char/prepare.py`:

```python
import numpy as np, pickle

train = np.memmap('data/shakespeare_char/train.bin', dtype='uint16', mode='r')
print(f"train tokens: {len(train):,}")   # should be ~301,966

with open('data/shakespeare_char/meta.pkl', 'rb') as f:
    meta = pickle.load(f)
itos = meta['itos']
print(''.join(itos[i] for i in train[:100]))   # first 100 chars of Shakespeare
```

**Expected:** You see the opening of Shakespeare's text decoded correctly. The character count matches what `prepare.py` printed.

---

## Exercises

**1 (Easy) — Inspect the binary file:**  
After running `prepare.py`, open `data/shakespeare_char/train.bin` in a hex editor (or use Python's `struct` module) and verify that the first two bytes equal the integer ID of the first character in the text.

```python
import struct
with open('data/shakespeare_char/train.bin', 'rb') as f:
    raw = f.read(2)
print(struct.unpack('<H', raw)[0])  # little-endian uint16
```

**Success condition:** The printed integer matches `meta['stoi'][open('data/shakespeare_char/input.txt').read()[0]]`.

**2 (Medium) — Write a word-level tokenizer:**  
Create `data/shakespeare_word/prepare.py` that tokenizes by whitespace-split words instead of characters. Build `stoi`/`itos`, encode the corpus, and save `train.bin`, `val.bin`, `meta.pkl`. Run `train.py` with it.

**Success condition:** Training starts and prints a sensible vocab size. Generated text is recognizable English words (even if grammatically wrong).

**3 (Hard) — Measure data-loading throughput:**  
Modify `get_batch()` in `train.py` to time itself and print tokens/second. Compare three strategies: (a) current memmap-per-call, (b) one memmap object reused across calls, (c) `np.load()` of a pre-loaded array. Which is fastest? Which uses the most RAM?

**Success condition:** You can report throughput numbers for all three strategies and explain the trade-off between RAM usage and read speed.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| Byte-pair encoding explained | [Hugging Face NLP Course — Tokenizers](https://huggingface.co/learn/nlp-course/chapter6/5) | Clearest explanation of BPE with animations; directly relevant to tiktoken |
| tiktoken docs | [openai/tiktoken on GitHub](https://github.com/openai/tiktoken) | Source + README with all encoding names and API |
| NumPy memmap | [NumPy memmap docs](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html) | Official reference; read the `mode` parameter table carefully |
| Why uint16 | [NumPy dtype docs](https://numpy.org/doc/stable/reference/arrays.dtypes.html) | Explains all scalar types, sizes, and ranges |

---

## What's Next

[Module 02 — Configuration System](module_02_configuration_system.md): learn how nanoGPT uses `exec()` and `globals()` to merge config files and CLI flags without argparse.
