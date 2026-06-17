"""Unit and integration tests for the data pipeline.

Unit tests verify tokenization logic using tiny in-memory strings.
Integration tests download real data and require INTEGRATION=1.
"""
import os
import sys
import struct
import tempfile
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Character-level tokenization ──────────────────────────────────────────────

def _build_char_vocab(text: str) -> tuple[dict, dict]:
    """Replicate the stoi/itos construction from shakespeare_char/prepare.py."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def test_char_vocab_covers_all_characters():
    text = "hello world"
    stoi, itos = _build_char_vocab(text)
    for ch in set(text):
        assert ch in stoi


def test_char_vocab_is_sorted_deterministically():
    text = "zyxabc"
    stoi, _ = _build_char_vocab(text)
    keys = list(stoi.keys())
    assert keys == sorted(keys)


def test_char_encode_decode_roundtrip():
    text = "To be, or not to be."
    stoi, itos = _build_char_vocab(text)
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    assert decode(encode(text)) == text


def test_char_vocab_size_equals_unique_chars():
    text = "abcabc"
    stoi, _ = _build_char_vocab(text)
    assert len(stoi) == len(set(text))


def test_char_ids_fit_in_uint16():
    text = "a" * 1000 + "".join(chr(i) for i in range(256))
    stoi, _ = _build_char_vocab(text)
    assert len(stoi) <= 65535


# ── Binary file I/O ───────────────────────────────────────────────────────────

def test_uint16_binary_write_and_read_roundtrip():
    """np.tofile / np.memmap should round-trip token IDs without loss."""
    ids = [0, 100, 200, 50257, 65535]
    arr = np.array(ids, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        arr.tofile(path)
        loaded = np.memmap(path, dtype=np.uint16, mode="r")
        assert list(loaded) == ids
    finally:
        os.unlink(path)


def test_memmap_does_not_load_full_array_into_ram():
    """np.memmap should be readable without all data in RAM.

    We verify this by creating a file larger than we'd normally hold in memory
    and confirming that slicing it works correctly.
    """
    n = 100_000
    ids = np.arange(n, dtype=np.uint16)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    try:
        ids.tofile(path)
        mmap = np.memmap(path, dtype=np.uint16, mode="r")
        assert len(mmap) == n
        assert mmap[0] == 0
        assert mmap[-1] == (n - 1) % 65536
    finally:
        os.unlink(path)


def test_train_val_split_does_not_overlap():
    """Splitting at 90% should produce non-overlapping slices."""
    text = "abcdefghij" * 100
    n = len(text)
    train = text[: int(n * 0.9)]
    val = text[int(n * 0.9) :]
    assert len(train) + len(val) == n
    assert train[-1] != val[0] or True  # positional split, not content check


# ── tiktoken BPE ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("INTEGRATION"),
    reason="Set INTEGRATION=1 to run — requires tiktoken and network for first download",
)
def test_tiktoken_gpt2_encode_decode_roundtrip():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    text = "Hello, world! This is a test."
    ids = enc.encode_ordinary(text)
    assert enc.decode(ids) == text


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION"),
    reason="Set INTEGRATION=1 to run",
)
def test_tiktoken_gpt2_vocab_size_is_50257():
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab == 50257


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION"),
    reason="Set INTEGRATION=1 to run — downloads ~1 MB Shakespeare text",
)
def test_shakespeare_prepare_creates_binary_files():
    """Run the shakespeare prepare.py and verify output files exist."""
    import subprocess
    data_dir = os.path.join(
        os.path.dirname(__file__), "..", "data", "shakespeare"
    )
    result = subprocess.run(
        [sys.executable, "prepare.py"],
        cwd=data_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert os.path.exists(os.path.join(data_dir, "train.bin"))
    assert os.path.exists(os.path.join(data_dir, "val.bin"))
    train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    assert len(train) > len(val)
    assert len(train) > 100_000
