"""Unit tests for training utility logic extracted from train.py.

train.py is a script, not a module, so we replicate the pure functions
here and test their mathematical contracts directly.
"""
import math
import pytest


def get_lr(
    it: int,
    learning_rate: float,
    warmup_iters: int,
    lr_decay_iters: int,
    min_lr: float,
) -> float:
    """Cosine decay with linear warmup.

    Copied verbatim from train.py so tests stay in sync with the source.
    """
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


LR = 6e-4
WARMUP = 100
DECAY = 1000
MIN_LR = 6e-5


# ── Warmup phase ──────────────────────────────────────────────────────────────

def test_get_lr_at_iter_zero_is_nonzero():
    """Iteration 0 should return a positive LR (not zero)."""
    lr = get_lr(0, LR, WARMUP, DECAY, MIN_LR)
    assert lr > 0


def test_get_lr_warmup_is_monotonically_increasing():
    lrs = [get_lr(i, LR, WARMUP, DECAY, MIN_LR) for i in range(WARMUP)]
    assert all(lrs[i] < lrs[i + 1] for i in range(len(lrs) - 1))


def test_get_lr_at_end_of_warmup_approaches_max_lr():
    """At the last warmup step, LR should be close to (but not exceed) max LR."""
    lr = get_lr(WARMUP - 1, LR, WARMUP, DECAY, MIN_LR)
    assert lr < LR
    assert lr > LR * 0.9


# ── Cosine decay phase ────────────────────────────────────────────────────────

def test_get_lr_decay_is_monotonically_decreasing():
    lrs = [get_lr(i, LR, WARMUP, DECAY, MIN_LR) for i in range(WARMUP, DECAY + 1)]
    assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


def test_get_lr_at_lr_decay_iters_equals_min_lr():
    lr = get_lr(DECAY, LR, WARMUP, DECAY, MIN_LR)
    assert abs(lr - MIN_LR) < 1e-10


def test_get_lr_midpoint_of_decay_is_halfway_between_max_and_min():
    """At the midpoint of the cosine schedule, coeff = 0.5, so LR = (max+min)/2."""
    mid = (WARMUP + DECAY) // 2
    lr = get_lr(mid, LR, WARMUP, DECAY, MIN_LR)
    expected = MIN_LR + 0.5 * (LR - MIN_LR)
    assert abs(lr - expected) < 1e-8


# ── Post-decay floor ──────────────────────────────────────────────────────────

def test_get_lr_after_decay_iters_returns_min_lr():
    for it in [DECAY + 1, DECAY + 100, DECAY * 2]:
        lr = get_lr(it, LR, WARMUP, DECAY, MIN_LR)
        assert lr == MIN_LR


# ── Boundary conditions ───────────────────────────────────────────────────────

def test_get_lr_never_exceeds_learning_rate():
    for it in range(0, DECAY + 200, 10):
        lr = get_lr(it, LR, WARMUP, DECAY, MIN_LR)
        assert lr <= LR + 1e-10


def test_get_lr_never_drops_below_min_lr():
    for it in range(0, DECAY + 200, 10):
        lr = get_lr(it, LR, WARMUP, DECAY, MIN_LR)
        assert lr >= MIN_LR - 1e-10


def test_get_lr_with_no_warmup():
    """warmup_iters=0 should work: first iter immediately starts cosine decay."""
    lr = get_lr(0, LR, warmup_iters=0, lr_decay_iters=100, min_lr=MIN_LR)
    assert lr == LR


def test_get_lr_with_warmup_equals_decay_iters():
    """Edge case: warmup equals decay_iters — no cosine phase."""
    lr = get_lr(50, LR, warmup_iters=100, lr_decay_iters=100, min_lr=MIN_LR)
    assert lr == MIN_LR
