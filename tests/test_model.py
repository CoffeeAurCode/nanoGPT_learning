"""Unit and integration tests for model.py."""
import os
import sys
import math
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import LayerNorm, CausalSelfAttention, MLP, Block, GPTConfig, GPT


@pytest.fixture
def cfg():
    return GPTConfig(
        block_size=16,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        bias=True,
    )


# ── LayerNorm ─────────────────────────────────────────────────────────────────

def test_layer_norm_with_bias_preserves_shape(cfg):
    ln = LayerNorm(cfg.n_embd, bias=True)
    x = torch.randn(2, 8, cfg.n_embd)
    assert ln(x).shape == x.shape


def test_layer_norm_without_bias_has_no_bias_parameter():
    ln = LayerNorm(32, bias=False)
    assert ln.bias is None


def test_layer_norm_with_bias_creates_bias_parameter():
    ln = LayerNorm(32, bias=True)
    assert ln.bias is not None
    assert ln.bias.shape == (32,)


def test_layer_norm_zero_mean_unit_variance():
    ln = LayerNorm(64, bias=False)
    x = torch.randn(4, 8, 64) * 50
    out = ln(x)
    assert out.mean().abs().item() < 0.05
    assert abs(out.std().item() - 1.0) < 0.1


# ── CausalSelfAttention ───────────────────────────────────────────────────────

def test_causal_self_attention_output_shape(cfg):
    attn = CausalSelfAttention(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    assert attn(x).shape == x.shape


def test_causal_self_attention_rejects_bad_head_count():
    """n_embd must be divisible by n_head."""
    with pytest.raises(AssertionError):
        CausalSelfAttention(GPTConfig(n_embd=10, n_head=3, block_size=8))


def test_causal_self_attention_is_causal(cfg):
    """Token at position i must not see tokens at j > i.

    If we change only the last token of the sequence, every output position
    except the last should remain identical (causal masking prevents look-ahead).
    """
    attn = CausalSelfAttention(cfg)
    attn.eval()
    seq_len = 6
    x1 = torch.randn(1, seq_len, cfg.n_embd)
    x2 = x1.clone()
    x2[0, -1] = torch.randn(cfg.n_embd)
    out1 = attn(x1)
    out2 = attn(x2)
    assert torch.allclose(out1[0, :-1], out2[0, :-1], atol=1e-5)


# ── MLP ───────────────────────────────────────────────────────────────────────

def test_mlp_output_shape(cfg):
    mlp = MLP(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    assert mlp(x).shape == x.shape


def test_mlp_hidden_dimension_is_4x_embd(cfg):
    mlp = MLP(cfg)
    assert mlp.c_fc.out_features == 4 * cfg.n_embd


# ── Block ─────────────────────────────────────────────────────────────────────

def test_block_output_shape(cfg):
    block = Block(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    assert block(x).shape == x.shape


# ── GPT ───────────────────────────────────────────────────────────────────────

def test_gpt_forward_no_targets_returns_last_position_logits(cfg):
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 1, cfg.vocab_size)
    assert loss is None


def test_gpt_forward_with_targets_returns_full_logits_and_loss(cfg):
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    logits, loss = model(idx, targets)
    assert logits.shape == (2, 8, cfg.vocab_size)
    assert loss is not None
    assert loss.item() > 0


def test_gpt_loss_decreases_after_one_step(cfg):
    """One gradient step should reduce the training loss."""
    model = GPT(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    _, loss_before = model(idx, targets)
    loss_before.backward()
    optimizer.step()
    optimizer.zero_grad()
    _, loss_after = model(idx, targets)
    assert loss_after.item() < loss_before.item()


def test_gpt_weight_tying(cfg):
    """wte and lm_head must share the same weight tensor object."""
    model = GPT(cfg)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_gpt_get_num_params_excludes_position_embeddings_by_default(cfg):
    model = GPT(cfg)
    n_all = model.get_num_params(non_embedding=False)
    n_no_pos = model.get_num_params(non_embedding=True)
    pos_embed_params = cfg.block_size * cfg.n_embd
    assert n_all - n_no_pos == pos_embed_params


def test_gpt_generate_extends_sequence_by_max_new_tokens(cfg):
    model = GPT(cfg)
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (1, 6)


def test_gpt_generate_with_top_k(cfg):
    model = GPT(cfg)
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=4, top_k=3)
    assert out.shape == (1, 5)


def test_gpt_generate_temperature_zero_is_deterministic(cfg):
    """Temperature approaching zero should collapse to the argmax token."""
    model = GPT(cfg)
    model.eval()
    idx = torch.zeros((1, 1), dtype=torch.long)
    out1 = model.generate(idx.clone(), max_new_tokens=3, temperature=1e-10)
    out2 = model.generate(idx.clone(), max_new_tokens=3, temperature=1e-10)
    assert torch.equal(out1, out2)


def test_gpt_forward_rejects_sequence_longer_than_block_size(cfg):
    model = GPT(cfg)
    too_long = torch.zeros((1, cfg.block_size + 1), dtype=torch.long)
    with pytest.raises(AssertionError):
        model(too_long)


def test_gpt_crop_block_size_shrinks_position_embeddings(cfg):
    model = GPT(cfg)
    new_size = cfg.block_size // 2
    model.crop_block_size(new_size)
    assert model.config.block_size == new_size
    assert model.transformer.wpe.weight.shape[0] == new_size


def test_gpt_crop_block_size_rejects_larger_size(cfg):
    model = GPT(cfg)
    with pytest.raises(AssertionError):
        model.crop_block_size(cfg.block_size + 1)


# ── Integration ───────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("INTEGRATION"),
    reason="Set INTEGRATION=1 to run — downloads ~500 MB from HuggingFace",
)
def test_gpt_from_pretrained_gpt2_has_correct_architecture():
    model = GPT.from_pretrained("gpt2")
    assert model.config.n_layer == 12
    assert model.config.n_head == 12
    assert model.config.n_embd == 768
    n_params = model.get_num_params()
    assert abs(n_params - 124_000_000) < 2_000_000


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION"),
    reason="Set INTEGRATION=1 to run — requires CUDA",
)
def test_gpt_forward_on_cuda(cfg):
    assert torch.cuda.is_available(), "CUDA required for this test"
    model = GPT(cfg).cuda()
    idx = torch.randint(0, cfg.vocab_size, (2, 8)).cuda()
    logits, _ = model(idx)
    assert logits.device.type == "cuda"
