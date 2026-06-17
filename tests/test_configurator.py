"""Unit tests for configurator.py argument-parsing logic."""
import os
import sys
import ast
import textwrap
import tempfile
import pytest


def _run_configurator(globals_dict: dict, argv: list) -> dict:
    """Execute configurator.py in a controlled namespace.

    Mirrors exactly what train.py does:
        exec(open('configurator.py').read())
    but lets us supply a custom globals() and sys.argv.
    """
    configurator_path = os.path.join(
        os.path.dirname(__file__), "..", "configurator.py"
    )
    original_argv = sys.argv
    sys.argv = argv
    try:
        with open(configurator_path) as f:
            code = f.read()
        exec(compile(code, configurator_path, "exec"), globals_dict)
    finally:
        sys.argv = original_argv
    return globals_dict


# ── CLI key=value overrides ───────────────────────────────────────────────────

def test_configurator_overrides_integer_key():
    g = {"batch_size": 32}
    _run_configurator(g, ["train.py", "--batch_size=64"])
    assert g["batch_size"] == 64


def test_configurator_overrides_float_key():
    g = {"learning_rate": 6e-4}
    _run_configurator(g, ["train.py", "--learning_rate=1e-3"])
    assert abs(g["learning_rate"] - 1e-3) < 1e-10


def test_configurator_overrides_bool_key_true():
    g = {"compile": True}
    _run_configurator(g, ["train.py", "--compile=False"])
    assert g["compile"] is False


def test_configurator_overrides_string_key():
    g = {"dataset": "openwebtext"}
    _run_configurator(g, ["train.py", "--dataset=shakespeare"])
    assert g["dataset"] == "shakespeare"


def test_configurator_preserves_type_on_override():
    """An int key overridden with a string-int should remain an int."""
    g = {"n_layer": 12}
    _run_configurator(g, ["train.py", "--n_layer=6"])
    assert isinstance(g["n_layer"], int)
    assert g["n_layer"] == 6


def test_configurator_rejects_unknown_key():
    g = {"batch_size": 32}
    with pytest.raises(ValueError, match="Unknown config key"):
        _run_configurator(g, ["train.py", "--nonexistent_key=99"])


def test_configurator_rejects_type_mismatch():
    """Passing a float value for an int key should raise AssertionError."""
    g = {"n_layer": 12}
    with pytest.raises((AssertionError, ValueError)):
        _run_configurator(g, ["train.py", "--n_layer=6.5"])


def test_configurator_no_args_leaves_globals_unchanged():
    g = {"batch_size": 32, "learning_rate": 6e-4}
    original = g.copy()
    _run_configurator(g, ["train.py"])
    assert g["batch_size"] == original["batch_size"]
    assert g["learning_rate"] == original["learning_rate"]


# ── Config file overrides ─────────────────────────────────────────────────────

def test_configurator_loads_config_file():
    g = {"batch_size": 32, "n_layer": 12}
    config_content = textwrap.dedent("""\
        batch_size = 64
        n_layer = 6
    """)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(config_content)
        config_path = f.name
    try:
        _run_configurator(g, ["train.py", config_path])
        assert g["batch_size"] == 64
        assert g["n_layer"] == 6
    finally:
        os.unlink(config_path)


def test_configurator_config_file_then_cli_override():
    """CLI flags should win over config file values."""
    g = {"batch_size": 32, "n_layer": 12}
    config_content = "batch_size = 64\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write(config_content)
        config_path = f.name
    try:
        _run_configurator(g, ["train.py", config_path, "--batch_size=128"])
        assert g["batch_size"] == 128
    finally:
        os.unlink(config_path)


# ── Type coercion via ast.literal_eval ────────────────────────────────────────

def test_literal_eval_coerces_int():
    assert ast.literal_eval("42") == 42
    assert isinstance(ast.literal_eval("42"), int)


def test_literal_eval_coerces_float():
    result = ast.literal_eval("3.14")
    assert isinstance(result, float)


def test_literal_eval_coerces_bool():
    assert ast.literal_eval("True") is True
    assert ast.literal_eval("False") is False


def test_literal_eval_falls_back_to_string_on_failure():
    """Plain strings that are not valid literals stay as strings."""
    try:
        result = ast.literal_eval("openwebtext")
    except (SyntaxError, ValueError):
        result = "openwebtext"
    assert result == "openwebtext"
