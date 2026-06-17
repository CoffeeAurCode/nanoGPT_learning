# Module 02 — Configuration System

**Source files:** `configurator.py` · `config/train_shakespeare_char.py` · `config/finetune_shakespeare.py`  
**Estimated time:** 30 minutes  
**Next:** [Module 03 — Transformer Building Blocks](module_03_transformer_building_blocks.md)

---

## What You Are Building

Most projects use `argparse`, Hydra, or OmegaConf to manage configuration. nanoGPT uses a 40-line `exec()`-based system. You will understand exactly what `exec()` does to Python's namespace, why Karpathy chose this approach, and what its trade-offs are. You will also be able to write new config files and override any parameter from the CLI.

---

## Concept Deep-Dives

### 1. Python's `exec()` and `globals()`

**Definition:** `exec(code_string)` evaluates arbitrary Python code in the current scope. `globals()` returns the calling frame's global namespace as a live dictionary — writes to it change the actual globals.

**Minimal standalone example:**

```python
# globals() is a live dictionary of the current module's names
x = 10
print(globals()['x'])   # 10

# exec() runs code as if you typed it in the current scope
exec("x = 99")
print(x)   # 99  ← the actual variable x was changed!

# exec() can also run an entire file
exec(open('some_config.py').read())
```

**How nanoGPT uses it:** `train.py` defines all hyperparameters as plain module-level variables (e.g., `batch_size = 12`), then calls:

```python
exec(open('configurator.py').read())
```

Because `exec()` runs in the caller's scope, the code inside `configurator.py` can read and write `train.py`'s globals directly — no `import`, no `config.` prefix needed.

**What breaks naively:** If `configurator.py` were a regular module and you called `import configurator; configurator.override()`, the function's `globals()` would return `configurator`'s namespace, not `train.py`'s. The overrides would silently go into the wrong namespace.

---

### 2. `ast.literal_eval` for Safe Type Coercion

**Definition:** `ast.literal_eval(s)` parses a string and returns a Python literal (int, float, bool, string, list, etc.) without executing arbitrary code. It is safer than `eval()` because it rejects anything that isn't a literal.

```python
from ast import literal_eval

literal_eval("42")          # → int 42
literal_eval("3.14")        # → float 3.14
literal_eval("True")        # → bool True
literal_eval("[1, 2, 3]")   # → list [1, 2, 3]

# This is safe — it will not execute os.system("rm -rf /")
try:
    literal_eval("__import__('os').system('echo pwned')")
except (ValueError, SyntaxError):
    print("rejected")   # ← this is what happens
```

**How nanoGPT uses it:** When you pass `--batch_size=64` on the CLI, `configurator.py` receives the string `"64"`. It must become the integer `64`, not the string `"64"`, so that type-checking against the existing default (`batch_size = 12`, an int) passes.

```python
# From configurator.py:
try:
    attempt = literal_eval(val)   # "64" → 64
except (SyntaxError, ValueError):
    attempt = val                  # fallback: keep as string
assert type(attempt) == type(globals()[key])
globals()[key] = attempt
```

**What breaks without it:** Using `eval()` instead of `literal_eval` would execute any Python expression passed as a CLI argument — a serious security vulnerability if config files or CLI flags ever come from untrusted input. Using `int(val)` would require separate branches for every type (int, float, bool, str).

---

### 3. The Config File Pattern

**Definition:** A config file is just a Python script that assigns values to variables. When `exec()`-ed inside `train.py`, it sets `train.py`'s globals.

```python
# config/my_experiment.py
batch_size = 128
learning_rate = 3e-4
n_layer = 4
wandb_log = True
wandb_run_name = 'my_experiment'
```

```bash
# Usage:
python train.py config/my_experiment.py --n_layer=8
```

The config file runs first (sets `n_layer=4`), then the CLI flag runs (`n_layer=8`), so CLI always wins.

**How nanoGPT uses it:** All configs in `config/` are plain Python files. They can include comments, arithmetic, and conditional logic:

```python
# config/finetune_shakespeare.py
init_from = 'gpt2'         # Load pretrained weights
max_iters = 20             # Only 20 iterations for fine-tuning
decay_lr = False           # No LR decay for such a short run
learning_rate = 3e-5       # Lower LR for fine-tuning
```

**What breaks with argparse:** You would need `parser.add_argument('--init_from', ...)` for every hyperparameter — dozens of lines. Changing a default requires editing `train.py`, not just a config file. Config files can't easily do arithmetic (e.g., `min_lr = learning_rate / 10`).

---

## Reading the Source File

### `configurator.py` — complete walkthrough

```python
import sys
from ast import literal_eval
```
Only two imports. `sys.argv` provides CLI arguments; `literal_eval` handles type coercion.

```python
for arg in sys.argv[1:]:
```
Iterates over everything after the script name. `sys.argv[0]` is the script name itself.

```python
    if '=' not in arg:
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
```
If an argument has no `=`, it's treated as a config file path. The `assert not arg.startswith('--')` prevents accidents: `--config.py` would be silently treated as a file path otherwise.

Printing the config file content before executing it is deliberate: you always know exactly what values were used, even if the config file was generated programmatically.

```python
    else:
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]   # strip the '--'
```
Forces `--key=value` format. This catches typos: if you write `-batch_size=64` (one dash), the assert fires immediately.

```python
        if key in globals():
            try:
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                attempt = val
            assert type(attempt) == type(globals()[key])
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
```
The `type(attempt) == type(globals()[key])` check prevents silent type errors. If you pass `--n_layer=6.5` and `n_layer` is an int, you get an AssertionError immediately rather than discovering the bug hours into training.

**Design pattern:** This is the **Namespace Injection** pattern — injecting values into a foreign scope via `exec()` + `globals()`. It is unusual in production Python code but elegant for scripts where simplicity matters more than encapsulation.

---

## Why This Design

**Why `exec()` over argparse?**  
argparse requires declaring every hyperparameter twice: once as a default in `train.py` and once as `add_argument()`. With 20+ hyperparameters, this is 40+ lines of boilerplate. The exec approach: declare once, done.

**Why not Hydra or OmegaConf?**  
Both are excellent for large projects. For a 300-line training script, they add more complexity than they remove. Hydra also changes the working directory, which breaks relative path assumptions in nanoGPT.

**When would you choose differently?**  
Use Hydra/OmegaConf when: (1) your config has nested structure, (2) you need config composition (merge multiple files), (3) you need config versioning and reproducibility guarantees, (4) configs come from untrusted users (security matters). For a research training script owned by one person, `exec()` is fine.

**The security note:**  
The `exec(open(config_file).read())` call will execute any Python code in the config file. Never use this pattern with config files from untrusted sources — a malicious config file could delete files, exfiltrate data, or install malware.

---

## Running the Tests

```bash
pytest tests/test_configurator.py -v
```

Key tests:
- `test_configurator_overrides_integer_key`: verifies `--batch_size=64` changes the int
- `test_configurator_preserves_type_on_override`: confirms int stays int, not str
- `test_configurator_rejects_unknown_key`: confirms unknown keys raise immediately
- `test_configurator_config_file_then_cli_override`: verifies CLI beats config file

---

## Checkpoint ✓

```python
# In a Python REPL, simulate what train.py does:
import sys

# Pretend we are train.py — set up the globals
batch_size = 32
learning_rate = 6e-4
n_layer = 12

sys.argv = ['train.py', '--batch_size=64', '--n_layer=6']
exec(open('configurator.py').read())

print(batch_size)    # 64
print(n_layer)       # 6
print(learning_rate) # 6e-4  (unchanged)
```

Run this from the project root. All three assertions should hold.

---

## Exercises

**1 (Easy) — Trace the type-check:**  
Add a print statement inside `configurator.py` that prints `type(attempt)` and `type(globals()[key])` just before the `assert`. Run `python train.py --batch_size=64` and observe. Then run `python train.py --batch_size=64.0` and observe the AssertionError.

**Success condition:** You can explain why `64.0` is rejected when `batch_size` is an int.

**2 (Medium) — Add a list-valued config key:**  
In `train.py`, add a new hyperparameter `log_layers = [0, 3, 6]` (a list of layer indices to log). Then pass `--log_layers="[0, 5, 11]"` from the CLI. Verify it updates correctly. Hint: `literal_eval` handles lists.

**Success condition:** `log_layers` becomes `[0, 5, 11]` after the override and has the correct type (list, not str).

**3 (Hard) — Replace `exec()` with argparse:**  
Rewrite `configurator.py` to use Python's `argparse` module. It should accept `--key=value` for every variable in the caller's globals that is an int, float, bool, or str. Compare the lines of code. Which version is more robust? Which is more maintainable?

**Success condition:** Your argparse version passes all the `test_configurator.py` tests (after adapting the test helpers). Document one case where argparse handles something the exec version doesn't, and one case where exec is simpler.

---

## Resources

| What | Resource | Why recommended |
|------|----------|----------------|
| Python `exec()` | [Python docs — exec()](https://docs.python.org/3/library/functions.html#exec) | Official reference; read the `globals` and `locals` parameter notes carefully |
| `ast.literal_eval` | [Python docs — ast.literal_eval](https://docs.python.org/3/library/ast.html#ast.literal_eval) | Official reference; explains exactly what literals are accepted |
| Hydra (the alternative) | [Hydra docs — Quick Start](https://hydra.cc/docs/intro/) | Best-in-class configuration library; understand what nanoGPT is trading off |

---

## What's Next

[Module 03 — Transformer Building Blocks](module_03_transformer_building_blocks.md): the four neural-network classes that make up a single transformer layer.
