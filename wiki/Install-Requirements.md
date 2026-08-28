# Install Requirements

Opal uses [uv](https://docs.astral.sh/uv/) — a fast Python package installer and resolver — to manage its virtual environment and dependencies. `pyproject.toml` / `uv.lock` are the source of truth for what gets installed.

## 1. Install uv

```shell
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 2. Clone and set up the project

```shell
git clone git@github.com:IBM/opal-sim.git
cd opal-sim

# Create a virtual environment with Python 3.11
uv venv --python 3.11

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the project with all dependencies
uv pip install -e .
```

**Note:** `.venv` has no `pip` binary. Use `uv pip install ...` for any pip-compatible package operation.

## 3. Watch out for conda

If you also have conda active (e.g. your shell prompt shows `(base)` or another conda env name): conda is not used by this project, but activating `.venv` does not remove conda from your `PATH` — whichever was activated *last* wins, and `pip` / `python3` can silently resolve to conda's copies instead of the project's. Check with:

```shell
which python3   # should print .../opal-sim/.venv/bin/python3
```

The most robust option is to skip activation entirely and prefix every command with `uv run`, which always uses `.venv` regardless of what else is active in your shell:

```shell
uv run pytest
uv run python3 ./opal/main.py
```

## Optional: installing and running with `pypy`

`pypy` is a fast JIT compiler that supports Python `3.11`. It can give a significant speedup for long-running simulations (JIT compilation needs time to warm up, so the benefit is mainly for long runs — we've seen close to 2x on a Moonshot conversation trace replay with a single worker).

### Installing `pypy`

```shell
brew install uv
# Make PyPy 3.11 available on this machine
uv python install pypy@3.11
# Creates a virtual environment using the PyPy interpreter you just installed
uv venv --python pypy@3.11
```

The default `transformers` package does not work under `pypy`, so install it separately without its dependencies (it's only needed here for config parsing):

```shell
uv pip install --no-deps transformers
```

Then remove `transformers` as a dependency from `pyproject.toml`, e.g.:

```patch
diff --git a/pyproject.toml b/pyproject.toml
index 2e96d77..b8e3b4a 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -19,7 +19,6 @@ dependencies = [
     "tqdm",
     "pytest>=9.0",
     "black>=25.0",
-    "transformers>=4.55.4",
 ]

 [project.optional-dependencies]
```

Then install the rest as usual:

```shell
uv pip install -e .
```

### Running opal with `pypy`

Under `pypy`, Hugging Face model names can't be used directly since `transformers` isn't fully functional. Instead, point the config at a complete, locally-downloaded model config file:

```json
"model": {
  "model_params": {
    "name": "Llama-3.3-70B-Instruct",
    "config_dir": "./model_config/"
  }
}
```

This expects the model config at `./model_config/Llama-3.3-70B-Instruct/config.json` — make sure it exists. With that in place, run:

```shell
pypy3 ./opal/main.py
```

---

Once installed, see [[Running]] for how to run your first simulation.
