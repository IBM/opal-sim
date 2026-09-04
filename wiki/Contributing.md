# Contributing

Thanks for your interest in improving Opal. This page covers how to report issues, how to submit pull requests, and the formatting convention the project expects.

## Before you start

Set up your environment first — see [[Install Requirements]] for the `uv`/virtual-environment setup. This page assumes you already have a working checkout and can run the simulator.

Before submitting any change, make sure it passes the test suite — see [[Testing]] for how to run the tests.

## Reporting issues

Open an issue on GitHub. Three issue templates are available to help structure the report:

* **Bug report** — describe the bug, steps to reproduce (which config file, exact command line, and the error/crash output), and expected behavior.
* **Feature request** — describe the problem you're trying to solve and the solution you'd like.
* **Custom issue template** — a free-form template for anything that doesn't fit the other two (problem statement + desired functionality).

If you're unsure whether something is a bug or by design, open an issue anyway — that's the right place to ask.

## Submitting a pull request

1. Fork/branch, make your change.
2. Format the code (see below).
3. Run the test suite — see [[Testing]].
4. Open a pull request against `main`.

There is no separate PR template; a clear description of what changed and why is sufficient.

## Code formatting: Black

The project follows the [Black](https://black.readthedocs.io/) Python formatter as its one required convention. The configuration lives in `pyproject.toml`:

```toml
[tool.black]
line-length = 120
target-version = ["py311"]
```

Install Black and run the formatter from the top-level directory before sending a pull request:

```shell
uv pip install black
sh-black-formatter.sh
```

`sh-black-formatter.sh` simply runs `black --config ./pyproject.toml --no-cache .` over the whole repository, so it reformats every Python file, not just the ones you touched.

## Linting: Ruff

The project also runs [Ruff](https://docs.astral.sh/ruff/) to catch unused imports, unused local variables, and redefinitions (`F401`, `F811`, `F841`). Configuration lives in `pyproject.toml` alongside the Black config:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["F401", "F811", "F841"]
```

Install Ruff and run it from the top-level directory before sending a pull request:

```shell
uv pip install ruff
ruff check --config ./pyproject.toml opal/ tests/
```

The pre-commit hook (below) runs Ruff on just your staged files, so pre-existing issues elsewhere in the tree won't block your commit — but a full `ruff check` over `opal/` and `tests/` may still surface older findings unrelated to your change; fixing those isn't required to get your PR merged.

### Optional: pre-commit hook

The repository ships a git hook at `.githooks/pre-commit` that runs Black and then Ruff automatically on just the staged `.py` files at commit time (Black's formatting is re-staged automatically; a Ruff failure aborts the commit so you can fix it). It is **not** enabled by default — turn it on once per clone with:

```shell
git config core.hooksPath .githooks
```

This is a convenience so you don't forget to format and lint before committing; running the commands above manually before opening the PR achieves the same result.

## Configuration changes

If your change adds or modifies simulation configuration fields, please also update the documentation on [[Configuration Simulation]] so the config reference stays in sync with the code.

## Contacts / questions

If you have questions or run into issues, open an issue on GitHub and tag **@animeshtrivedi** and **@raduioanstoica**.

## License

Opal is released under the Apache License 2.0. By contributing, you agree that your contributions will be licensed under the same terms.
