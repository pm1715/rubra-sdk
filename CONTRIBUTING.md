# Contributing to Rubra

Thank you for your interest in contributing to Rubra!

## Development Setup

```bash
git clone https://github.com/yourusername/rubra-sdk.git
cd rubra-sdk
pip install -e ".[dev]"
```

## Running Tests

```bash
# All tests
pytest

# Unit tests only (fast, ~2s)
pytest tests/unit/ -v

# Integration tests (uses real SQLite)
pytest tests/integration/ -v

# Single file
pytest tests/unit/test_tracer.py -v
```

## Code Style

Rubra uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
ruff check rubra/          # lint
ruff format rubra/         # format
ruff check --fix rubra/    # auto-fix
```

CI will reject PRs that fail lint.

## Adding a New Metric

1. Choose the right category: `rubra/core/metrics/{execution,tool,safety,quality,goal}/metrics.py`
2. Add a function that takes a `Trace` and returns a `MetricResult`
3. Add it to the `run_*_metrics()` function in the same file
4. Add tests in `tests/unit/test_{category}_metrics.py`
5. Update the metric count in the README

Metric naming convention: `snake_case`, verb-noun (e.g. `tool_call_success_rate`).

## Adding a New Integration

Create a new directory under `rubra/integrations/{framework}/`:

```
rubra/integrations/myframework/
├── __init__.py        # re-export public API
└── patch.py           # or callback.py / middleware.py
```

- Must work as a no-op when no `@rubra.agent` trace is active
- Must not import the framework at module level (lazy import inside the function)
- Add `myframework = ["myframework>=x.y"]` to `[project.optional-dependencies]` in `pyproject.toml`

## Pull Request Guidelines

- One feature or fix per PR
- Tests required for new metrics and integrations
- Keep `pyproject.toml` dependencies minimal — core stays at 4 deps
- Update `CHANGELOG.md` under `## Unreleased`
- Descriptive commit messages: `feat(metrics): add tool_selection_f1`, `fix(storage): upsert uses trace_id not pk`

## Reporting Issues

Use [GitHub Issues](https://github.com/yourusername/rubra-sdk/issues).  
Include: Python version, rubra version (`rubra version`), minimal reproduction, and expected vs actual behaviour.
