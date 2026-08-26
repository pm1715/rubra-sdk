# The [project.entry-points."pytest11"] entry in pyproject.toml registers
# rubra.pytest_plugin automatically once the package is installed (editable
# or not) — which is required for `import rubra` to work in tests anyway.
# Do NOT also register it here via `pytest_plugins = [...]`: pytest has
# already loaded it under the name "rubra" by the time this file is read,
# so a second explicit registration raises
# "ValueError: Plugin already registered under a different name".
