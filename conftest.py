# Register the rubra pytest plugin when running tests from source.
# When installed via pip, the [project.entry-points."pytest11"] entry in
# pyproject.toml handles registration automatically.
pytest_plugins = ["rubra.pytest_plugin"]
