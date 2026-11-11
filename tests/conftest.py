"""Shared pytest setup: make ``src/`` importable without an editable install."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="also run tests marked slow")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: minutes-long solver tests; run with --runslow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    import pytest

    skip = pytest.mark.skip(reason="slow; run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
