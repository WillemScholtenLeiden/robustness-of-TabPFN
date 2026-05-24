"""
Ensures the project root is on sys.path so tests can import `helpers.*` and
`experiments.*` when pytest is run from anywhere, and registers the `slow`
marker used to gate TabPFN-based tests.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that need TabPFN inference")
