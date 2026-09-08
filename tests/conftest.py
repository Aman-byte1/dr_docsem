"""Shared pytest fixtures."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docsem.io_utils import ensure_dirs  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _dirs():
    ensure_dirs()


@pytest.fixture()
def tiny_blocks():
    return {
        "b01": "Regional Operations Brief",
        "b06": "Inventory: 40 units of stock A in the west warehouse.",
        "b10": "A shark is 10 feet long and two remoras are 6 inches each; "
        "what percentage of the shark length is the pair?",
        "b13": "This block is narrative and is not a calculation request.",
    }
