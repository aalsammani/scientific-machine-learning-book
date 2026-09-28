"""Shared helpers for the test suite. Tests never write into results/, data/ or figures/."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE = REPO_ROOT / "code"
RESULTS = REPO_ROOT / "results"
DATA = REPO_ROOT / "data"

# Directories whose modules import each other by name.
for sub in ("reference_problems", "chapter01", "chapter09/darcy", "chapter09/fno"):
    p = str(CODE / sub)
    if p not in sys.path:
        sys.path.insert(0, p)


def load_module(relpath, name=None):
    """Import a script from code/ by path without running its __main__ block."""
    path = CODE / relpath
    name = name or path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_json(relpath):
    with open(REPO_ROOT / relpath) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
