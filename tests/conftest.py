import os
from pathlib import Path

import pytest

TESTS_DIR = Path(os.path.dirname(__file__))


@pytest.fixture
def image_dir() -> Path:
    return TESTS_DIR / "files"
