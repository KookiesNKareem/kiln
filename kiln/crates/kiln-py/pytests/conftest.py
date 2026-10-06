import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
DESIGNS = ROOT / "designs"


@pytest.fixture
def program(tmp_path):
    def write(body: str) -> Path:
        p = tmp_path / "candidate.py"
        p.write_text(textwrap.dedent(body))
        return p

    return write


@pytest.fixture(scope="session")
def session():
    import kiln

    return kiln.Session(cache="none")
