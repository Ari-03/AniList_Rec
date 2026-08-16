from pathlib import Path

import pytest
from fixture_bundle import make_fixture_bundle


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    return make_fixture_bundle(tmp_path / "bundle")
