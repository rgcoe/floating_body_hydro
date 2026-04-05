from __future__ import annotations

from pathlib import Path

import pytest

import pdstripy


@pytest.fixture(scope="session")
def pdstrip_fixture_dir() -> Path:
    path = Path(__file__).resolve().parent / "data" / "spheroid" / "pdstrip_run"
    required = ["pdstrip.out", "responsefunctions", "sectionresults"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        pytest.skip(f"Missing fixture files in {path}: {missing}")
    return path


@pytest.fixture(scope="session")
def parsed_components(pdstrip_fixture_dir: Path):
    metadata, meta_messages = pdstripy.parse_pdstrip_out(pdstrip_fixture_dir / "pdstrip.out")
    response_ds, response_messages = pdstripy.parse_responsefunctions(
        pdstrip_fixture_dir / "responsefunctions", metadata=metadata
    )
    section_ds, section_messages = pdstripy.parse_sectionresults(
        pdstrip_fixture_dir / "sectionresults", metadata=metadata
    )
    return {
        "metadata": metadata,
        "meta_messages": meta_messages,
        "response_ds": response_ds,
        "response_messages": response_messages,
        "section_ds": section_ds,
        "section_messages": section_messages,
    }


@pytest.fixture(scope="session")
def parsed_folder(pdstrip_fixture_dir: Path):
    return pdstripy.parse_pdstrip_folder(pdstrip_fixture_dir)
