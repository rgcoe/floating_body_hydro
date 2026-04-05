from __future__ import annotations

from pathlib import Path

import pytest

import pdstripy


def test_parse_pdstrip_out_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        pdstripy.parse_pdstrip_out(Path("does_not_exist_pdstrip.out"))


def test_parse_responsefunctions_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        pdstripy.parse_responsefunctions(Path("does_not_exist_responsefunctions"))


def test_parse_sectionresults_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        pdstripy.parse_sectionresults(Path("does_not_exist_sectionresults"))


def test_parse_responsefunctions_truncated_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "responsefunctions"
    path.write_text("title\nonly one header line\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pdstripy.parse_responsefunctions(path)


def test_parse_sectionresults_truncated_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "sectionresults"
    path.write_text("title\n1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        pdstripy.parse_sectionresults(path)
