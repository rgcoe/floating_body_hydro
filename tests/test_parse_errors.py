from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
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


def test_parse_responsefunctions_truncated_file_raises(
    parse_error_fixture_dir: Path,
) -> None:
    path = parse_error_fixture_dir / "responsefunctions_truncated"
    with pytest.raises(ValueError):
        pdstripy.parse_responsefunctions(path)


def test_parse_sectionresults_truncated_file_raises(
    parse_error_fixture_dir: Path,
) -> None:
    path = parse_error_fixture_dir / "sectionresults_truncated"
    with pytest.raises(ValueError):
        pdstripy.parse_sectionresults(path)


def test_parse_responsefunctions_empty_file_warns_and_returns_empty_dataset(
    parse_error_fixture_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = parse_error_fixture_dir / "responsefunctions_empty"

    with caplog.at_level(logging.WARNING):
        dataset, messages = pdstripy.parse_responsefunctions(path)

    assert len(dataset.data_vars) == 0
    assert len(dataset.coords) == 0
    assert len(messages.errors) == 0
    assert len(messages.warnings) == 1
    assert "responsefunctions file is empty" in messages.warnings[0]
    assert "responsefunctions file is empty" in caplog.text


def test_parse_responsefunctions_short_row_count_warns(
    parse_error_fixture_dir: Path,
) -> None:
    path = parse_error_fixture_dir / "responsefunctions_short_rows"

    dataset, messages = pdstripy.parse_responsefunctions(path)

    assert "rao_magnitude" in dataset.data_vars
    assert any(
        "responsefunctions rows parsed=1 but expected=2" in warning
        for warning in messages.warnings
    )


def test_parse_responsefunctions_section_count_mismatch_warns(
    parse_error_fixture_dir: Path,
) -> None:
    path = parse_error_fixture_dir / "responsefunctions_section_mismatch"
    metadata = pdstripy.PDStripMetadata(
        n_sections=3,
        section_x=np.array([0.0, 1.0, 2.0]),
        cases={},
    )

    dataset, messages = pdstripy.parse_responsefunctions(path, metadata=metadata)

    assert "rao_magnitude" in dataset.data_vars
    assert any(
        "Section count mismatch: pdstrip.out=3, responsefunctions=2." in warning
        for warning in messages.warnings
    )


def test_parse_sectionresults_without_metadata_warns_when_sections_cannot_be_inferred(
    parse_error_fixture_dir: Path,
) -> None:
    path = parse_error_fixture_dir / "sectionresults_three_blocks_declared_two_freq"

    dataset, messages = pdstripy.parse_sectionresults(path)

    assert dataset.sizes["section"] == 1
    assert dataset.sizes["omega"] == 2
    assert any(
        "Could not infer number of sections exactly" in warning
        for warning in messages.warnings
    )


def test_parse_sectionresults_block_count_mismatch_warns_with_metadata(
    parse_error_fixture_dir: Path,
) -> None:
    path = parse_error_fixture_dir / "sectionresults_three_blocks_declared_two_freq"
    metadata = pdstripy.PDStripMetadata(
        n_sections=2,
        section_x=np.array([0.0, 1.0]),
        cases={"wave_angle_deg_unique": np.array([0.0])},
    )

    dataset, messages = pdstripy.parse_sectionresults(path, metadata=metadata)

    assert dataset.sizes["section"] == 2
    assert any(
        "Block count mismatch: parsed blocks=3, expected=4" in warning
        for warning in messages.warnings
    )


def test_parse_sectionresults_incomplete_block_warns_and_skips_values(
    parse_error_fixture_dir: Path,
) -> None:
    path = parse_error_fixture_dir / "sectionresults_incomplete_block"

    dataset, messages = pdstripy.parse_sectionresults(path)

    assert "added_mass" in dataset.data_vars
    assert np.isnan(dataset["added_mass"].values).all()
    assert any("expected at least 15" in warning for warning in messages.warnings)


def test_parse_sectionresults_debug_log_mentions_frequency_blocks_per_section(
    pdstrip_fixture_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    metadata, _ = pdstripy.parse_pdstrip_out(pdstrip_fixture_dir / "pdstrip.out")

    with caplog.at_level(logging.DEBUG):
        pdstripy.parse_sectionresults(
            pdstrip_fixture_dir / "sectionresults",
            metadata=metadata,
        )

    assert "Parsed sectionresults header:" in caplog.text
    assert "frequency blocks per section" in caplog.text


def test_parse_pdstrip_folder_debug_logs_absolute_paths(
    pdstrip_fixture_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.DEBUG):
        pdstripy.parse_pdstrip_folder(pdstrip_fixture_dir)

    assert "Absolute path for PDStrip folder is" in caplog.text
    assert str(pdstrip_fixture_dir.resolve()) in caplog.text
