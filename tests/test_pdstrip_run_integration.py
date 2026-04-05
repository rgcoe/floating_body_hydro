from __future__ import annotations

import shutil

import numpy as np

import pdstripy


def test_metadata_from_fixture(parsed_components) -> None:
    metadata = parsed_components["metadata"]
    messages = parsed_components["meta_messages"]

    assert metadata.n_sections == 9
    assert np.allclose(
        metadata.section_x, np.array([-4.0, -3.0, -2.0, -1.0, -0.0, 1.0, 2.0, 3.0, 4.0])
    )
    assert metadata.section_y.shape == (9, 10)
    assert metadata.section_z.shape == (9, 10)
    assert np.all(metadata.section_n_points == 10)
    assert np.isclose(metadata.water_depth, 1000000.75)
    assert len(messages.errors) == 0


def test_responsefunctions_dimensions(parsed_components) -> None:
    response = parsed_components["response_ds"]
    messages = parsed_components["response_messages"]

    assert response.sizes["wavelength"] == 14
    assert response.sizes["forward_speed"] == 1
    assert response.sizes["wave_direction"] == 2
    assert response.sizes["influenced_dof"] == 6
    assert "rao_magnitude" in response.data_vars
    assert "rao_rotation_normalized" in response.data_vars
    assert len(messages.errors) == 0


def test_sectionresults_dimensions(parsed_components) -> None:
    section = parsed_components["section_ds"]

    assert section.sizes["section"] == 9
    assert section.sizes["omega"] == 52
    assert section.sizes["influenced_dof"] == 6
    assert section.sizes["radiating_dof"] == 6
    assert section.sizes["point"] == 10
    assert "added_mass" in section.data_vars
    assert "radiation_damping" in section.data_vars


def test_parse_pdstrip_folder_merged_output(parsed_folder) -> None:
    dataset = parsed_folder["dataset"]
    messages = parsed_folder["messages"]

    assert dataset.sizes["section"] == 9
    assert dataset.sizes["omega"] == 52
    assert dataset.sizes["wavelength"] == 14
    assert dataset.sizes["wave_direction"] == 2
    assert dataset["added_mass"].shape == (9, 52, 6, 6)
    assert dataset["radiation_damping"].shape == (9, 52, 6, 6)
    assert "x" in dataset
    assert "y" in dataset
    assert "z" in dataset
    assert messages["errors"] == []


def test_notebook_relevant_heave_roll_selection(parsed_folder) -> None:
    dataset = parsed_folder["dataset"]

    heave = (
        dataset["added_mass"]
        .integrate("section")
        .sel(influenced_dof="Heave", radiating_dof="Heave")
    )
    roll = (
        dataset["added_mass"]
        .integrate("section")
        .sel(influenced_dof="Roll", radiating_dof="Roll")
    )

    assert heave.sizes["omega"] == 52
    assert roll.sizes["omega"] == 52
    assert np.isfinite(heave.values).all()
    assert np.isfinite(roll.values).all()


def test_parse_pdstrip_folder_skips_empty_responsefunctions(
    pdstrip_fixture_dir, tmp_path
) -> None:
    for name in ["pdstrip.out", "sectionresults"]:
        shutil.copyfile(pdstrip_fixture_dir / name, tmp_path / name)
    (tmp_path / "responsefunctions").write_text("", encoding="utf-8")

    parsed = pdstripy.parse_pdstrip_folder(tmp_path)

    dataset = parsed["dataset"]
    response_ds = parsed["component_datasets"]["responsefunctions"]
    warnings = parsed["messages"]["warnings"]

    assert "rao_magnitude" not in dataset.data_vars
    assert "rao_rotation_normalized" not in dataset.data_vars
    assert len(response_ds.data_vars) == 0
    assert any("responsefunctions file is empty" in warning for warning in warnings)
    assert "added_mass" in dataset.data_vars
