from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np
import xarray as xr


LOGGER = logging.getLogger(__name__)


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?")
COMPLEX_PAIR_RE = re.compile(
    r"\(\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)\s*\)"
)


@dataclass
class ParseMessages:
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class PDStripMetadata:
    n_sections: int
    section_x: np.ndarray
    cases: dict[str, np.ndarray]
    water_depth: float | None = None
    section_y: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    section_z: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    section_n_points: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=int)
    )


def _parse_floats(text: str) -> list[float]:
    return [float(m.group(0)) for m in FLOAT_RE.finditer(text)]


def _unique_preserve(values: list[float], atol: float = 1e-9) -> np.ndarray:
    unique_vals: list[float] = []
    for value in values:
        if not any(
            np.isclose(value, seen, atol=atol, rtol=0.0) for seen in unique_vals
        ):
            unique_vals.append(value)
    return np.asarray(unique_vals, dtype=float)


def _apply_common_coordinate_attrs(dataset: xr.Dataset) -> xr.Dataset:
    if "omega" in dataset.coords:
        dataset.coords["omega"].attrs.update(
            {
                "long_name": "Radial frequency",
                "units": "rad/s",
            }
        )
    if "wavelength" in dataset.coords:
        dataset.coords["wavelength"].attrs.update(
            {
                "long_name": "Wavelength",
                "units": "m",
            }
        )
    if "wave_direction" in dataset.coords:
        dataset.coords["wave_direction"].attrs.update(
            {
                "long_name": "Wave direction",
                "units": "rad",
            }
        )
    return dataset


def parse_pdstrip_out(file_path: Path) -> tuple[PDStripMetadata, ParseMessages]:
    messages = ParseMessages()

    if not file_path.exists():
        raise FileNotFoundError(f"Missing pdstrip.out file: {file_path}")

    n_sections = None
    section_x: list[float] = []
    section_y_lists: list[list[float]] = []
    section_z_lists: list[list[float]] = []
    omega: list[float] = []
    omega_encounter: list[float] = []
    wavelengths: list[float] = []
    wave_numbers: list[float] = []
    wave_angles_deg: list[float] = []
    forward_speeds: list[float] = []
    z_still_waterline: float | None = None
    z_water_bottom: float | None = None

    section_pattern = re.compile(
        r"Section\s+no\.\s+\d+\s+at\s+x=\s*([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
    )
    nsec_pattern = re.compile(r"Number of sections\s+(\d+)")
    wave_pattern = re.compile(
        r"Wave circ\. frequency\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
        r"\s+encounter frequ\.\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
        r"\s+wave length\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
        r"\s+wave number\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
        r"\s+wave angle\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
    )
    speed_pattern = re.compile(r"^\s*speed\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)")
    still_waterline_pattern = re.compile(
        r"z of still waterline\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
    )
    water_bottom_pattern = re.compile(
        r"z of water bottom\s+([-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?)"
    )

    current_section_idx: int | None = None

    for raw_line in file_path.read_text().splitlines():
        line = raw_line.rstrip("\n")

        nsec_match = nsec_pattern.search(line)
        if nsec_match:
            n_sections = int(nsec_match.group(1))

        still_waterline_match = still_waterline_pattern.search(line)
        if still_waterline_match:
            z_still_waterline = float(still_waterline_match.group(1))

        water_bottom_match = water_bottom_pattern.search(line)
        if water_bottom_match:
            z_water_bottom = float(water_bottom_match.group(1))

        sec_match = section_pattern.search(line)
        if sec_match:
            section_x.append(float(sec_match.group(1)))
            section_y_lists.append([])
            section_z_lists.append([])
            current_section_idx = len(section_x) - 1

        if current_section_idx is not None:
            stripped = line.strip()
            if stripped.startswith("y:"):
                section_y_lists[current_section_idx].extend(_parse_floats(stripped[2:]))
            elif stripped.startswith("z:"):
                section_z_lists[current_section_idx].extend(_parse_floats(stripped[2:]))

        wave_match = wave_pattern.search(line)
        if wave_match:
            omega.append(float(wave_match.group(1)))
            omega_encounter.append(float(wave_match.group(2)))
            wavelengths.append(float(wave_match.group(3)))
            wave_numbers.append(float(wave_match.group(4)))
            wave_angles_deg.append(float(wave_match.group(5)))

        speed_match = speed_pattern.search(line)
        if speed_match:
            forward_speeds.append(float(speed_match.group(1)))

    if n_sections is None:
        messages.errors.append("Could not find 'Number of sections' in pdstrip.out.")
        n_sections = len(section_x)

    if len(section_x) != n_sections:
        messages.warnings.append(
            f"Found {len(section_x)} section x-values but metadata says {n_sections} sections."
        )

    if not omega:
        messages.errors.append("No wave cases were parsed from pdstrip.out.")

    n_sections_parsed = len(section_x)
    section_n_points = np.zeros(n_sections_parsed, dtype=int)
    for idx in range(n_sections_parsed):
        ny = len(section_y_lists[idx])
        nz = len(section_z_lists[idx])
        if ny != nz:
            messages.warnings.append(
                f"Section {idx+1} y/z point count mismatch (y={ny}, z={nz}); truncating to shortest length."
            )
        section_n_points[idx] = min(ny, nz)

    max_points = int(section_n_points.max()) if section_n_points.size else 0
    section_y = np.full((n_sections_parsed, max_points), np.nan, dtype=float)
    section_z = np.full((n_sections_parsed, max_points), np.nan, dtype=float)
    for idx in range(n_sections_parsed):
        npts = int(section_n_points[idx])
        if npts == 0:
            continue
        section_y[idx, :npts] = np.asarray(section_y_lists[idx][:npts], dtype=float)
        section_z[idx, :npts] = np.asarray(section_z_lists[idx][:npts], dtype=float)

    cases = {
        "omega": np.asarray(omega, dtype=float),
        "omega_encounter": np.asarray(omega_encounter, dtype=float),
        "wavelength": np.asarray(wavelengths, dtype=float),
        "wave_number": np.asarray(wave_numbers, dtype=float),
        "wave_angle_deg": np.asarray(wave_angles_deg, dtype=float),
        "forward_speed": np.asarray(forward_speeds, dtype=float),
        "omega_unique": _unique_preserve(omega),
        "omega_encounter_unique": _unique_preserve(omega_encounter),
        "wavelength_unique": _unique_preserve(wavelengths),
        "wave_number_unique": _unique_preserve(wave_numbers),
        "wave_angle_deg_unique": _unique_preserve(wave_angles_deg),
        "forward_speed_unique": _unique_preserve(forward_speeds),
    }

    water_depth = None
    if z_still_waterline is not None and z_water_bottom is not None:
        water_depth = z_still_waterline - z_water_bottom

    metadata = PDStripMetadata(
        n_sections=n_sections,
        section_x=np.asarray(section_x, dtype=float),
        cases=cases,
        water_depth=water_depth,
        section_y=section_y,
        section_z=section_z,
        section_n_points=section_n_points,
    )
    return metadata, messages


def parse_responsefunctions(
    file_path: Path, metadata: PDStripMetadata | None = None
) -> tuple[xr.Dataset, ParseMessages]:
    messages = ParseMessages()

    if not file_path.exists():
        raise FileNotFoundError(f"Missing responsefunctions file: {file_path}")

    raw_text = file_path.read_text()
    if not raw_text.strip():
        warning = f"responsefunctions file is empty: {file_path}; skipping response-function variables."
        LOGGER.warning(warning)
        messages.warnings.append(warning)
        return xr.Dataset(), messages

    lines = raw_text.splitlines()
    if len(lines) < 3:
        raise ValueError(f"responsefunctions file appears truncated: {file_path}")

    title = lines[0].strip()

    header2 = lines[1].split()
    if len(header2) < 5:
        raise ValueError(
            "Second header line in responsefunctions has unexpected format."
        )

    nb = int(float(header2[0]))
    intersection_flag = header2[1]
    g = float(header2[2])
    rho = float(header2[3])
    rml = float(header2[4])

    header3_vals = _parse_floats(lines[2])
    if len(header3_vals) < 4:
        raise ValueError(
            "Third header line in responsefunctions has unexpected format."
        )

    cursor = 0
    nom = int(round(header3_vals[cursor]))
    cursor += 1
    wavelengths = np.asarray(header3_vals[cursor : cursor + nom], dtype=float)
    cursor += nom

    nv = int(round(header3_vals[cursor]))
    cursor += 1
    forward_speeds = np.asarray(header3_vals[cursor : cursor + nv], dtype=float)
    cursor += nv

    nmu = int(round(header3_vals[cursor]))
    cursor += 1
    mu_rad = np.asarray(header3_vals[cursor : cursor + nmu], dtype=float)
    cursor += nmu

    nse = int(round(header3_vals[cursor]))
    cursor += 1
    _section_x = np.asarray(header3_vals[cursor : cursor + nse], dtype=float)

    n_lines_expected = nom * nv * nmu
    response_rows: list[list[float]] = []

    for line in lines[3:]:
        vals = _parse_floats(line)
        if len(vals) >= 6:
            response_rows.append(vals[:6])
        if len(response_rows) >= n_lines_expected:
            break

    if len(response_rows) < n_lines_expected:
        messages.warnings.append(
            f"responsefunctions rows parsed={len(response_rows)} but expected={n_lines_expected}."
        )

    if not response_rows:
        raise ValueError("No response rows parsed from responsefunctions.")

    response_arr = np.asarray(response_rows, dtype=float)
    response_magnitude = np.sqrt(np.maximum(response_arr, 0.0))

    # MATLAB equivalent: k = 2*pi/lambda, repeated over speed and encounter angle loops.
    k_lambda = 2.0 * np.pi / wavelengths
    k = np.repeat(k_lambda, nv * nmu)
    if len(k) != response_magnitude.shape[0]:
        if len(k) > response_magnitude.shape[0]:
            k = k[: response_magnitude.shape[0]]
        else:
            k = np.pad(k, (0, response_magnitude.shape[0] - len(k)), mode="edge")
            messages.warnings.append(
                "Wave slope array padded due to row-count mismatch."
            )

    response_rotation_normalized = response_magnitude.copy()
    response_rotation_normalized[:, 3:6] = (
        response_rotation_normalized[:, 3:6] / k[:, None]
    )

    dof_names = np.asarray(
        ["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"], dtype=object
    )
    rows = response_magnitude.shape[0]
    nominal_rows = nom * nv * nmu

    if rows == nominal_rows:
        shape = (nom, nv, nmu, 6)
        response_magnitude_4d = response_magnitude.reshape(shape)
        response_rotation_normalized_4d = response_rotation_normalized.reshape(shape)

        dataset = xr.Dataset(
            data_vars={
                "rao_magnitude": (
                    ("wavelength", "forward_speed", "wave_direction", "influenced_dof"),
                    response_magnitude_4d,
                ),
                "rao_rotation_normalized": (
                    ("wavelength", "forward_speed", "wave_direction", "influenced_dof"),
                    response_rotation_normalized_4d,
                ),
            },
            coords={
                "wavelength": wavelengths,
                "forward_speed": forward_speeds,
                "wave_direction": mu_rad,
                "influenced_dof": dof_names,
            },
            attrs={
                "title": title,
                "nb": nb,
                "intersection_flag": intersection_flag,
                "g": g,
                "rho": rho,
                "rml": rml,
                "nom": nom,
                "nv": nv,
                "nmu": nmu,
                "nse": nse,
            },
        )
    else:
        messages.warnings.append(
            "responsefunctions could not be reshaped to [wavelength, forward_speed, wave_direction, influenced_dof]; exposing case dimension."
        )
        dataset = xr.Dataset(
            data_vars={
                "rao_magnitude": (("case", "influenced_dof"), response_magnitude),
                "rao_rotation_normalized": (
                    ("case", "influenced_dof"),
                    response_rotation_normalized,
                ),
            },
            coords={
                "case": np.arange(rows),
                "influenced_dof": dof_names,
            },
            attrs={
                "title": title,
                "nb": nb,
                "intersection_flag": intersection_flag,
                "g": g,
                "rho": rho,
                "rml": rml,
                "nom": nom,
                "nv": nv,
                "nmu": nmu,
                "nse": nse,
            },
        )

    if metadata is not None:
        if metadata.n_sections != nse:
            messages.warnings.append(
                f"Section count mismatch: pdstrip.out={metadata.n_sections}, responsefunctions={nse}."
            )

    dataset = _apply_common_coordinate_attrs(dataset)
    return dataset, messages


def _is_section_header(line: str) -> bool:
    if "(" in line or ")" in line:
        return False
    vals = _parse_floats(line)
    if len(vals) < 3:
        return False
    nmu = vals[1]
    return abs(nmu - round(nmu)) < 1e-12 and 1 <= int(round(nmu)) <= 64


def parse_sectionresults(
    file_path: Path,
    metadata: PDStripMetadata | None = None,
) -> tuple[xr.Dataset, ParseMessages]:
    messages = ParseMessages()

    if not file_path.exists():
        raise FileNotFoundError(f"Missing sectionresults file: {file_path}")

    lines = file_path.read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"sectionresults file appears truncated: {file_path}")

    title = lines[0].strip()
    second_line_vals = _parse_floats(lines[1])
    if not second_line_vals:
        raise ValueError(
            "Could not parse frequency block count from sectionresults header."
        )
    n_freq_declared = int(round(second_line_vals[0]))

    blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None

    for raw_line in lines[2:]:
        line = raw_line.strip()
        if not line:
            continue

        if _is_section_header(line):
            if current_block is not None:
                blocks.append(current_block)

            vals = _parse_floats(line)
            current_block = {
                "omega": float(vals[0]),
                "nmu": int(round(vals[1])),
                "angle0": float(vals[2]),
                "complex": [],
            }
            continue

        if current_block is None:
            continue

        for real_str, imag_str in COMPLEX_PAIR_RE.findall(line):
            current_block["complex"].append(complex(float(real_str), float(imag_str)))

    if current_block is not None:
        blocks.append(current_block)

    if not blocks:
        raise ValueError("No sectionresults data blocks parsed.")

    n_sections = metadata.n_sections if metadata is not None else None
    if n_sections is None:
        if len(blocks) % n_freq_declared != 0:
            messages.warnings.append(
                "Could not infer number of sections exactly from blocks/frequency count; assuming 1 section."
            )
            n_sections = 1
        else:
            n_sections = len(blocks) // n_freq_declared

    if n_sections * n_freq_declared != len(blocks):
        messages.warnings.append(
            f"Block count mismatch: parsed blocks={len(blocks)}, expected={n_sections*n_freq_declared} from section/frequency counts."
        )

    nmu_max = max(block["nmu"] for block in blocks)
    dof6 = np.asarray(["Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"], dtype=object)

    added_mass = np.full((n_sections, n_freq_declared, 6, 6), np.nan, dtype=float)
    radiation_damping = np.full(
        (n_sections, n_freq_declared, 6, 6), np.nan, dtype=float
    )
    am_complex = np.full(
        (n_sections, n_freq_declared, 6, 6), np.nan + 1j * np.nan, dtype=np.complex128
    )
    diffraction_force = np.full(
        (n_sections, n_freq_declared, nmu_max, 6),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    froude_krylov_force = np.full(
        (n_sections, n_freq_declared, nmu_max, 6),
        np.nan + 1j * np.nan,
        dtype=np.complex128,
    )
    omega_grid = np.full((n_sections, n_freq_declared), np.nan, dtype=float)

    for idx, block in enumerate(blocks):
        sec_idx = idx // n_freq_declared
        freq_idx = idx % n_freq_declared
        if sec_idx >= n_sections or freq_idx >= n_freq_declared:
            continue

        omega = block["omega"]
        nmu = block["nmu"]
        data = block["complex"]
        omega_grid[sec_idx, freq_idx] = omega

        expected = 9 + 6 * nmu
        if len(data) < expected:
            messages.warnings.append(
                f"Block {idx} has {len(data)} complex entries but expected at least {expected}."
            )
            continue

        am_vals = np.asarray(data[:9], dtype=np.complex128).reshape(3, 3)
        # PDStrip section outputs carry sway/heave/roll blocks; map them to 6-DOF names.
        dof_map = [1, 2, 3]
        am_complex_block = am_complex[sec_idx, freq_idx]
        am_complex_block[np.ix_(dof_map, dof_map)] = am_vals
        am_complex[sec_idx, freq_idx] = am_complex_block
        if abs(omega) > 0.0:
            added_mass_block = added_mass[sec_idx, freq_idx]
            damping_block = radiation_damping[sec_idx, freq_idx]
            added_mass_block[np.ix_(dof_map, dof_map)] = am_vals.real / (omega**2)
            damping_block[np.ix_(dof_map, dof_map)] = am_vals.imag / (-omega)
            added_mass[sec_idx, freq_idx] = added_mass_block
            radiation_damping[sec_idx, freq_idx] = damping_block

        forces = np.asarray(data[9 : 9 + 6 * nmu], dtype=np.complex128).reshape(nmu, 6)
        diffraction_force_block = diffraction_force[sec_idx, freq_idx, :nmu]
        froude_krylov_force_block = froude_krylov_force[sec_idx, freq_idx, :nmu]
        diffraction_force_block[:, dof_map] = forces[:, :3]
        froude_krylov_force_block[:, dof_map] = forces[:, 3:6]
        diffraction_force[sec_idx, freq_idx, :nmu] = diffraction_force_block
        froude_krylov_force[sec_idx, freq_idx, :nmu] = froude_krylov_force_block

    section_coord = np.arange(1, n_sections + 1, dtype=int)
    if metadata is not None and len(metadata.section_x) >= n_sections:
        section_x = metadata.section_x[:n_sections]
    else:
        section_x = np.arange(n_sections, dtype=float)

    omega_first_section = omega_grid[0, :]
    if metadata is not None and metadata.cases["wave_angle_deg_unique"].size == nmu_max:
        wave_direction_coord = np.deg2rad(metadata.cases["wave_angle_deg_unique"])
    else:
        wave_direction_values = _unique_preserve([block["angle0"] for block in blocks])
        if wave_direction_values.size == nmu_max:
            if np.nanmax(np.abs(wave_direction_values)) > 2.0 * np.pi + 1e-9:
                wave_direction_coord = np.deg2rad(wave_direction_values)
            else:
                wave_direction_coord = wave_direction_values
        else:
            wave_direction_coord = np.arange(nmu_max, dtype=float)

    data_vars: dict[str, Any] = {
        "x": (("section",), section_x),
    }
    coords: dict[str, Any] = {
        "section": section_coord,
        "omega": omega_first_section,
        "influenced_dof": dof6,
        "radiating_dof": dof6,
        "wave_direction": wave_direction_coord,
    }

    if (
        metadata is not None
        and metadata.section_y.ndim == 2
        and metadata.section_z.ndim == 2
    ):
        n_point = min(metadata.section_y.shape[1], metadata.section_z.shape[1])
        if n_point > 0:
            data_vars["y"] = (
                ("section", "point"),
                metadata.section_y[:n_sections, :n_point],
            )
            data_vars["z"] = (
                ("section", "point"),
                metadata.section_z[:n_sections, :n_point],
            )
            data_vars["section_n_points"] = (
                ("section",),
                metadata.section_n_points[:n_sections],
            )
            coords["point"] = np.arange(n_point, dtype=int)

    data_vars.update(
        {
            "added_mass": (
                ("section", "omega", "influenced_dof", "radiating_dof"),
                added_mass,
            ),
            "radiation_damping": (
                ("section", "omega", "influenced_dof", "radiating_dof"),
                radiation_damping,
            ),
            "am_complex": (
                ("section", "omega", "influenced_dof", "radiating_dof"),
                am_complex,
            ),
            "diffraction_force": (
                ("section", "omega", "wave_direction", "influenced_dof"),
                diffraction_force,
            ),
            "Froude_Krylov_force": (
                ("section", "omega", "wave_direction", "influenced_dof"),
                froude_krylov_force,
            ),
        }
    )

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "title": title,
            "n_freq_declared": n_freq_declared,
        },
    )

    if "added_mass" in dataset.data_vars:
        dataset["added_mass"].attrs["long_name"] = "Added mass"
    if "radiation_damping" in dataset.data_vars:
        dataset["radiation_damping"].attrs["long_name"] = "Radiation damping"

    dataset = _apply_common_coordinate_attrs(dataset)
    return dataset, messages


def parse_pdstrip_folder(folder: str | Path = ".") -> dict[str, Any]:
    folder_path = Path(folder)

    metadata, meta_messages = parse_pdstrip_out(folder_path / "pdstrip.out")
    response_ds, response_messages = parse_responsefunctions(
        folder_path / "responsefunctions", metadata=metadata
    )
    section_ds, section_messages = parse_sectionresults(
        folder_path / "sectionresults", metadata=metadata
    )

    merged = xr.merge([response_ds, section_ds], compat="no_conflicts", join="outer")
    merged.attrs["source_folder"] = str(folder_path)
    merged.attrs["n_sections_pdstrip_out"] = int(metadata.n_sections)
    if metadata.water_depth is not None:
        merged.attrs["water_depth"] = float(metadata.water_depth)
    merged = _apply_common_coordinate_attrs(merged)

    all_warnings = (
        meta_messages.warnings + response_messages.warnings + section_messages.warnings
    )
    all_errors = (
        meta_messages.errors + response_messages.errors + section_messages.errors
    )

    return {
        "dataset": merged,
        "metadata": metadata,
        "messages": {
            "warnings": all_warnings,
            "errors": all_errors,
        },
        "component_datasets": {
            "responsefunctions": response_ds,
            "sectionresults": section_ds,
        },
    }
