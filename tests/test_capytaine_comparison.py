from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.slow
@pytest.mark.capytaine
def test_capytaine_vs_pdstrip_heave_added_mass_trend(parsed_folder) -> None:
    cpy = pytest.importorskip("capytaine")

    mesh_path = Path(__file__).resolve().parent.parent / "spheroid" / "spheroid_6_1.stl"
    if not mesh_path.exists():
        pytest.skip(f"Missing mesh required for Capytaine comparison: {mesh_path}")

    mesh = cpy.load_mesh(str(mesh_path))
    hull_mesh, lid_mesh = mesh.extract_lid()
    body = cpy.FloatingBody(
        mesh=hull_mesh,
        name="6:1 spheroid",
        center_of_mass=[0.0, 0.0, -0.75],
        lid_mesh=lid_mesh,
    )
    body.add_all_rigid_body_dofs()
    body.inertia_matrix = body.compute_rigid_body_inertia()
    body.compute_hydrostatics()

    rho = 1000.0
    omega = np.array(
        [
            0.361663,
            0.511468,
            0.636773,
            0.808703,
            1.022937,
            1.252837,
            1.576452,
            2.013654,
            2.479435,
            2.847736,
            3.234811,
            3.616628,
            4.043513,
            4.502666,
            4.985178,
            5.602857,
            6.264184,
            6.86207,
            7.672027,
            8.858894,
            10.22937,
            11.719215,
            13.288341,
            14.91174,
            18.083141,
            22.873566,
        ],
        dtype=float,
    )
    problems = [
        cpy.RadiationProblem(
            body=body, radiating_dof=dof, omega=w, rho=rho, forward_speed=0.0
        )
        for dof in body.dofs
        for w in omega
    ]

    solver = cpy.BEMSolver()
    results = solver.solve_all(problems, n_jobs=1, keep_details=False)
    ds_cpy = cpy.assemble_dataset(results)

    ds_pd = parsed_folder["dataset"]
    pd_heave = (
        ds_pd["added_mass"]
        .integrate("section")
        .sel(influenced_dof="Heave", radiating_dof="Heave")
    )
    pd_interp = pd_heave.interp(omega=omega)

    cpy_heave = ds_cpy["added_mass"].sel(influenced_dof="Heave", radiating_dof="Heave")
    assert cpy_heave.sizes["omega"] == omega.size

    ratio = cpy_heave.values / pd_interp.values
    assert np.allclose(cpy_heave.values, pd_interp.values, rtol=1e0)
