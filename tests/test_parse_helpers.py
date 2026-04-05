from __future__ import annotations

import numpy as np

import pdstripy


def test_parse_floats_handles_scientific_notation() -> None:
    values = pdstripy._parse_floats(" 1.0 -2.5 3.2E+01 -4.5e-2 ")
    assert values == [1.0, -2.5, 32.0, -0.045]


def test_unique_preserve_drops_near_duplicates() -> None:
    unique = pdstripy._unique_preserve([1.0, 1.0 + 1e-10, 2.0, 2.0], atol=1e-9)
    assert np.allclose(unique, np.array([1.0, 2.0]))


def test_is_section_header_accepts_expected_format() -> None:
    assert pdstripy._is_section_header("0.361662805 1 0.00000000")


def test_is_section_header_rejects_complex_line() -> None:
    assert not pdstripy._is_section_header("(24.3,-0.01) (0.0,0.0)")
