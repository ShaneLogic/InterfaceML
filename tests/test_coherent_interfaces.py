"""Check the physical labels and geometry of coherent interface construction."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from interfaceml.core.adsorbate import molecular_unwrap
from interfaceml.core.interfaces_builder import (
    apply_registry_twist,
    build_coherent_interfaces,
    list_interface_terminations,
)


@pytest.fixture
def crystal_pair():
    substrate = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
    film = Structure(Lattice.cubic(3), ["Ge"], [[0, 0, 0]])
    return substrate, film


def test_termination_labels_follow_material_identity(crystal_pair):
    terms = list_interface_terminations(*crystal_pair, max_area=10)
    assert terms["n_zsl_matches"] > 0
    assert all(item["formula"] == "Si" for item in terms["substrate_terminations"])
    assert all(item["formula"] == "Ge" for item in terms["film_terminations"])


def test_explicit_termination_matches_materials(crystal_pair):
    terms = list_interface_terminations(*crystal_pair, max_area=10)
    labels = terms["substrate_terminations"] + terms["film_terminations"]
    substrate = next(item["label"] for item in labels if item["formula"] == "Si")
    film = next(item["label"] for item in labels if item["formula"] == "Ge")
    candidates = build_coherent_interfaces(
        *crystal_pair, termination=(substrate, film), max_area=10, top_k=1
    )
    assert len(candidates) == 1
    interface = candidates[0]["interface"]
    assert {interface[i].specie.symbol for i in interface.substrate_indices} == {"Si"}
    assert {interface[i].specie.symbol for i in interface.film_indices} == {"Ge"}
    assert candidates[0]["von_mises_strain"] == pytest.approx(0, abs=1e-10)


def test_registry_shift_preserves_film_labels_and_substrate(crystal_pair):
    interface = build_coherent_interfaces(*crystal_pair, max_area=10, top_k=1)[0]["interface"]
    shifted = apply_registry_twist(interface, xy_shift=(0.25, 0.1), xy_units="fractional")
    assert shifted.site_properties == interface.site_properties
    assert shifted.film_indices == interface.film_indices
    np.testing.assert_allclose(
        shifted.cart_coords[interface.substrate_indices],
        interface.cart_coords[interface.substrate_indices],
    )
    expected = 0.25 * interface.lattice.matrix[0] + 0.1 * interface.lattice.matrix[1]
    np.testing.assert_allclose(
        shifted.cart_coords[interface.film_indices] - interface.cart_coords[interface.film_indices],
        [expected] * len(interface.film_indices),
        atol=1e-12,
    )


def test_unwrap_preserves_site_properties():
    molecule = Structure(
        Lattice.cubic(10),
        ["C", "H"],
        [[0.5, 0.5, 0.95], [0.5, 0.5, 0.05]],
        site_properties={"selective_dynamics": [[False] * 3, [True] * 3]},
    )
    repaired = molecular_unwrap(molecule)
    assert repaired.site_properties == molecule.site_properties
    assert np.linalg.norm(repaired.cart_coords[0] - repaired.cart_coords[1]) == pytest.approx(1)
