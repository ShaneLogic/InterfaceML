"""Exercise DFT archive generation and reported failures through the web API."""

import io
import zipfile

import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from interfaceml.web.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app()
    app.config.update(TESTING=True, UPLOAD_FOLDER=str(tmp_path))
    with app.test_client() as client:
        yield client


def upload_carbon(client, filename="carbon.vasp"):
    structure = Structure(Lattice.cubic(10), ["C", "C"], [[0.1, 0.1, 0.1], [0.1, 0.1, 0.4]])
    poscar = Poscar(structure, selective_dynamics=[[False] * 3, [True] * 3])
    response = client.post(
        "/api/upload", data={"file": (io.BytesIO(str(poscar).encode()), filename)}
    )
    assert response.status_code == 200
    return response.get_json()["file_info"]["filename"]


def test_standalone_archive_contains_cp2k_constraints(client):
    filename = upload_carbon(client)
    response = client.post("/api/pipeline/dft-prep", json={"structure_filename": filename})
    assert response.status_code == 200
    body = response.get_json()
    assert body["n_jobs"] == 1
    download = client.get(body["download_url"])
    assert download.status_code == 200
    with zipfile.ZipFile(io.BytesIO(download.data)) as archive:
        cp2k_name = next(name for name in archive.namelist() if name.endswith("/cp2k.inp"))
        cp2k = archive.read(cp2k_name).decode()
        assert "&FIXED_ATOMS" in cp2k
        assert "LIST 1" in cp2k
        assert any(name.endswith("/structure.xyz") for name in archive.namelist())
        assert any(name.endswith("/submit_all.sh") for name in archive.namelist())


def test_failed_preparation_is_not_reported_as_success(client):
    client.post("/api/upload", data={"file": (io.BytesIO(b"not a POSCAR"), "broken.vasp")})
    response = client.post("/api/pipeline/dft-prep", json={"structure_filename": "broken.vasp"})
    assert response.status_code == 422
    assert response.get_json()["status"] == "error"
    assert "download_url" not in response.get_json()


@pytest.mark.parametrize(
    "payload", [[], "invalid", {"structure_filename": "carbon.vasp", "dft": {"cutoff": -1}}]
)
def test_invalid_payload_is_a_client_error(client, payload):
    upload_carbon(client)
    response = client.post("/api/pipeline/dft-prep", json=payload)
    assert response.status_code == 400


def test_unknown_download_is_not_found(client):
    assert client.get("/api/pipeline/download/unknown").status_code == 404


def test_build_and_prepare_archive(client):
    crystal = Structure(
        Lattice.cubic(6.3),
        ["Cs", "Pb", "I", "I", "I"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )
    client.post(
        "/api/upload", data={"file": (io.BytesIO(str(Poscar(crystal)).encode()), "perovskite.vasp")}
    )
    molecule = upload_carbon(client)
    response = client.post(
        "/api/pipeline/build-and-prep",
        json={
            "perovskite_filename": "perovskite.vasp",
            "fullerene_filename": molecule,
            "supercell": "1x1",
            "slab_thickness": 8,
            "vacuum": 10,
            "fix_bottom_layers": 1,
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["n_interfaces"] == body["n_jobs"] == 1
    assert client.get(body["download_url"]).status_code == 200


def test_coherent_scan_and_build_use_matching_parameters(client):
    uploaded = {}
    for element in ("Si", "Ge"):
        crystal = Structure(Lattice.cubic(3), [element], [[0, 0, 0]])
        response = client.post(
            "/api/upload",
            data={"file": (io.BytesIO(str(Poscar(crystal)).encode()), f"{element}.vasp")},
        )
        uploaded[element] = response.get_json()["file_info"]["filepath"]
    payload = {
        "base_file": uploaded["Si"],
        "film_file": uploaded["Ge"],
        "max_area": 10,
        "matching_mode": "forward",
        "top_k": 1,
    }
    scan = client.post("/api/list-interface-terminations", json=payload)
    assert scan.status_code == 200
    terms = scan.get_json()
    assert terms["substrate_terminations"][0]["formula"] == "Si"
    assert terms["film_terminations"][0]["formula"] == "Ge"
    payload.update(
        substrate_termination=terms["substrate_terminations"][0]["label"],
        film_termination=terms["film_terminations"][0]["label"],
    )
    response = client.post("/api/build-interface", json=payload)
    assert response.status_code == 200
    result = response.get_json()
    assert result["n_built"] == 1
    assert result["matching_mode"] == "forward"
    assert result["strained_layer"] == "film"
    assert result["warnings"] == []
    assert client.get(result["results"][0]["download_url"]).status_code == 200

    response = client.post("/api/build-interface", json={**payload, "twist_deg": 15})
    assert response.get_json()["warnings"]
    assert "commensurability" in response.get_json()["warnings"][0]
    invalid = client.post(
        "/api/build-interface", json={**payload, "matching_mode": "substrate_only"}
    )
    assert invalid.status_code == 400
