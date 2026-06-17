from pathlib import Path
import numpy as np
import pytest

import spglib as spg

from yaiv import cell

FILES = [
    "vasp/POSCAR",
    "qe/results/Si.scf.pwo",
]
IDS = FILES[:]  # nicer test IDs


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_ase2spglib_and_spglib2ase(data_dir, require, fname):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))

    lat, pos, nums = c.spglib
    assert lat.shape == (3, 3)
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert nums.ndim == 1 and len(nums) == len(c.atoms)

    atoms2 = cell.spglib2ase((lat, pos, nums))
    # Lattice consistency
    np.testing.assert_allclose(
        np.asarray(atoms2.cell), np.asarray(c.atoms.cell), rtol=1e-6, atol=1e-6
    )
    # Atomic numbers match
    np.testing.assert_array_equal(
        atoms2.get_atomic_numbers(), c.atoms.get_atomic_numbers()
    )
    # Scaled positions match up to periodic wrapping
    sp1 = atoms2.get_scaled_positions() % 1.0
    sp0 = c.atoms.get_scaled_positions() % 1.0
    np.testing.assert_allclose(
        np.sort(sp1, axis=0), np.sort(sp0, axis=0), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_read_spg_direct(data_dir, require, fname):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    lat, pos, nums = cell.read_spg(str(f))
    assert lat.shape == (3, 3)
    assert pos.ndim == 2 and pos.shape[1] == 3
    assert nums.ndim == 1 and len(nums) == pos.shape[0]


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_Cell_from_file_and_tuple(data_dir, require, fname):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))
    assert hasattr(c, "atoms")
    assert hasattr(c, "spglib")

    lat, pos, nums = c.spglib
    assert lat.shape == (3, 3)
    assert pos.shape[1] == 3
    assert len(nums) == pos.shape[0]

    # dunder helpers
    assert len(c) == 3
    assert np.shape(c[0]) == (3, 3)
    lat2, pos2, nums2 = tuple(iter(c))
    np.testing.assert_allclose(lat2, lat)
    np.testing.assert_allclose(pos2, pos)
    np.testing.assert_array_equal(nums2, nums)

    # Construct from tuple and round-trip
    c2 = cell.Cell.from_spglib_tuple((lat, pos, nums))
    np.testing.assert_allclose(
        np.asarray(c2.atoms.cell), np.asarray(c.atoms.cell), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_Cell_repr(data_dir, require, fname):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))
    r = repr(c)
    assert "<Cell with" in r
    assert str(len(c.spglib[2])) in r


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_Cell_view(data_dir, require, fname):
    nv = pytest.importorskip("nglview")
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))
    widget = c.view()

    # Test widget type
    assert isinstance(widget, nv.NGLWidget)

    # Test default widget layout size
    assert widget.layout.width == "800px"
    assert widget.layout.height == "500px"

    # Test camera setting
    assert widget.camera == "orthographic"


@pytest.mark.parametrize("rep", ([1, 1, 1], [2, 1, 1], [2, 2, 1], [2, 2, 2]))
@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_get_supercell(data_dir, require, fname, rep):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))
    sc = c.get_supercell(rep)

    # Atom count scales
    n0 = len(c.spglib[2])
    n1 = len(sc.spglib[2])
    assert n1 == n0 * int(np.prod(rep))

    # Lattice scales along each vector; positions remain fractional in [0,1)
    lat0 = c.spglib[0].copy()
    lat1 = sc.spglib[0].copy()
    for i in range(3):
        np.testing.assert_allclose(lat1[i], lat0[i] * rep[i], rtol=1e-6, atol=1e-6)

    # Positions should be in [0,1)
    pos1 = sc.spglib[1]
    assert np.all((pos1 >= -1e-9) & (pos1 <= 1 + 1e-9))

    # Same space group:
    ds0 = spg.get_symmetry_dataset(c.spglib, _throw=True)
    ds1 = spg.get_symmetry_dataset(sc.spglib, _throw=True)
    assert (ds0.international, ds0.number) == (ds1.international, ds1.number)


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_get_wyckoff_positions(data_dir, require, fname):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))

    # Some structures or very low symprec may cause spglib to fail to find symmetry; skip then
    c.get_wyckoff_positions(symprec=cell.defaults.symprec)

    w = c.wyckoff
    assert hasattr(w, "symbols")
    assert hasattr(w, "labels")
    assert hasattr(w, "positions")
    assert hasattr(w, "indices")

    total = sum(len(idx) for idx in w.indices)
    assert total == len(c.spglib[2])


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_get_sym_info_prints(data_dir, require, fname, capsys):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))
    c.get_sym_info(symprec=cell.defaults.symprec)

    out = capsys.readouterr().out
    assert "SpaceGroup =" in out
    assert "ATOMS:" in out
    assert "Wyckoffs:" in out
    assert "Symmetry Operations:" in out


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_write_espresso_in_default(data_dir, require, tmp_path, fname):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))

    out = tmp_path / "out.pwi"
    c.write_espresso_in(str(out))
    assert out.exists()
    txt = out.read_text()
    # ASE's espresso-in writer produces typical geometry sections
    assert "ATOMIC_POSITIONS" in txt
    assert "CELL_PARAMETERS" in txt or "CELL_PARAMETERS" in txt.upper()


@pytest.mark.parametrize("fname", FILES, ids=IDS)
def test_write_espresso_in_with_template(data_dir, require, tmp_path, fname):
    f = data_dir / fname
    require(f, f"Missing test data: {fname}")
    c = cell.Cell.from_file(str(f))

    template = data_dir / "qe/results/Si.scf.pwi"
    out = tmp_path / "from_template.pwi"
    c.write_espresso_in(str(out), template=str(template))
    assert out.exists()
    txt = out.read_text()
    assert "ATOMIC_POSITIONS" in txt
    assert "CELL_PARAMETERS" in txt
    assert "ibrav = 0" in txt or "ibrav=0" in txt
    assert "pseudo_dir = '$PSEUDO_DIR'" in txt
    assert "outdir = './tmp'" in txt


def test_rot_name_identity_and_rotations():
    E = np.eye(3)

    # Identity -> E
    sym = cell._rot_name(E, E)
    assert sym.label == "E"
    assert sym.order == 1
    assert sym.det == pytest.approx(1.0)
    assert sym.direction_crystal is None

    # Proper 180° around z: diag(-1,-1,1) => C2
    C2z = np.diag([-1.0, -1.0, 1.0])
    sym2 = cell._rot_name(C2z, np.eye(3))
    assert sym2.label == "C2"
    assert sym2.det == pytest.approx(1.0)
    assert sym2.order >= 2
    axis = np.abs(sym2.direction_crystal / np.linalg.norm(sym2.direction_crystal))
    np.testing.assert_allclose(axis, np.array([0.0, 0.0, 1.0]), atol=1e-6)

    # Inversion: -I => "I"
    inv = -np.eye(3)
    sym3 = cell._rot_name(inv, np.eye(3))
    assert sym3.label == "I"
    assert sym3.det == pytest.approx(-1.0)
    assert sym3.direction_crystal is None
