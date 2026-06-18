import numpy as np
import pytest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, LineCollection

from yaiv.spectrum import Spectrum, ElectronBands, PhononBands, Density, _Has_lattice
from yaiv.defaults.config import ureg
from yaiv import utils as ut
from yaiv import grep
from conftest import quantity_allclose

# Force non-interactive backend for headless testing
matplotlib.use("Agg")

# The same matrix and ids you used in your grep tests
FILES = [
    ("qe/results/Si.scf.pwo", "qe_scf_out"),
    ("qe/results/Si.scf.pwi", "qe_scf_in"),
    ("qe/results/Si.ph.pwo", "qe_ph_out"),
    ("qe/results/Si.bands.pwi", "qe_bands_in"),
    ("qe/results/matdyn.in", "matdyn_in"),
    ("qe/results/Si.dyn1", "qe_dyn"),
    ("qe/results/Si.freq", "qe_freq_out"),
    ("qe/results/Si.proj.pwo", "qe_proj_out"),
    ("qe/results/bands.xml", "qe_xml"),
    ("vasp/RESULTS/PROCAR", "procar"),
    ("vasp/RESULTS/OUTCAR_BS", "outcar"),
    ("vasp/RESULTS/EIGENVAL_BS", "eigenval"),
    ("vasp/KPATH", "kpath"),
    ("vasp/POSCAR", "poscar"),
]
IDS = [f for f, _ in FILES]

# Supported kinds per Spectrum subclass path
SUPPORTED_ELECTRON_BANDS = {"qe_xml", "qe_scf_out", "outcar", "eigenval"}
SUPPORTED_PHONON_BANDS = {"qe_freq_out"}


def _first_existing_by_kind(data_dir, require, kinds):
    """
    Helper to pick the first file from FILES in the provided kinds.
    Skips if none found.
    """
    for fname, kind in FILES:
        if kind in kinds:
            f = data_dir / fname
            if f.exists():
                return f, kind
    pytest.skip(f"No test data found for kinds: {kinds}")


def test_Has_lattice():
    # Start with real space
    lattice = np.diag([2.0, 3.0, 4.0]) * ureg.angstrom
    expected_k_lattice = ut.reciprocal_basis(lattice)
    has_lattice = _Has_lattice(lattice=lattice)

    quantity_allclose(has_lattice.lattice, lattice)
    quantity_allclose(has_lattice.k_lattice, expected_k_lattice)
    quantity_allclose(has_lattice.volume, 24.0 * ureg.ang**3)
    assert has_lattice.lattice.check(ureg.angstrom)
    assert has_lattice.k_lattice.check(ureg._2pi / ureg.angstrom)

    # Start with k-space
    k_lattice = np.diag([2.0, 3.0, 4.0]) * (1 / ureg.angstrom)
    expected_lattice = ut.reciprocal_basis(k_lattice)
    has_lattice = _Has_lattice(k_lattice=k_lattice)
    quantity_allclose(has_lattice.volume, (1 / 24) * ureg("_2pi * ang") ** 3)

    quantity_allclose(has_lattice.lattice, expected_lattice)
    quantity_allclose(has_lattice.k_lattice, k_lattice)

    # Test sync between lattice and k_lattice uppon change
    new_lattice = np.eye(3) * 5.0 * ureg.angstrom
    expected_new_k_lattice = ut.reciprocal_basis(new_lattice)
    has_lattice.lattice = new_lattice
    quantity_allclose(has_lattice.lattice, new_lattice)
    quantity_allclose(has_lattice.k_lattice, expected_new_k_lattice)


@pytest.mark.parametrize("fname, kind", FILES, ids=IDS)
def test_ElectronBands_construction(data_dir, require, fname, kind):
    f = data_dir / fname
    if kind in SUPPORTED_ELECTRON_BANDS:
        require(f, f"Missing test data: {fname}")

        # Build
        bands = ElectronBands(str(f))

        # Core arrays and units
        assert hasattr(bands, "eigenvalues")
        assert hasattr(bands, "kpoints")
        assert bands.eigenvalues.check(
            ureg.eV
        )  # energies are in eV in grep.kpointsEnergies
        assert bands.kpoints.check(ureg._2pi / ureg.crystal) or bands.kpoints.check(
            ureg._2pi / ureg.bohr
        )
        assert bands.eigenvalues.magnitude.ndim == 2
        assert bands.kpoints.magnitude.ndim == 2

        # Electron info
        # Some files may not have fermi available -> ElectronBands sets None
        assert isinstance(bands.electron_num, int)
        assert (bands.fermi is None) or hasattr(bands.fermi, "units")

        # Lattice might be None for some inputs
        if bands.lattice is not None:
            assert bands.lattice.shape == (3, 3)
            assert hasattr(bands.lattice, "units")
    else:
        with pytest.raises(NotImplementedError):
            ElectronBands(str(f))


def test_ElectronBands_plot_structure(data_dir, require):
    # Pick a supported file that exists (any kind ElectronBands supports)
    f, kind = _first_existing_by_kind(data_dir, require, SUPPORTED_ELECTRON_BANDS)
    bands = ElectronBands(str(f))

    # Smoke test plot
    ax = bands.plot(color="C0", linewidth=1.0)
    assert ax is not None
    lines = ax.get_lines()
    # The plot() draws all bands; ensure we have lines
    assert len(lines) == bands.eigenvalues.shape[1]
    # x limits normalized to [0,1]
    x0, x1 = ax.get_xlim()
    assert x0 == pytest.approx(0.0)
    assert x1 == pytest.approx(1.0)

    # Fat-band plotting
    W = np.abs(bands.eigenvalues.magnitude)  # any positive weights of matching shape
    ax2, coll = bands.plot_fat(weights=W, ax=None, cmap="viridis", size_change=True)
    assert isinstance(coll, PathCollection)

    # Color-gradient line plotting
    ax3, line = bands.plot_color(weights=W, ax=None, cmap="viridis")
    assert isinstance(line, LineCollection)


def test_ElectronBands_dos_compute_and_integrate(data_dir, require):
    f, kind = _first_existing_by_kind(data_dir, require, SUPPORTED_ELECTRON_BANDS)
    print(str(f))
    bands = ElectronBands(str(f))

    # Compute DOS with default args
    bands.get_DOS(cutoff_sigmas=5)
    assert bands.DOS is not None
    assert hasattr(bands.DOS, "grid") and hasattr(bands.DOS, "density")
    # Units: vgrid ~ energy units; DOS ~ 1/energy
    assert bands.DOS.grid.check(ureg.eV)
    assert bands.DOS.density.check(1 / ureg.eV)
    assert bands.DOS.grid.shape == bands.DOS.density.shape

    # Integrate full DOS range
    val, err = bands.DOS.integrate(limit=bands.DOS.grid[-1] * 1.1)
    assert np.isclose(val, bands.eigenvalues.shape[1])

    # Plot DOS (smoke)
    ax = bands.DOS.plot(color="C1")
    assert ax is not None


@pytest.mark.parametrize("fname, kind", FILES, ids=IDS)
def test_PhononBands_construction(data_dir, require, fname, kind):
    f = data_dir / fname
    if kind in SUPPORTED_PHONON_BANDS:
        require(f, f"Missing test data: {fname}")
        ph = PhononBands(str(f))
        assert ph.eigenvalues.check(ureg.c / ureg.cm)
        assert ph.kpoints.check(ureg._2pi / ureg.alat)
        assert ph.eigenvalues.magnitude.ndim == 2
        assert ph.kpoints.magnitude.ndim == 2

        if ph.lattice is not None:
            assert ph.lattice.shape == (3, 3)
            assert hasattr(ph.lattice, "units")
    else:
        with pytest.raises(NotImplementedError):
            PhononBands(str(f))


def test_PhononBands_plot_and_dos(data_dir, require):
    f, kind = _first_existing_by_kind(data_dir, require, SUPPORTED_PHONON_BANDS)
    ph = PhononBands(str(f))

    # plot (smoke)
    ax = ph.plot(color="C2")
    assert ax is not None

    # DOS compute + plot (phonon DOS often plotted, use defaults)
    ph.get_DOS()
    assert ph.DOS.grid.check(ureg.c / ureg.cm)
    assert ph.DOS.density.check(1 / (ureg.c / ureg.cm))

    ax2 = ph.DOS.plot(color="C3")
    assert ax2 is not None


def test_get_1Dkpath_basic_and_patched():
    # Simple 1D path in cartesian units, with a jump to test "patched"
    kpts = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.0, 0.0],
            [0.02, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # a big jump (discontinuity)
            [10.01, 0.0, 0.0],
        ]
    ) * (ureg._2pi / ureg.angstrom)

    spec = Spectrum(eigenvalues=np.zeros((len(kpts), 2)) * ureg.eV, kpoints=kpts)

    kpath_unpatched = spec.get_1Dkpath(patched=False)
    kpath_patched = spec.get_1Dkpath(patched=True)

    # Both must be non-decreasing and end positive
    assert np.all(np.diff(kpath_unpatched.magnitude) >= 0)
    assert np.all(np.diff(kpath_patched.magnitude) >= 0)
    assert kpath_unpatched[-1] > 0 * kpath_unpatched.units
    assert kpath_patched[-1] > 0 * kpath_patched.units

    # Patched should be strictly smaller than unpatched due to zeroing the large jump
    assert kpath_patched[-1].magnitude < kpath_unpatched[-1].magnitude


def test_dos_integrate_occ_states_inverse():
    # Synthetic DOS: a Gaussian centered at 0 with unit area
    x = np.linspace(-2.0, 2.0, 2001) * ureg.eV
    y = ut._normal_dist(x.magnitude, mean=0, sd=0.2, A=1) / ureg.eV
    d = Density(density=y, grid=x)

    # Target: half the area ~ 0.5 -> energy near 0
    e, err = d.integrate(amount=0.5)
    assert e.check(ureg.eV)
    assert np.isclose(0, e.magnitude)
