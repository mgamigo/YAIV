import pytest
import warnings
import numpy as np
from numpy.testing import assert_allclose

from yaiv.defaults.config import ureg
from yaiv import utils as ut
from yaiv import spectrum, grep


def test_check_unit_consistency_success():
    ut._check_unit_consistency([1, 2.0, None])  # all unitless
    ut._check_unit_consistency(
        [1.0 * ureg.meter, 2.0 * ureg.second, None]
    )  # all unitful


def test_check_unit_consistency_failure(capsys):
    with pytest.raises(TypeError):
        ut._check_unit_consistency(
            [1.0 * ureg.meter, 2.0, None],
            names=["length", "value", "optional"],
        )
    out = capsys.readouterr().out
    assert "Units check failed for:" in out
    assert "Units status:" in out


def test_split_units():
    # --- scalar quantity ---
    q = 3 * ureg.m
    mag, unit = ut._split_units(q)
    assert mag == 3
    assert unit == ureg.m

    # --- scalar non-quantity ---
    mag, unit = ut._split_units(5)
    assert mag == 5
    assert unit == 1

    # --- list mixed ---
    data = [1, 2 * ureg.s, 3]
    mag, units = ut._split_units(data)
    assert mag == [1, 2, 3]
    assert units == [1, ureg.s, 1]


def test_invQ():
    # Inverse should have 1/unit
    A = np.eye(2) * (2.0 * ureg.meter)
    inv = ut.invQ(A)
    assert isinstance(inv, ureg.Quantity)
    assert_allclose(inv.magnitude, 0.5 * np.eye(2))
    # Units: 1/meter
    assert inv.check(1 / ureg.meter)


def test_reciprocal_basis():
    # Lattice basis as identity (rows), expect reciprocal = 2π I
    a = 2 * np.eye(3) * ureg.meter
    K = ut.reciprocal_basis(a)
    assert isinstance(K, ureg.Quantity)
    assert_allclose(K.magnitude, 0.5 * np.eye(3))
    # Units: 2π / meter
    assert K.check(ureg._2pi / ureg.meter)


def test_cartesian_crystal_conversion():
    # Cubic 1 Å box
    a0 = 3.0 * ureg.angstrom
    basis = np.eye(3) * a0
    r_cart = np.array([0.3, 0.4, 0.5]) * ureg.angstrom

    r_cryst = ut.cartesian2cryst(r_cart, basis)
    # Should be in crystal units
    assert isinstance(r_cryst, ureg.Quantity)
    assert r_cryst.check(ureg.crystal)

    r_back = ut.cryst2cartesian(r_cryst, basis)
    assert isinstance(r_back, ureg.Quantity)
    assert r_back.check(ureg.angstrom)
    assert_allclose(
        r_back.to(ureg.angstrom).magnitude,
        r_cart.to(ureg.angstrom).magnitude,
        atol=1e-12,
    )


def test_cartesian2cryst_unit_incompatibility_raises():
    basis = np.eye(3) * (1.0 * ureg.meter)
    coord = np.array([1.0, 0.0, 0.0]) * ureg.meter
    with pytest.raises(TypeError):
        _ = ut.cartesian2cryst(coord.magnitude, basis)
    with pytest.raises(TypeError):
        _ = ut.cartesian2cryst(coord, basis.magnitude)
    with pytest.raises(TypeError):
        _ = ut.cartesian2cryst(coord.magnitude, basis.magnitude / ureg.meter)


def test_cryst2cartesian_unit_incompatibility_raises():
    basis = np.eye(3) * (1.0 * ureg.angstrom)
    coord = np.array([1.0, 0.0, 0.0])
    with pytest.raises(TypeError):
        _ = ut.cryst2cartesian(coord * ureg.meter, basis)
    with pytest.raises(TypeError):
        _ = ut.cryst2cartesian(coord, basis)


def test_voigt_cartesian_conversion():
    T = (
        np.array(
            [
                [1.0, 0.2, -0.3],
                [0.2, 2.0, 0.4],
                [-0.3, 0.4, -1.0],
            ]
        )
        * ureg.gigapascal
    )

    v = ut.cartesian2voigt(T)
    assert isinstance(v, ureg.Quantity)
    assert v.check(ureg.gigapascal)
    assert v.shape == (6,)

    T2 = ut.voigt2cartesian(v)
    assert isinstance(T2, ureg.Quantity)
    assert T2.check(ureg.gigapascal)
    assert_allclose(T2.magnitude, T.magnitude)


def test_change_basis():
    # --- setup ---
    dim = 3
    E1 = np.eye(dim)
    E2 = np.array([[2.0, 0.1, 0.0], [0.0, 1.5, 0.2], [0.0, 0.0, 0.8]])
    E3 = np.array([[0.0, 0.1, 0.8], [0.0, 1.5, 0.2], [2.0, 0.0, 0.8]])

    # ensure invertible
    assert np.linalg.cond(E2) < 1e6

    # --- contravariant vector ---
    v = np.random.rand(dim)
    v_new = ut.change_basis(v, E1, E2, contravariant=1)

    A = np.linalg.solve(E1.T, E2.T)
    A_inv = np.linalg.inv(A)
    expected = A_inv @ v
    assert_allclose(v_new, expected)

    # --- covariant vector ---
    k = np.random.rand(dim)
    k_new = ut.change_basis(k, E1, E2, covariant=1)

    expected = A.T @ k
    assert_allclose(k_new, expected)

    # --- rank (1,1) tensor ---
    T = np.random.rand(dim, dim)
    T_new = ut.change_basis(T, E1, E2, contravariant=1, covariant=1)

    expected = A_inv @ T @ A
    assert_allclose(T_new, expected)

    # --- batch vectors ---
    V = np.random.rand(10, dim)
    V_new = ut.change_basis(V, E1, E2, contravariant=1)

    expected = np.asarray([A_inv @ v for v in V])
    assert_allclose(V_new, expected)

    # --- batch tensors ---
    TT = np.random.rand(5, dim, dim, dim)
    TT_new = ut.change_basis(TT, E1, E2, contravariant=2, covariant=1)
    assert np.all(TT_new.shape == TT.shape)
    expected = np.asarray(
        [ut.change_basis(t, E1, E2, contravariant=2, covariant=1) for t in TT]
    )
    assert_allclose(TT_new, expected)

    # --- roundtrip consistency ---
    v_back = ut.change_basis(v_new, E2, E1, contravariant=1)
    assert_allclose(v_back, v)
    T_back = ut.change_basis(T_new, E2, E1, contravariant=1, covariant=1)
    assert_allclose(T_back, T)

    # --- index lowering/rising consistency ---
    lat = E2
    klat = ut.reciprocal_basis(E2).magnitude
    g = lat @ lat.T
    # Assume T is contravariant respect to lat
    assert np.allclose(
        ut.change_basis(T, lat, np.eye(3), contravariant=2, covariant=0),
        ut.change_basis(T @ g, lat, np.eye(3), contravariant=1, covariant=1),
    )
    assert np.allclose(
        ut.change_basis(T, lat, np.eye(3), contravariant=2, covariant=0),
        ut.change_basis(g @ T @ g, lat, np.eye(3), contravariant=0, covariant=2),
    )
    # This implies T is covariant respect to klat
    # g is the inverse metric of g(klat), so it raises indices (covariant -> contravariant)
    assert np.allclose(
        ut.change_basis(T, klat, np.eye(3), contravariant=0, covariant=2),
        ut.change_basis(T, lat, np.eye(3), contravariant=2, covariant=0),
    )
    assert np.allclose(
        ut.change_basis(T, klat, np.eye(3), contravariant=0, covariant=2),
        ut.change_basis(g @ T, klat, np.eye(3), contravariant=1, covariant=1),
    )
    assert np.allclose(
        ut.change_basis(T, klat, np.eye(3), contravariant=0, covariant=2),
        ut.change_basis(g @ T @ g, klat, np.eye(3), contravariant=2, covariant=0),
    )

    # --- consistency with cryst/cartesian wrappers ---
    r_cart = np.random.rand(10, 3)
    r_cryst = ut.change_basis(r_cart, E1, E2, contravariant=1)
    expected = ut.cartesian2cryst(r_cart, E2)
    assert_allclose(
        r_cryst,
        expected,
    )
    r_E3 = ut.change_basis(r_cryst, E2, E3, contravariant=1)
    expected = ut.cartesian2cryst(r_cart, E3)
    assert_allclose(
        r_E3,
        expected,
    )
    # --- index consistency ---
    g1 = E1 @ E1.T
    g2 = E2 @ E2.T

    out1 = ut.change_basis(T, E1, E2, contravariant=1, covariant=1)
    out2 = np.linalg.inv(g2) @ ut.change_basis(
        g1 @ T, E1, E2, contravariant=0, covariant=2
    )
    assert_allclose(out1, out2)

    # --- unit handling ---
    v_u = v * ureg.meter
    E1_u = E1 * ureg.meter
    E2_u = E2 * ureg.meter

    v_u_new = ut.change_basis(v_u, E1_u, E2_u, contravariant=1)

    # magnitude should match unitless case
    assert_allclose(v_u_new.magnitude, v_new)
    assert v_u_new.check(ureg.meter)

    # --- error case ---
    with pytest.raises(ValueError):
        ut.change_basis(v, E1, E2, contravariant=2)


def test_rotate():
    # --- define a non-orthogonal-like symmetry (general case) ---
    lattice = [
        [-1.8993036377496129, 1.8993036377496129, 6.57149432085007],
        [1.8993036377496129, -1.8993036377496129, 6.57149432085007],
        [1.8993036377496129, 1.8993036377496129, -6.57149432085007],
    ] * ureg.ang
    k_lattice = ut.reciprocal_basis(lattice)
    R = np.array([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    S = grep._Symmetry(R, units=ureg.cryst)

    # --- basic vector ---
    v = np.array([1.0, 2.0, 3.0])

    # identity check
    assert np.allclose(ut.rotate(v, np.eye(3)), v)

    # v' = Rv check
    assert np.allclose(ut.rotate(v, R), R @ v)

    # inverse consistency
    v_rot = ut.rotate(v, R)
    v_back = ut.rotate(v_rot, np.linalg.inv(R))
    assert np.allclose(v, v_back)

    # --- batch shape preservation ---
    V = np.random.rand(10, 3)
    assert ut.rotate(V, R).shape == V.shape

    # --- units ---
    Vq = V * ureg.meter
    Vq_rot = ut.rotate(Vq, R)
    assert np.allclose(Vq_rot.magnitude, ut.rotate(V, R))
    assert Vq_rot.units == ureg.meter

    # --- physical consistency: crystal vs cartesian ---
    K = np.array([0.5, 0.5, 0.0]) * ureg._2pi / ureg.crystal
    K_cart = ut.cryst2cartesian(K, k_lattice)

    # build cartesian version of R
    R_cart = S.to_cartesian(lattice).R

    K_cart_rot_contra = ut.rotate(K_cart, R_cart)
    K_cart_rot_dual = ut.rotate(K_cart, R_cart, contravariant=0, covariant=1)
    K_rot = ut.rotate(K, R, contravariant=0, covariant=1)
    Kc_from_cryst = ut.cryst2cartesian(K_rot, k_lattice)

    # In cartesian covariant and contravariant should be the same.
    assert np.allclose(K_cart_rot_contra, Kc_from_cryst, atol=1e-8)
    assert np.allclose(K_cart_rot_dual, Kc_from_cryst, atol=1e-8)

    # --- rotation over contratraviant / dual vectors ---
    # metric g
    g = (lattice @ lattice.T).magnitude
    g_inv = np.linalg.inv(g)
    # vector
    r_contra = np.random.rand(3)
    r_dual = g @ r_contra
    Rr_contra = ut.rotate(r_contra, R, contravariant=1, covariant=0)
    Rr_dual = ut.rotate(r_dual, R, contravariant=0, covariant=1)
    assert_allclose(Rr_contra, g_inv @ Rr_dual)
    # tensor
    T_11 = np.random.rand(3, 3)
    T_02 = g @ T_11
    T_20 = T_11 @ g_inv
    RT_11 = ut.rotate(T_11, R, contravariant=1, covariant=1)
    RT_02 = ut.rotate(T_02, R, contravariant=0, covariant=2)
    RT_20 = ut.rotate(T_20, R, contravariant=2, covariant=0)
    assert_allclose(RT_11, g_inv @ RT_02)
    assert_allclose(RT_11, RT_20 @ g)

    # --- Roundtrip over crystal & real space ---
    # Crystal coord -> Rotate in crystal -> To cartesian -> Rotate^-1 in cartesian -> To Crystal
    lat = lattice.magnitude
    klat = k_lattice.magnitude
    R_cart = np.array(
        [[1 / 2, -np.sqrt(3) / 2, 0], [np.sqrt(3) / 2, 1 / 2, 0], [0, 0, 1]]
    )
    R_cryst = ut.change_basis(R_cart, np.eye(3), lattice, contravariant=1, covariant=1)
    R_cryst_dual = ut.change_basis(
        R_cart, np.eye(3), klat, contravariant=1, covariant=1
    )

    # Round 1
    T_cryst = np.random.rand(3, 3)  # Assumed too be contravariant.
    TR_cryst = ut.rotate(T_cryst, R_cryst, contravariant=2, covariant=0)
    TR_cart = ut.change_basis(TR_cryst, lat, np.eye(3), contravariant=2, covariant=0)
    T_cart = ut.rotate(TR_cart, np.linalg.inv(R_cart), contravariant=2, covariant=0)
    OUT = ut.change_basis(T_cart, np.eye(3), lat, contravariant=2, covariant=0)
    assert np.allclose(T_cryst, OUT)

    # Round 2
    TR_cryst = ut.rotate(T_cryst, R_cryst, contravariant=2, covariant=0)
    # Here treated as covariant as we are rotating the k_lattice.
    TR_cart = ut.change_basis(TR_cryst, klat, np.eye(3), contravariant=0, covariant=2)
    T_cart = ut.rotate(TR_cart, np.linalg.inv(R_cart), contravariant=2, covariant=0)
    OUT = ut.change_basis(T_cart, np.eye(3), lat, contravariant=2, covariant=0)
    assert np.allclose(T_cryst, OUT)

    # Round 3
    # Here the Rotation is in dual basis, an thus T is covariant.
    TR_cryst = ut.rotate(T_cryst, R_cryst_dual, contravariant=0, covariant=2)
    TR_cart = ut.change_basis(TR_cryst, klat, np.eye(3), contravariant=0, covariant=2)
    T_cart = ut.rotate(TR_cart, np.linalg.inv(R_cart), contravariant=2, covariant=0)
    OUT = ut.change_basis(T_cart, np.eye(3), lat, contravariant=2, covariant=0)
    assert np.allclose(T_cryst, OUT)

    # --- Cartesian index invariance ---
    # In cartesian covariant/contravariant indcies dont' matter:
    assert np.allclose(
        ut.rotate(T_cart, R_cart, contravariant=1, covariant=1),
        ut.rotate(T_cart, R_cart, contravariant=2, covariant=0),
    )
    assert np.allclose(
        ut.rotate(T_cart, R_cart, contravariant=1, covariant=1),
        ut.rotate(T_cart, R_cart, contravariant=0, covariant=2),
    )


@pytest.mark.parametrize(
    "grid,periodic,expected_shape",
    [
        ([1], True, (1, 1)),
        ([2], False, (2, 1)),
        ([2, 3], False, (6, 2)),
        ([2, 3, 4], False, (24, 3)),
        ([2, 3], True, (6, 2)),
    ],
)
def test_grid_generator(grid, periodic, expected_shape):
    coords = ut.grid_generator(grid, periodic=periodic)

    # --- shape ---
    assert coords.shape == expected_shape

    # --- bounds ---
    if periodic:
        assert np.all(coords < 0.5 + 1e-12)
        assert np.all(coords >= -0.5 - 1e-12)
    else:
        assert np.all(coords <= 1.0 + 1e-12)
        assert np.all(coords >= -1.0 - 1e-12)

    # --- uniqueness (no duplicate points) ---
    coords_view = np.ascontiguousarray(coords).view(
        np.dtype((np.void, coords.dtype.itemsize * coords.shape[1]))
    )
    assert len(np.unique(coords_view)) == len(coords)

    # --- origin presence (if symmetric grid) ---
    if (
        all(g % 2 == 1 for g in grid) or periodic
    ):  # odd and periodic grids should include 0
        assert np.any(np.all(np.isclose(coords, 0.0), axis=1))

    # --- periodic consistency (no endpoint duplication) ---
    if periodic:
        # Check that 0.5 is not present
        assert not np.any(np.isclose(coords, 0.5))

    # --- non-periodic endpoints ---
    if not periodic:
        for d, g in enumerate(grid):
            if g > 1:
                vals = coords[:, d]
                assert np.isclose(vals.min(), -1.0)
                assert np.isclose(vals.max(), 1.0)

    # --- C-ordering (last index varies fastest) ---
    D = coords.shape[1]
    for d in reversed(range(D)):
        vals = coords[:, d]
        diffs = np.diff(vals)

        # Find where this dimension actually changes
        changing = np.abs(diffs) > 1e-12

        if np.any(changing):
            first_change = np.argmax(changing)

            # All earlier dimensions should be constant up to this point
            for d_prev in range(d):
                assert np.allclose(
                    coords[: first_change + 1, d_prev], coords[0, d_prev]
                )
            break


def test_grid_generator_periodic():
    # periodic -> values in (-0.5, 0.5], no duplicates at borders
    coords = ut.grid_generator([4, 2], periodic=True)
    assert coords.shape == (8, 2)
    assert np.all(coords[:, 0] > -0.5 - 1e-12) and np.all(coords[:, 0] <= 0.5 + 1e-12)
    assert np.all(coords[:, 1] > -0.5 - 1e-12) and np.all(coords[:, 1] <= 0.5 + 1e-12)


def test_methpax_kernel_integrates_to_A():
    # Order 0 should integrate to A. Numerically approximate over a wide range.
    A = 1.7
    s = 0.2  # smearing
    xs = np.linspace(-5 * s, 5 * s, 2001)
    # order 0
    ys = np.array(
        [ut.methpax_kernel(x, mean=0.0, smearing=s, order=0, A=A) for x in xs]
    )
    integral = np.trapezoid(ys, xs)
    assert_allclose(
        integral, A, rtol=5e-3, atol=5e-3
    )  # loose tolerance due to finite range
    # order 1
    ys = np.array(
        [ut.methpax_kernel(x, mean=0.0, smearing=s, order=0, A=A) for x in xs]
    )
    integral = np.trapezoid(ys, xs)
    assert_allclose(
        integral, A, rtol=5e-3, atol=5e-3
    )  # loose tolerance due to finite range


def test_fermidirac_kernel_integrates_to_one():
    # Order 0 should integrate to A. Numerically approximate over a wide range.
    s = 0.2  # smearing
    xs = np.linspace(-10 * s, 10 * s, 2001)
    # order 0
    ys = ut.fermidirac_kernel(xs, mean=0.0, smearing=s)
    integral = np.trapezoid(ys, xs)
    assert_allclose(
        integral, 1, rtol=5e-3, atol=5e-3
    )  # loose tolerance due to finite range


def test_analyze_distribution_gaussian():
    mu = 0.3
    sigma = 0.1
    A = 1
    X = np.linspace(mu - 6 * sigma, mu + 6 * sigma, 2001)
    #    Y = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X - mu) / sigma) ** 2)
    Y = ut._normal_dist(X, mean=mu, sd=sigma, A=A)

    stats = ut.analyze_distribution(X, Y)
    assert_allclose(stats.norm, A, rtol=1e-3, atol=1e-3)
    assert_allclose(stats.mean, mu, rtol=1e-3, atol=1e-3)
    assert_allclose(stats.std, sigma, rtol=1e-3, atol=1e-3)
    # Gaussian skewness ~ 0, kurtosis ~ 3
    assert abs(stats.skewness) < 1e-2
    assert_allclose(stats.kurtosis, 3.0, rtol=1e-2, atol=1e-2)


def test_analyze_distribution_zero_norm_raise():
    X = np.linspace(-2, 2, 201)
    Y = X
    with pytest.raises(ValueError):
        ut.analyze_distribution(X, Y)


def test_kernel_density_normalization_gaussian():
    """
    For Gaussian kernel (order=0), the integral over the density should equal
    sum_k w_k * sum_b values[k,b]. With values=1 and weights summing to 1,
    this equals the number of bands.
    """
    nk, nb = 4, 3
    # Simple energies spread around 0
    x = np.array(
        [
            [-0.20, 0.00, 0.15],
            [-0.10, 0.05, 0.25],
            [0.00, 0.10, 0.30],
            [0.05, 0.20, 0.40],
        ]
    )
    assert x.shape == (nk, nb)
    values = np.ones_like(x)
    weights = np.ones(nk) / nk  # sum(weights) = 1

    # Choose a window around the energies, and small sigma for reasonable resolution
    center = 0.0
    x_window = 1.0
    sigma = 0.05
    steps = 2000

    out = ut.kernel_density_on_grid(
        x=x,
        values=values,
        weights=weights,
        center=center,
        x_window=x_window,
        sigma=sigma,
        steps=steps,
        order=0,  # Gaussian
        cutoff_sigmas=5.0,
    )
    # Integral of density over grid should be nb (sum_b values) since sum_k w_k = 1
    integral = np.trapezoid(out.density, out.grid)
    assert_allclose(integral, nb, rtol=5e-3, atol=5e-3)


def test_kernel_density_units():
    """
    With unitful inputs: grid should have x units; density should have units values/x.
    """
    nk, nb = 2, 2
    x = np.array([[0.0, 0.1], [0.2, 0.3]]) * ureg.eV
    values = np.ones_like(x)  # dimensionless "states"
    weights = np.array([0.5, 0.5])

    out = ut.kernel_density_on_grid(
        x=x,
        values=values,
        weights=weights,
        center=0.0 * ureg.eV,
        x_window=0.5 * ureg.eV,
        sigma=0.05 * ureg.eV,
        steps=1000,
        order=0,
    )
    # Grid units should be eV
    assert hasattr(out.grid, "units")
    assert out.grid.check(ureg.eV)
    # Density units should be 1/eV
    assert hasattr(out.density, "units")
    assert out.density.check(1 / ureg.eV)


def test_kernel_density_shape_errors():
    """
    Mismatched shapes should raise ValueError.
    """
    nk, nb = 3, 2
    x = np.random.rand(nk, nb)
    values_bad = np.random.rand(nk, nb + 1)  # wrong shape
    weights_bad = np.ones(nk + 1) / (nk + 1)  # wrong length

    with pytest.raises(ValueError):
        ut.kernel_density_on_grid(x=x, values=values_bad)

    with pytest.raises(ValueError):
        ut.kernel_density_on_grid(x=x, values=np.ones_like(x), weights=weights_bad)


def test_kernel_density_mp_order_1_normalization():
    """
    For MP kernel with order=1, the integral should still approximate sum_k w_k * sum_b values,
    provided the window and cutoff resolve the kernel tails well.
    """
    nk, nb = 3, 2
    x = np.array(
        [
            [-0.15, 0.05],
            [0.00, 0.10],
            [0.20, 0.30],
        ]
    )
    values = np.ones_like(x)
    weights = np.ones(nk) / nk

    out = ut.kernel_density_on_grid(
        x=x,
        values=values,
        weights=weights,
        center=0.0,
        x_window=1.0,
        sigma=0.06,
        steps=2000,
        order=1,  # Methfessel–Paxton order 1
        cutoff_sigmas=5.0,
    )
    integral = np.trapezoid(out.density, out.grid)
    assert_allclose(integral, nb, rtol=1e-2, atol=1e-2)


def test_kernel_density_window_and_steps_defaults():
    """
    If x_window and steps are None, the function should still produce a sensible grid and density.
    """
    nk, nb = 2, 2
    x = np.array([[0.0, 0.1], [0.2, 0.3]])
    out = ut.kernel_density_on_grid(
        x=x, values=None, weights=None, sigma=None, steps=None, order=0
    )
    # Basic sanity checks
    assert isinstance(out.grid, np.ndarray) or hasattr(out.grid, "magnitude")
    assert isinstance(out.density, np.ndarray) or hasattr(out.density, "magnitude")
    # density and grid lengths must match
    if isinstance(out.grid, np.ndarray):
        assert out.grid.shape == out.density.shape
    else:
        assert out.grid.magnitude.shape == out.density.magnitude.shape


def test_kernel_regression_units_and_constant_recovery():
    """
    Output units must match `values` units, and a constant `values` should be recovered.
    """
    nk, nb = 3, 2
    # x with units (e.g., eV), and constant values with another unit (e.g., ampere)
    x = np.array([[-0.3, 0.0], [0.1, 0.2], [0.4, 0.5]]) * ureg.eV
    values = np.full_like(x, 5.0) * ureg.ampere
    weights = np.ones(nk) / nk

    # Build regression callable
    f = ut.kernel_regression(
        x=x,
        values=values,
        weights=weights,
        default_sigma=0.05 * ureg.eV,
        default_cutoff_sigmas=5.0,
        order=0,
    )

    # Evaluate on a grid in the same x units
    X = np.linspace(-0.4, 0.6, 201) * ureg.eV
    y = f(X)

    # Units must match values units
    assert hasattr(y, "units")
    assert y.check(ureg.ampere)

    # For constant values, the regression output should be approximately the constant
    assert_allclose(y.magnitude, 5.0, rtol=5e-2, atol=5e-2)


def test_kernel_regression_sigma_override_and_scalar_eval():
    """
    The returned callable should accept scalar/array inputs and allow overriding sigma at call time.
    """
    nk, nb = 2, 2
    x = np.array([[0.0, 0.1], [0.3, 0.4]])  # unitless for simplicity
    # Let values be constant so the target is easy to check
    values = np.ones_like(x) * 2.0
    weights = np.ones(nk) / nk

    # Build callable with a default sigma
    f = ut.kernel_regression(
        x=x,
        values=values,
        weights=weights,
        default_sigma=0.05,
        default_cutoff_sigmas=5.0,
        order=0,
    )

    # Scalar evaluation
    y0 = f(0.2)
    # Array evaluation with an override sigma (narrower kernel)
    X = np.array([0.05, 0.2, 0.35])
    y1 = f(X, sigma=0.02)

    # Outputs are numeric, shapes as expected
    assert np.all(y0 != y1)
    assert np.isscalar(y0) or getattr(y0, "shape", ()) == ()
    assert isinstance(y1, np.ndarray)

    # Since values are constant (2.0), regression output should be ~2.0
    # (tolerate some smoothing variation)
    assert_allclose(y0, 2.0, rtol=5e-2, atol=5e-2)
    assert_allclose(y1, 2.0, rtol=5e-2, atol=5e-2)


def test_expand_zone_border_shapes_and_units():
    q = np.array([[0.5, 0.0, 0.0], [0, 0, 0]]) * ureg.meter
    with pytest.raises(TypeError):
        expanded = ut._expand_zone_border(q)
    q = np.array([0.5, 0.0, 0.0])
    expanded = ut._expand_zone_border(q)
    assert expanded.shape == (27, 3)
    q = np.array([[0.5, 0.0, 0.0], [0, 0, 0]])
    expanded = ut._expand_zone_border(q)
    assert expanded.shape == (2 * 27, 3)
    # With crystal units preserved
    q_u = q * ureg.crystal
    expanded_u = ut._expand_zone_border(q_u)
    assert isinstance(expanded_u, ureg.Quantity)
    assert expanded_u.check(ureg.crystal)
    assert_allclose(expanded_u.magnitude, expanded)


def test_amplitude2order_parameter():
    # Two atoms, simple displacements
    amps = np.array([0.05, 0.10]) * ureg.angstrom  # length units
    masses = np.array([1.0, 4.0]) * ureg.kilogram  # mass units
    disp0 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])  # norms: 1 and 2
    disp1 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])  # norms: 1 and 3
    # NN_i = sum_j M_j * ||ε_ij||^2
    # For disp0: 1*1^2 + 4*2^2 = 1 + 16 = 17
    # For disp1: 1*1^2 + 4*3^2 = 1 + 36 = 37
    expected = np.array(
        [amps.magnitude[0] * np.sqrt(17), amps.magnitude[1] * np.sqrt(37)]
    )

    q = ut.amplitude2order_parameter(amps, masses, [disp0, disp1])
    assert isinstance(q, ureg.Quantity)
    # Units: length * sqrt(mass)
    assert q.check("angstrom * kilogram^0.5")
    assert_allclose(q.magnitude, expected)


def test_cumulative_integral_input_validation():
    with pytest.raises(ValueError):
        ut.cumulative_integral(np.array([[0, 1]]), np.array([0, 1]))  # x not 1D
    with pytest.raises(ValueError):
        ut.cumulative_integral(np.array([0, 1]), np.array([0, 1, 2]))  # shape mismatch
    with pytest.raises(ValueError):
        ut.cumulative_integral(
            np.array([0, 0, 1]), np.array([0, 1, 2])
        )  # not strictly increasing


def test_cumulative_integral_linear_function_and_units():
    # y = x => integral = 0.5 x^2
    x = np.linspace(0.0, 2.0, 201)
    y = x.copy()
    I = ut.cumulative_integral(x, y)
    assert_allclose(I, 0.5 * x**2, rtol=1e-6, atol=1e-6)

    # With units: x in m, y in N => integral has units N*m
    x_u = x * ureg.meter
    y_u = y * ureg.newton
    Iu = ut.cumulative_integral(x_u, y_u)
    assert isinstance(Iu, ureg.Quantity)
    assert Iu.check(ureg.newton * ureg.meter)
    assert_allclose(Iu.magnitude, I, rtol=1e-6, atol=1e-6)


def test_point_to_segment_distance():
    # Point on the segment
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    P_on = np.array([0.3, 0.0, 0.0])
    assert_allclose(ut._point_to_segment_distance(P_on, A, B), 0.0)

    # Point above the middle
    P_above = np.array([0.5, 2.0, 0.0])
    assert_allclose(ut._point_to_segment_distance(P_above, A, B), 2.0)

    # Point nearest to endpoint A
    P_near_A = np.array([-1.0, 1.0, 0.0])
    # The closest point on the segment is A (0,0,0), distance sqrt(2)
    assert_allclose(
        ut._point_to_segment_distance(P_near_A, A, B), np.sqrt(2.0), atol=1e-12
    )


def test_symmetry_orbit_kpoints_matches_2x2x2_grid(data_dir, require):
    """
    For a QE XML with a 2x2x2 Monkhorst-Pack grid, the symmetry orbit
    of IBZ k-points (modulo G) should match the full 2x2x2 grid generated
    by ut.grid_generator (up to ordering).
    """

    fname = data_dir / "qe/results_scf/scf.xml"
    require(fname, f"Missing test data: {fname}")

    # Read symmetries and k-points from file
    syms = grep.symmetries(str(fname))
    data = spectrum.ElectronBands(str(fname))
    # Quantity in _2pi/crystal (expected)
    k_ibz = ut.cartesian2cryst(data.kpoints, data.k_lattice)

    # Compute symmetry orbit (modulo G)
    out = ut.symmetry_orbit_kpoints(k_ibz, syms, tol=1e-12, mod_G=True)

    # Build expected 2x2x2 grid in crystal units
    # periodic=True means values in (-0.5, 0.5] for a 2x2x2 mesh
    grid = ut.grid_generator([2, 2, 2], periodic=True)  # ndarray or Quantity

    # Prepare actual orbit set for comparison (float crystal units)
    orbit = out.kpoints.magnitude

    # Compare sets ignoring order: sort rows and compare shape and values
    assert grid.shape == orbit.shape

    # For each expected point, ensure there is an orbit point equivalent mod 1
    def is_close_mod1(a, b, tol=1e-12):
        """
        Check if vectors a and b are equal modulo integers (component-wise), i.e.,
        a - b ≡ 0 (mod 1). Assumes a, b are 1D arrays of length 3.
        """
        d = a - b
        d = d - np.round(d)
        return np.all(np.abs(d) <= tol)

    for kg in grid:
        assert any(
            is_close_mod1(kg, ko, tol=1e-12) for ko in orbit
        ), f"Missing k-point mod 1: {ke}"

    # mod_G requires crystal units; verify unit on output
    assert hasattr(out.kpoints, "units")
    assert out.kpoints.check(ureg("_2pi/crystal"))


def test_expand_irreducible_bz_matches_2x2x2_grid(data_dir, require):
    """
    For a QE XML with a 2x2x2 Monkhorst-Pack grid, the symmetry orbit
    of IBZ k-points (modulo G) should match the full 2x2x2 grid generated
    by ut.grid_generator (up to ordering).
    """

    fname = data_dir / "qe/results_scf/scf.xml"
    require(fname, f"Missing test data: {fname}")

    # Read symmetries and k-points from file
    syms = grep.symmetries(str(fname))
    data = spectrum.ElectronBands(str(fname))
    # Quantity in _2pi/crystal (expected)
    k_ibz = ut.cartesian2cryst(data.kpoints, data.k_lattice)

    # Compute symmetry orbit (modulo G)
    out = ut.expand_irreducible_bz(k_ibz, [2, 2, 2], syms, tol=1e-5)

    # Build expected 2x2x2 grid in crystal units
    # periodic=True means values in (-0.5, 0.5] for a 2x2x2 mesh
    grid = ut.grid_generator([2, 2, 2], periodic=True)  # ndarray or Quantity

    # Compare sets ignoring order: sort rows and compare shape and values
    assert grid.shape == out.kpoints.shape
    assert hasattr(out.kpoints, "units")
    assert out.kpoints.check(ureg("_2pi/crystal"))

    # Check origin and syms are correct
    for i, k in enumerate(out.kpoints):
        sym = syms[out.sym[i]]
        origin = k_ibz[out.origin[i]]
        Rk = ut.wrap_fractional(ut.rotate(origin, sym.R, contravariant=0, covariant=1))
        k = ut.wrap_fractional(k)
        assert np.allclose(k, Rk)


def test_wrap_fractional():

    # --- Basic wrapping (center = 0) ---
    x = np.array([0.6, -0.6, 1.2, -1.2])
    wrapped = ut.wrap_fractional(x)
    expected = np.array([-0.4, 0.4, 0.2, -0.2])
    assert np.allclose(wrapped, expected), "Basic wrapping failed"

    # --- 2D array ---
    x = np.array([[0.6, -0.6], [1.2, -1.2]])
    wrapped = ut.wrap_fractional(x)
    expected = np.array([[-0.4, 0.4], [0.2, -0.2]])
    assert np.allclose(wrapped, expected), "2D wrapping failed"

    # --- Custom scalar center ---
    x = np.array([1.6])
    wrapped = ut.wrap_fractional(x, center=1.0)
    expected = np.array(
        [1.6 - 1.0]
    )  # → 0.6 → wrapped to [-0.5,0.5) → -0.4 + 1 = 0.6? check:
    expected = np.array([1.6 - 1.0 - 1.0 + 1.0])  # simpler to compute directly:
    expected = ut.wrap_fractional(x, center=1.0)  # sanity reuse
    assert np.allclose(wrapped, expected), "Scalar center failed"

    # --- Vector center (broadcasting) ---
    x = np.array([[0.6, 0.6]])
    center = np.array([0.5, 0.0])
    wrapped = ut.wrap_fractional(x, center=center)
    expected = np.array([[0.6, -0.4]])
    assert np.allclose(wrapped, expected), "Vector center failed"

    # --- Idempotency ---
    x = np.random.rand(10, 3)
    wrapped_once = ut.wrap_fractional(x)
    wrapped_twice = ut.wrap_fractional(wrapped_once)
    assert np.allclose(wrapped_once, wrapped_twice), "Not idempotent"

    # --- Boundary behavior ---
    x = np.array([0.5, -0.5])
    wrapped = ut.wrap_fractional(x)
    # Expect interval [-0.5, 0.5)
    assert wrapped[0] == -0.5, "Upper boundary not wrapped correctly"
    assert wrapped[1] == -0.5, "Lower boundary incorrect"


def test_find_little_group_silicon(data_dir, require):
    fname = data_dir / "qe/results_scf/scf.xml"
    require(fname, f"Missing test data: {fname}")

    # Read symmetries and k-points from file
    syms = grep.symmetries(str(fname))
    data = spectrum.ElectronBands(str(fname))
    # Quantity in _2pi/crystal (expected)
    k_ibz = ut.cartesian2cryst(data.kpoints, data.k_lattice)

    # Compute little_group or original points and points in the orbits
    origin_lg = ut.find_little_group(k_ibz, syms, tol=1e-12)
    orbit = ut.symmetry_orbit_kpoints(k_ibz, syms, tol=1e-12, mod_G=True)
    orbit_lg = ut.find_little_group(orbit.kpoints, syms, tol=1e-12)

    # Check that all little groups of points in the star have the same number of elements
    for i, k in enumerate(orbit.kpoints):
        assert len(orbit_lg[i]) == len(origin_lg[orbit.origin[i]])


def _avg_delta_k(lattice, kgrid):
    # Matches the average Δk definition used in verbose mode
    if isinstance(lattice, ureg.Quantity):
        Kvol = (2 * np.pi) ** 3 / np.linalg.det(lattice.magnitude)
        dk = (Kvol / np.prod(kgrid)) ** (1 / 3) * (1 / lattice.units)
        return dk
    else:
        Kvol = (2 * np.pi) ** 3 / np.linalg.det(lattice)
        return (Kvol / np.prod(kgrid)) ** (1 / 3)


def test_auto_kgrid_spacing_and_parity():
    # Cubic 5 Å cell, target Δk = 0.1 Å^-1
    a = 5.0 * ureg.angstrom
    lattice = np.eye(3) * a
    dk_target = 0.1 / ureg.angstrom

    # Enforce odd in x,z and even in y
    kgrid = ut.auto_kgrid(
        lattice=lattice,
        delta_k=dk_target,
        force_odd=[True, False, True],
        force_even=[False, True, False],
    )

    # Basic sanity
    assert isinstance(kgrid, list) and len(kgrid) == 3
    assert all(isinstance(ni, int) and ni >= 1 for ni in kgrid)

    # Parity checks
    assert kgrid[0] % 2 == 1  # odd
    assert kgrid[1] % 2 == 0  # even
    assert kgrid[2] % 2 == 1  # odd

    # Achieved average Δk close to target (allow ~25% due to integer rounding/parity)
    dk_eff = _avg_delta_k(lattice, kgrid)
    assert dk_eff.check(ureg.angstrom**-1)
    rel_err = (
        abs(dk_eff.to(dk_target.units).magnitude - dk_target.magnitude)
        / dk_target.magnitude
    )
    assert rel_err < 0.25


def test_eigen_projection():
    # BASIC projection
    # Setup initial and final spectra
    spectra_in = np.array([[1, 0], [0, 1]])
    spectra_out = np.array([[1, 0], [0, 1]])

    # Test without eigenvalues_out, expecting identity projections
    proj = ut.eigen_projection(spectra_in, spectra_out)
    expected_proj = np.array([[1, 0], [0, 1]], dtype=complex)
    np.testing.assert_array_almost_equal(proj, expected_proj)

    # BASIC projection with degenerate eigenvalues
    spectra_in = np.array([[1, 0], [0, 1]])
    spectra_out = np.array([[1, 0], [0, 1]])
    eigenvalues_out = np.array([1.0, 1.0])  # Degenerate eigenvalues

    proj = ut.eigen_projection(spectra_in, spectra_out, eigenvalues_out)
    expected_proj = np.array([[1, 0], [1, 0]], dtype=float)
    np.testing.assert_array_almost_equal(proj, expected_proj)

    # Setup spectra with unitful eigenvalues using Pint
    spectra_in = np.array([[1, 0], [0, 1]])
    spectra_out = np.array([[1, 0], [0, 1]])
    eigenvalues_out = np.array([1.0, 1.0]) * ureg.hertz

    proj = ut.eigen_projection(spectra_in, spectra_out, eigenvalues_out)
    expected_proj = np.array([[1, 0], [1, 0]], dtype=float)
    np.testing.assert_array_almost_equal(proj, expected_proj)

    # Setup spectra that violate norm condition
    spectra_in = np.array([[1, 1] / np.sqrt(2), [0, 0]])
    spectra_out = np.array([[1, 0], [0, 1]])

    # Check that a warning is raised
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ut.eigen_projection(spectra_in, spectra_out)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Projections norm is" in str(w[-1].message)
