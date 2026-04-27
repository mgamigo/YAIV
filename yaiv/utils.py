"""
YAIV | yaiv.utils
=================

This module provides general-purpose utility functions that are used across various classes
and methods in the codebase. They are also intended to be reusable by the user for custom
workflows, especially when combined with the data extraction tools.

Functions
---------
invQ(matrix)
    Returns the inverse of a matrix, preserving units if present.

reciprocal_basis(lattice)
    Computes the reciprocal lattice vectors from a real-space lattice basis.

cartesian2cryst(cartesian_coord, cryst_basis)
    Transforms coordinates from Cartesian to crystal basis, with unit handling.

def rotate(coords, R)
    Rotates vectors using a symmetry rotation matrix (for contravariant and covariant coordintes).

cryst2cartesian(crystal_coord, cryst_basis)
    Transforms coordinates from crystal to Cartesian basis, with unit handling.

cartesian2voigt(xyz)
    Converts a 3×3 symmetric tensor to 6-element Voigt notation.

voigt2cartesian(voigt)
    Converts a 6-element Voigt vector to a 3×3 symmetric tensor.

grid_generator(grid, periodic=False)
    Generates a uniform D-dimensional grid in either periodic or bounded mode.

methpax_kernel(x, mean=0.0, smearing=0.1, order=1, A=1.0)
    Evaluates the Methfessel–Paxton brodening kernel approximation to the delta function up to a given order.

fermidirac_kernel(x, mean=0.0, smearing=0.1)
    Fermi–Dirac (FD) broadening kernel.

analyze_distribution(x, y)
    Computes the mean, std, skewness, kurtosis and normalization of a distribution defined over `X`.

kernel_density(x, values)
    Build a callable density(X) that returns the kernel-broadened density evaluated at arbitrary positions X.

kernel_regression(x, values)
    Build a callable that performs 1D kernel regression (Nadaraya–Watson estimator) from samples.

kernel_density_on_grid(x, values)
    Compute a kernel-broadened density on a grid from samples located at `x`.

amplitude2order_parameter(amplitudes, masses, displacements)
    Convert displacement amplitudes into proper order parameters with [length × sqrt(mass)] units.

cumulative_integral(X, Y)
    Compute the cumulative integral of a function defined by discrete x and y values.

wrap_fractional(coords)
    Wrap fractional coordinates into a periodic unit cell.

find_little_group(kpoints,symmetries)
    Compute the little group for each input k-point.

symmetry_orbit_kpoints(kpoints,symmetries)
    Apply all symmetry rotations to a set of k-points (row vectors) and return the unique full set.

expand_irreducible_bz(kpoints, grid, symmetries)
    Expand a set of irreducible k-points to the full Brillouin zone using symmetry operations.

auto_kgrid(lattice)
    Compute a k-grid from a target k-point spacing or target kpoints-per-reciprocal-atom (KPPRA).

eigen_projection(spectra_in, spectra_out)
    Projects initial spectra onto final spectra.

Private Utilities
-----------------
_check_unit_consistency(quantities, names=None)
    Verifies that a list of variables are either all unitful or all unitless. Raises `TypeError` if mixed.

_normal_dist(x, mean=0, sd=0.1, A=1)
    Computes the value of a normalized Gaussian distribution.

_expand_zone_border(q_point)
    Returns a q-point and its ±1-shifted equivalents along each reciprocal direction.

_point_to_segment_distance(point, endpoint_a, endpoint_b)
    Compute the Euclidean distance from a point to a line segment in 3D.

See Also
--------
yaiv.grep             : File parsing functions that uses these utilities.
yaiv.spectrum         : Core spectral class storing eigenvalues and k-points.
"""

import warnings
from types import SimpleNamespace
from typing import Sequence, Any

import numpy as np

from yaiv.defaults.config import ureg, defaults

__all__ = [
    "invQ",
    "reciprocal_basis",
    "rotate" "cartesian2cryst",
    "cryst2cartesian",
    "cartesian2voigt",
    "voigt2cartesian",
    "grid_generator",
    "methpax_kernel",
    "fermidirac_kernel",
    "analyze_distribution",
    "kernel_density",
    "kernel_regression",
    "kernel_density_on_grid",
    "amplitude2order_parameter",
    "cumulative_integral",
    "wrap_fractional",
    "find_little_group",
    "symmetry_orbit_kpoints",
    "expand_irreducible_bz",
    "auto_kgrid",
    "eigen_projection",
]


def _check_unit_consistency(quantities: Sequence[Any], names: Sequence[str] = None):
    """
    Ensure that all (non-None) inputs are either unitful (pint.Quantity) or all unitless.

    Parameters
    ----------
    quantities : list | tuple
        List of values to check (e.g., eigenvalues, shift, etc.).
    names : list[str], optional
        Names of variables (for debugging messages).

    Raises
    ------
    TypeError
        If the list contains a mix of unitful and unitless variables.
    """
    has_units = [
        isinstance(x, ureg.Quantity) if x is not None else x for x in quantities
    ]
    S = set(has_units)
    S.discard(None)
    if len(S) != 1:
        if names is not None:
            print("Units check failed for:", names)
        print("Units status:", has_units)
        raise TypeError("Either all or none of the variables must have units.")


def invQ(matrix: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
    """
    Inverts a matrix with (or without) units of 1/[input_units].

    Parameters
    ----------
    matrix : np.ndarray | ureg.Quantity
        Square matrix, with or without units.

    Returns
    -------
    inverse : np.ndarray | ureg.Quantity
        Square matrix, with (1/[input]) or without units (depending on the input).
    """
    if isinstance(matrix, ureg.Quantity):
        return np.linalg.inv(matrix.magnitude) * (1 / matrix.units)
    else:
        return np.linalg.inv(matrix)


def reciprocal_basis(lattice: np.ndarray | ureg.Quantity) -> ureg.Quantity:
    """
    Compute reciprocal lattice vectors (rows) from a direct lattice basis.

    Parameters
    ----------
    lattice : np.ndarray
        Direct lattice vectors in rows, optionally with units as ureg.Quantity.

    Returns
    -------
    K_vec : ureg.Quantity
        Reciprocal lattice vectors in rows, with units of 2π / [input_units].
    """
    K_vec = (invQ(lattice) * ureg._2pi).transpose()  # reciprocal vectors in rows
    return K_vec


def rotate(
    coords: np.ndarray | ureg.Quantity,
    R: np.ndarray,
    *,
    covariant: bool = False,
) -> np.ndarray | ureg.Quantity:
    """
    Rotate vectors using a symmetry rotation matrix.

    The function differs between contravariant and covariant/dual vectors:
    - contravariant → R @ x
    - covariant → R^{-T} @ x

    Parameters
    ----------
    coords : np.ndarray | ureg.Quantity
        Array of vectors (..., N), in row-vector convention.
    R : np.ndarray, shape (N, N)
        Rotation matrix (in the same basis as coords).
    covariant : bool, optional
        If True, treat vectors as covariant (dual vectors).
        Otherwise contravariant (default).

    Returns
    -------
    np.ndarray | ureg.Quantity
        Rotated vectors with same shape and units.
    """

    R = np.asarray(R, dtype=float)

    # --- apply transformation ---
    # row convention for coords
    if covariant:
        # covariant → R^{-T} @ x
        coords_rot = coords @ np.linalg.inv(R)
    else:
        # contravariant → R @ x
        coords_rot = coords @ R.T

    return coords_rot


def cartesian2cryst(
    cartesian_coord: np.ndarray | ureg.Quantity, cryst_basis: np.ndarray | ureg.Quantity
) -> np.ndarray | ureg.Quantity:
    """
    Convert from Cartesian to crystal coordinates.

    Parameters
    ----------
    cartesian_coord : np.ndarray | ureg.Quantity
        Vector or matrix in Cartesian coordinates. May include units.
    cryst_basis : np.ndarray | ureg.Quantity
        Basis vectors written as rows. May include units.

    Returns
    -------
    crystal_coord : np.ndarray | ureg.Quantity
        Result in crystal coordinates, with modified units if possible.

    Raises
    ------
    TypeError
        If the input units are not compatible with the basis units (i.e., their ratio is not dimensionless).
    """
    _check_unit_consistency([cartesian_coord, cryst_basis], ["coordinates", "basis"])

    inv = invQ(cryst_basis)
    crystal_coord = cartesian_coord @ inv
    if isinstance(cartesian_coord, ureg.Quantity) and isinstance(
        cryst_basis, ureg.Quantity
    ):
        if not crystal_coord.dimensionless:
            raise TypeError(
                "Input and basis units are not compatible for coordinate transformation"
            )
        in_units = cartesian_coord.units
        if in_units.dimensionality in [
            ureg.meter.dimensionality,
            ureg.alat.dimensionality,
        ]:
            crystal_coord = crystal_coord * (ureg.crystal)

        elif in_units.dimensionality in [
            1 / ureg.meter.dimensionality,
            1 / ureg.alat.dimensionality,
        ]:
            crystal_coord = crystal_coord * (ureg._2pi / ureg.crystal)
        else:
            raise TypeError(
                "Input units must have dimensionality of [length] or [1/length]"
            )

    return crystal_coord


def cryst2cartesian(
    crystal_coord: np.ndarray | ureg.Quantity, cryst_basis: np.ndarray | ureg.Quantity
) -> np.ndarray | ureg.Quantity:
    """
    Convert from crystal to Cartesian coordinates.

    Parameters
    ----------
        crystal_coord : np.ndarray | ureg.Quantity
            Coordinates or matrix in crystal units.
        cryst_basis : np.ndarray | ureg.Quantity
            Basis vectors written as rows.

    Returns
    -------
        cartesian_coord : np.ndarray | ureg.Quantity
            Result in cartesian coordinates, with modified units if possible.

    Raises
    ------
    TypeError
        If the input units are not correct (i.e., not providing crystal units).
    """
    _check_unit_consistency([crystal_coord, cryst_basis], ["coordinates", "basis"])

    if isinstance(crystal_coord, ureg.Quantity) and isinstance(
        cryst_basis, ureg.Quantity
    ):
        if crystal_coord.dimensionality == ureg.crystal.dimensionality:
            crystal_coord = crystal_coord * (1 / ureg.crystal)
        elif crystal_coord.dimensionality == 1 / ureg.crystal.dimensionality:
            crystal_coord = crystal_coord * (ureg.crystal / ureg._2pi)
        else:
            raise TypeError("Input units are not crystal units.")
    cartesian_coord = crystal_coord @ cryst_basis

    return cartesian_coord


def cartesian2voigt(xyz: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
    """
    Convert a symmetric 3x3 tensor from Cartesian (matrix) to Voigt notation.

    This is commonly used for stress and strain tensors, where the 3x3 symmetric
    tensor is flattened into a 6-element vector:
        [xx, yy, zz, yz, xz, xy]

    Parameters
    ----------
    xyz : np.ndarray | ureg.Quantity
        A numpy array with last two indices being a 3x3 symmetric tensor
        in Cartesian notation (a,b,...,3,3). Can optionally carry physical units.

    Returns
    -------
    np.ndarray | ureg.Quantity
        A (a,b,...,6) array in Voigt notation. If the input had units, they are preserved.
    """
    if isinstance(xyz, ureg.Quantity):
        units = xyz.units
        xyz = xyz.magnitude
    else:
        units = 1
    voigt = np.array(
        [
            xyz[..., 0, 0],
            xyz[..., 1, 1],
            xyz[..., 2, 2],
            xyz[..., 1, 2],
            xyz[..., 0, 2],
            xyz[..., 0, 1],
        ]
    )
    # reshape: (a, b, c, d, e) -> result shape: (b, c, d, e, a)
    voigt = np.moveaxis(voigt, (0), (-1))
    return voigt * units


def voigt2cartesian(voigt: np.ndarray | ureg.Quantity) -> np.ndarray | ureg.Quantity:
    """
    Convert a symmetric tensor from Voigt to Cartesian (3x3 matrix) notation.

    This reverses the `cartesian2voigt` operation, converting a 6-element vector into
    a symmetric 3x3 matrix:
        [xx, yy, zz, yz, xz, xy] → [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]

    Parameters
    ----------
    voigt : np.ndarray | ureg.Quantity
        An array with last index being length 6 in Voigt notation (a,b,...,6).
        If the input had units, they are preserved.

    Returns
    -------
    np.ndarray | ureg.Quantity
        A (a,b,...,3,3) symmetric tensor in Cartesian matrix notation.
        If the input had units, they are preserved.
    """
    units = 1
    if isinstance(voigt, ureg.Quantity):
        units = voigt.units
        voigt = voigt.magnitude
    else:
        units = 1
    xyz = np.array(
        [
            [voigt[..., 0], voigt[..., 5], voigt[..., 4]],
            [voigt[..., 5], voigt[..., 1], voigt[..., 3]],
            [voigt[..., 4], voigt[..., 3], voigt[..., 2]],
        ]
    )
    # reshape: (a, b, c, d, e) -> result shape: (c, d, e, a, b)
    xyz = np.moveaxis(xyz, (0, 1), (-2, -1))
    return xyz * units


def grid_generator(grid: list[int], periodic: bool = False) -> np.ndarray:
    """
    Generate a uniform real-space grid of points within [-1, 1]^D or [-0.5, 0.5)^D,
    where D is the grid dimensionality.

    This function constructs a D-dimensional mesh by specifying the number of
    points along each axis. The resulting points are returned as a (N, D) array,
    where N is the total number of grid points.

    Parameters
    ----------
    grid : list[int]
        List of integers specifying the number of points along each dimension.
        For example, [10, 10, 10] creates a 10×10×10 grid.
    periodic : bool, optional
        If True, the grid will in periodic boundary style. Centered at 0 (Γ) with
        values [-0.5,0.5) avoiding duplicate zone borders.
        If False (default), the grid spans from -1 to 1 (inclusive).

    Returns
    -------
    np.ndarray
        Array of shape (N, D), where each row is a point in the D-dimensional grid.
    """
    # Generate the GRID
    DIM = len(grid)
    temp = []
    for g in grid:
        if periodic:
            s = 0
            temp = temp + [np.linspace(s, 1, g, endpoint=False)]
        elif g == 1:
            s = 1
            temp = temp + [np.linspace(s, 1, g)]
        else:
            s = -1
            temp = temp + [np.linspace(s, 1, g)]
    res_to_unpack = np.meshgrid(*temp)
    assert len(res_to_unpack) == DIM

    # Unpack the grid as points
    for x in res_to_unpack:
        c = x.reshape(np.prod(np.shape(x)), 1)
        try:
            coords = np.hstack((coords, c))
        except NameError:
            coords = c
    if periodic == True:
        for c in coords:
            c[c >= 0.5] -= 1  # remove 1 to all values above or equal 0.5
    return coords


def _normal_dist(x, mean=0, sd=0.1, A=1):
    """
    Evaluate a normalized Gaussian (normal) distribution.

    Parameters
    ----------
    x : float or np.ndarray
        Point(s) at which to evaluate the distribution.
    mean : float
        Center (mean) of the Gaussian.
    sd : float
        Standard deviation (width) of the Gaussian.
    A : float, optional
        Amplitude factor. If A=1, the distribution integrates to unity. Default is 1.

    Returns
    -------
    y : float or np.ndarray
        Value(s) of the normalized Gaussian distribution at `x`.

    Notes
    -----
    The Gaussian is defined as:
        A / (σ√(2π)) * exp(-0.5 * ((x - μ) / σ)^2)
    where μ is the mean and σ is the standard deviation.
    """
    y = A / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return y


def methpax_kernel(
    x: float, mean: float = 0.0, smearing: float = 0.1, order: int = 1, A: float = 1.0
) -> float:
    """
    Methfessel-Paxton (MP) broadening kernel approximation to the delta fuction δ(x).

    The MP impulse function generalizes a Gaussian by adding Hermite polynomial
    corrections to improve convergence in electronic structure integrations.

    For order = 0, the function reduces to a normalized Gaussian:
        A / (σ√(2π)) * exp(-0.5 * ((x - μ) / σ)^2)

    Parameters
    ----------
    x : float
        Point(s) at which to evaluate the smearing function.
    mean : float, optional
        Center (mean) of the distribution. Default is 0.
    smearing : float, optional
        Smearing width, directly related to the standard deviation (σ) in the Gaussian case,
        being [smearing = (σ√2)]. Default is 0.1.
    order : int, optional
        Order of the Methfessel-Paxton expansion. Order 0 recovers a Gaussian.
    A : float, optional
        Amplitude factor. If A=1, the result integrates to unity. Default is 1.

    Returns
    -------
    y : float
        Value(s) of the MP-smearing function evaluated at x.
    """
    from scipy.special import factorial, hermite

    def A_n(n):
        return (-1) ** n / (factorial(n) * (4**n) * np.sqrt(np.pi))

    x_scaled = (x - mean) / (smearing)

    y = 0
    for n in range(order + 1):
        coeff = A_n(n)
        H = hermite(2 * n)(x_scaled)  # Evaluate the Hermite polynomial
        y += coeff * H * np.exp(-(x_scaled**2))

    normalization = A / smearing  # Restores scale in physical units
    return normalization * y


def fermidirac_kernel(x: float, mean: float = 0.0, smearing: float = 0.1) -> float:
    """
    Fermi–Dirac (FD) broadening kernel.

    This function is the (positive) derivative of the Fermi–Dirac occupation
    with respect to energy and acts as a normalized impulse:
        g(x; μ, s) = exp((x - μ)/s) / [ s · (1 + exp((x - μ)/s))^2 ]
    where μ = mean and s = smearing (often k_B T in energy units).
    It satisfies ∫ g(x; μ, s) dx = 1, and as s → 0, g → δ(x − μ).

    Parameters
    ----------
    x : float or array-like
        Point(s) at which to evaluate the broadening kernel.
    mean : float, optional
        Center μ of the kernel. Default is 0.
    smearing : float, optional
        Width parameter s (> 0), typically corresponding to thermal smearing
        (e.g., k_B·T if x is in energy units). Default is 0.1.

    Returns
    -------
    y : float
        Fermi-Dirac kernel evaluated at x. Returns an ndarray if x is array-like.
    """
    exp = np.exp((x - mean) / smearing)
    return (exp / smearing) / (exp + 1) ** 2


def analyze_distribution(X, Y):
    """
    Analyze the statistical properties of a function defined over a domain.

    Parameters
    ----------
    X : array_like
        1D array representing the domain (e.g., energy, position).
    Y : array_like
        1D array representing the function values over X (e.g., DOS, intensity).
        Does not need to be normalized; normalization is handled internally.

    Returns
    -------
    stats : SimpleNamespace
        Object containing:
            - mean : float
                First moment (average) of the distribution.
            - variance : float
                Second central moment (spread squared).
            - std : float
                Standard deviation (spread of the function).
            - skewness : float
                Third standardized moment (asymmetry).
                Skewness > 0: tail on the right; < 0: tail on the left.
            - kurtosis : float
                Fourth standardized moment (peakedness, excess).
                Kurtosis > 3: sharper than Gaussian; < 3: flatter.
            - norm : float
                Area under the curve (integral of Y over X).

    Raises
    ------
    ValueError
        If the integral (normalization) of Y is zero. This means the function has
        no area under the curve, and statistical quantities become ill-defined.
    """
    # Ensure arrays
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Compute differential element
    dx = np.gradient(X)

    # Normalization factor
    norm = np.sum(Y * dx)

    # Avoid division by zero
    if norm == 0:
        raise ValueError("Integral (normalization) of Y is zero.")

    # Mean
    mean = np.sum(X * Y * dx) / norm

    # Central moments
    x_shifted = X - mean
    variance = np.sum((x_shifted) ** 2 * Y * dx) / norm
    std = np.sqrt(variance)

    # Higher moments
    skewness = np.sum((x_shifted) ** 3 * Y * dx) / (norm * std**3)
    kurtosis = np.sum((x_shifted) ** 4 * Y * dx) / (norm * std**4)

    return SimpleNamespace(
        mean=mean,
        variance=variance,
        std=std,
        skewness=skewness,
        kurtosis=kurtosis,
        norm=norm,
    )


def kernel_density(
    x: np.ndarray | ureg.Quantity,
    values: np.ndarray | ureg.Quantity = None,
    weights: np.ndarray | None = None,
    default_sigma: float | ureg.Quantity | None = None,
    default_cutoff_sigmas: float = defaults.cutoff_sigmas,
    order: int = 0,
) -> callable:
    """
    Build a callable density(X, cutoff_sigmas=None) that returns the kernel-broadened
    density evaluated at arbitrary positions X.

    This implements a DOS-like convolution:
        density(X) = sum_i values_i * K_sigma(X - x_i) * w_k(i)
    where K is either a Gaussian (order=0) or a Methfessel–Paxton kernel (order>=0).

    Parameters
    ----------
    x : np.ndarray | pint.Quantity, shape (nk, nb)
        Sample locations (e.g., energies). If unitful, `default_sigma` must be compatible.
    values : np.ndarray | pint.Quantity, optional, shape (nk, nb)
        Amplitudes per sample. Defaults to ones giving a DOS like quantity.
    weights : np.ndarray, optional, shape (nk,)
        k-point weights (sum to 1). Defaults to uniform weights 1/nk.
    default_sigma : float | pint.Quantity, optional
        Default kernel width of the returned function. Defaults to ((x.max-x.min)/ 200). (smearing)
    default_cutoff_sigmas : float, optional
        Default cutoff in units of sigma used when density(X) is called without an explicit cutoff.
        Defaults to yaiv.defaults.config.defaults.cutoff_sigmas.
    order : int, optional
        Kernel type: 0=Gaussian; >0=Methfessel–Paxton order; -1=Fermi-Dirac.

    Returns
    -------
    func : callable
        density(X, cutoff_sigmas=None, sigma=None) → array/Quantity. If cutoff_sigmas is None,
        uses `default_cutoff_sigmas`.

    Raises
    ------
    ValueError
        If shapes are inconsistent (e.g., weights length not equal to nk, or values shape
        does not match `x`).

    Notes
    -----
    - Units: output has units (values units)/(x units).
    - The callable supports arrays for X and uses the preprocessed (sorted) samples.
    - To change sigma at call time, pass it as a Quantity or float compatible with x units.
    """
    # Default DOS
    if values is None:
        values = np.ones_like(x)

    # Units consistency
    quantities = [x, default_sigma]
    names = ["x", "default_sigma"]
    _check_unit_consistency(quantities, names)

    # Units normalization
    if isinstance(x, ureg.Quantity):
        x_units = x.units
        x = np.asarray(x.magnitude)
        default_sigma = (
            default_sigma.to(x_units).magnitude if default_sigma is not None else None
        )
    else:
        x_units = 1
        x = np.asarray(x)
        default_sigma = float(default_sigma) if default_sigma is not None else None

    if isinstance(values, ureg.Quantity):
        val_units = values.units
        val = np.asarray(values.magnitude)
    else:
        val_units = 1
        val = np.asarray(values)

    # Shapes/checks
    nk = x.shape[0]
    if values.shape != x.shape:
        raise ValueError("`values` and `x` must have the same shape (nk, nb).")
    if weights is None:
        w = np.ones(nk, dtype=float) / nk
    else:
        w = np.asarray(weights, dtype=float)
    if w.shape[0] != x.shape[0]:
        raise ValueError(
            "`weights` must have shape (nk,) matching the number of k-points."
        )

    # Flatten, align, sort by x for efficient search
    x_flat = x.flatten()
    v_flat = val.flatten()
    w_flat = np.repeat(w, int(len(x_flat) / nk))

    # Sort by x for efficient windowing with searchsorted
    sort_idx = np.argsort(x_flat)
    x_flat = x_flat[sort_idx]
    v_flat = v_flat[sort_idx]
    w_flat = w_flat[sort_idx]

    # Default sigma
    if default_sigma is None:
        default_sigma = (x_flat[-1] - x_flat[0]) / 200

    # Kernel chooser
    def kernel(xloc, X, sig):
        if order == 0:
            return _normal_dist(xloc, mean=X, sd=sig)
        elif order == -1:
            return fermidirac_kernel(xloc, mean=X, smearing=sig)
        return methpax_kernel(xloc, mean=X, smearing=sig, order=order)

    # Build callable
    def density_func(
        X: float | np.ndarray | ureg.Quantity,
        cutoff_sigmas: float = None,
        sigma: float | ureg.Quantity = None,
    ) -> float | np.ndarray | ureg.Quantity:
        """
        Evaluate density at points X.

        Parameters
        ----------
        X : float | np.ndarray | pint.Quantity
            Evaluation points (same units as x).
        cutoff_sigmas : float, optional
            Cutoff in multiples of sigma; defaults to `default_cutoff_sigmas`
            used in the definition.
        sigma : float | pint.Quantity, optional
            Override kernel width at call time.

        Returns
        -------
        density : float | np.ndarray | pint.Quantity
            Computed density at X. Units are (values units) / (x units).

        Raises
        ------
        TypeError
            If unit consistency is violated among unitful inputs.
        """
        # Units for X and sigma at call
        if isinstance(X, ureg.Quantity):
            if X.units != x_units:
                X = np.asarray(X.to(x_units).magnitude)
            else:
                X = np.asarray(X.magnitude)
        elif x_units != 1:
            raise ValueError(f"X should be in dimensionality compatible with {x_units}")
        else:
            X = np.asarray(X)

        # Resolve sigma and cutoff_sigmas
        if sigma is None:
            sigma = default_sigma
        elif x_units != 1:
            sigma = sigma.to(x_units).magnitude
        cutoff = default_cutoff_sigmas if cutoff_sigmas is None else cutoff_sigmas

        # Accumulate contributions
        if len(X.shape) == 0:
            return_float = True
            X = np.asarray([X])
        else:
            return_float = False
            X = np.asarray(X)
        out = np.zeros(X.shape, dtype=float)
        lefts = X - cutoff * sigma
        rights = X + cutoff * sigma

        # vectorized window search
        starts = np.searchsorted(x_flat, lefts, side="left")
        stops = np.searchsorted(x_flat, rights, side="right")

        for i, (start, stop, X0) in enumerate(zip(starts, stops, X)):
            x_loc = x_flat[start:stop]
            v_loc = v_flat[start:stop]
            w_loc = w_flat[start:stop]
            if x_loc.size == 0:
                continue
            K = kernel(x_loc, X0, sigma)
            out[i] = np.sum(K * v_loc * w_loc)

        # Units: (values units)/(x units)
        OUT = out * (val_units / x_units)
        if return_float:
            return OUT[0]
        else:
            return OUT

    return density_func


def kernel_regression(
    x: np.ndarray | ureg.Quantity,
    values: np.ndarray | ureg.Quantity,
    weights: np.ndarray | None = None,
    default_sigma: float | ureg.Quantity | None = None,
    default_cutoff_sigmas: float = defaults.cutoff_sigmas,
    order: int = 0,
    default_reg: float | ureg.Quantity = 1e-10,
) -> callable:
    """
    Build a callable that performs 1D kernel regression (Nadaraya–Watson estimator)
    from samples (x, values), with optional k-point weights.

    The returned function evaluates
        f(X) = density(X) / (DOS(X) + reg)
    where:
      - density(X) = sum_i values_i * K_sigma(X - x_i) * w_i
      - DOS(X)     = sum_i 1         * K_sigma(X - x_i) * w_i
      - K is Gaussian (order=0), Methfessel–Paxton (order>0) or Fermi-Dirac (order=-1).
      - reg is a small regularization to avoid division by ~0.

    Parameters
    ----------
    x : np.ndarray | pint.Quantity, shape (nk, nb)
        Sample locations (e.g., energies). If unitful, `default_sigma` must be compatible.
    values : np.ndarray | pint.Quantity, shape (nk, nb)
        Amplitudes per sample. Defaults to ones giving a DOS like quantity.
    weights : np.ndarray, optional, shape (nk,)
        k-point weights (sum to 1). Defaults to uniform weights 1/nk.
    default_sigma : float | pint.Quantity, optional
        Default kernel width ("smearing") used by returned function.
        If not provided, it defaults to: 10 × (average spacing in x),
        check `Notes` to see how average spacing is defined.
    default_cutoff_sigmas : float, optional
        Default cutoff in units of sigma used when f(X) is called without an explicit cutoff.
        Defaults to yaiv.defaults.config.defaults.cutoff_sigmas.
    order : int, optional
        Kernel type: 0=Gaussian; >0=Methfessel–Paxton order; -1=Fermi-Dirac.
        Default being Gaussian (0).
    default_reg : float | pint.Quantity, optional
        Default regularization added to the denominator DOS to prevent division
        by zero. If x is unitful, `default_reg` should have units 1/x. Defaults
        to 1e-10 (interpreted as 1/x if x has units).

    Returns
    -------
    func : callable
        A function f(X, cutoff_sigmas=None, sigma=None, reg=None) that evaluates
        the kernel regression at points X. If `sigma`, `cutoff_sigmas` or `reg` are not
        provided at call time, the defaults passed here are used.

    Notes
    -----
    - Output has the same units as `values`.
    - The density and DOS funtions are built with utils.kernel_density()
    - The average distance in x is obtained after trimming band gaps: spacings are
    computed from sorted x, outliers larger than 200 × the median spacing are discarded,
    and the mean of the remaining spacings is used.
    """
    # Default sigma
    if default_sigma is None:
        flat = x.flatten()
        flat = flat[np.argsort(flat)]
        dist = np.diff(flat)
        thrushold = np.median(dist) * 200
        no_gaps = dist[np.where(dist < thrushold)[0]]
        default_sigma = 10 * (np.sum(no_gaps) / len(no_gaps))

    if isinstance(x, ureg.Quantity):
        x_units = x.units
    else:
        x_units = 1

    density = kernel_density(
        x=x,
        values=values,
        weights=weights,
        default_sigma=default_sigma,
        default_cutoff_sigmas=default_cutoff_sigmas,
        order=order,
    )

    DOS = kernel_density(
        x=x,
        values=np.ones_like(x),
        weights=weights,
        default_sigma=default_sigma,
        default_cutoff_sigmas=default_cutoff_sigmas,
        order=order,
    )

    # Build callable
    def kernel_regression_func(
        X: float | np.ndarray | ureg.Quantity,
        cutoff_sigmas: float = None,
        sigma: float | ureg.Quantity = None,
        reg: float = None,
    ) -> float | np.ndarray | ureg.Quantity:
        """
        Evaluate the kernel regression f(X) = density(X)/(DOS(X) + reg).

        Parameters
        ----------
        X : float | np.ndarray | pint.Quantity
            Evaluation points (same units as x).
        cutoff_sigmas : float, optional
            Cutoff in multiples of sigma; defaults to `default_cutoff_sigmas`
            used in the definition.
        sigma : float | pint.Quantity, optional
            Override kernel width at call time.
        reg : float | pint.Quantity, optional
            Override the regularization factor at call (units of 1/x).

        Returns
        -------
        estimate : float | np.ndarray | pint.Quantity
            Regression estimate f(X), with the same units as `values`.
        """
        # Resolve reg
        if reg is None:
            reg = default_reg * (1 / x_units)
        if isinstance(reg, ureg.Quantity):
            reg = reg.to(1 / x_units)
        else:
            reg *= 1 / x_units

        A = density(X=X, cutoff_sigmas=cutoff_sigmas, sigma=sigma)
        B = DOS(X=X, cutoff_sigmas=cutoff_sigmas, sigma=sigma)
        OUT = A / (B + reg)
        return OUT

    return kernel_regression_func


def kernel_density_on_grid(
    x: np.ndarray | ureg.Quantity,
    values: np.ndarray | ureg.Quantity = None,
    weights: np.ndarray | None = None,
    center: float | ureg.Quantity | None = None,
    x_window: float | list[float] | ureg.Quantity | None = None,
    sigma: float | ureg.Quantity | None = None,
    steps: int | None = None,
    order: int = 0,
    cutoff_sigmas: float = defaults.cutoff_sigmas,
) -> SimpleNamespace:
    """
    Compute a kernel-broadened density on a grid from samples located at `x`.

    This implements a DOS-like convolution:
        density(X) = sum_i values_i * K_sigma(X - x_i) * w_k(i)
    where K is either a Gaussian (order=0), Methfessel–Paxton (order>0) or Fermi-Dirac (order=-1).

    Parameters
    ----------
    x : np.ndarray | ureg.Quantity
        Sample locations (e.g., energies). If unitful, all other related inputs
        (center, x_window, sigma) must be compatible with `x` units. Shape (nkpts, nbnds)
    values : np.ndarray | ureg.Quantity, optional
        Amplitudes per sample (e.g., projections). Defaults to ones, producing a DOS.
        Shape (nkpts, nbnds)
    weights : np.ndarray, optional
        k-point weights that sum to 1. If None, uniform weights are used: 1/nkpts.
    center : float | ureg.Quantity, optional
        Center of the window (e.g., Fermi level). Defaults to 0.
    x_window : float | list[float] | ureg.Quantity, optional
        Window for the output grid. If a float, interpreted as symmetric [center - w, center + w].
        If a list/array, interpreted as [xmin, xmax] around `center`.
        If None, inferred from min/max(x) expanded by `sigma` and `cutoff_sigmas`.
    sigma : float | ureg.Quantity, optional
        Kernel width. Defaults to (window_size / 200). (smearing)
    steps : int, optional
        Number of grid points. Defaults to int(4 * (window_size / sigma)), with a minimum of 128.
    order : int, optional
        Kernel type: 0=Gaussian; >0=Methfessel–Paxton order; -1=Fermi-Dirac.
    cutoff_sigmas : float, optional
        Truncate kernel support to [-cutoff_sigmas * sigma, +cutoff_sigmas * sigma]
        when summing contributions. Default yaiv.defaults.config.defaults.cutoff_sigmas.

    Returns
    -------
    SimpleNamespace
        grid : np.ndarray | ureg.Quantity
            The output grid in the units of `x`.
        density : np.ndarray | ureg.Quantity
            The computed density at each grid point. Units are (values units) / (x units).
    """
    # Units consistency for x, center, x_window, sigma
    quantities = [x, center, x_window, sigma]
    names = ["x", "center", "x_window", "sigma"]
    _check_unit_consistency(quantities, names)

    # If unitful, convert all to common unit
    if isinstance(x, ureg.Quantity):
        x_units = x.units
        x, center, x_window, sigma = [
            x.to(x_units).magnitude if isinstance(x, ureg.Quantity) else x
            for x in quantities
        ]
    else:
        x_units = 1
    if isinstance(values, ureg.Quantity):
        val_units = values.units
        values = np.asarray(values.magnitude)
    else:
        val_units = 1

    # Determine computing center, window, sigma and steps
    center = 0 if center is None else center
    if x_window is None:
        x_min = x.min()
        x_max = x.max()
    elif isinstance(x_window, (float, int)):
        x_min, x_max = np.array([-x_window, x_window]) + center
    else:
        x_min, x_max = np.asarray(x_window) + center
    window_size = x_max - x_min
    if sigma is None:
        sigma = window_size / 200
    if steps is None:
        # Ensure at least some reasonable resolution
        steps = max(int(4.0 * (window_size / sigma)), 128)
    # If window was None, expand by cutoff beyond data range
    if x_window is None:
        x_min = x_min - sigma * (cutoff_sigmas + 1)
        x_max = x_max + sigma * (cutoff_sigmas + 1)

    grid = np.linspace(x_min, x_max, steps)

    # Build callable and evaluate on grid
    density = kernel_density(
        x,
        values=values,
        weights=weights,
        order=order,
        default_sigma=sigma,
        default_cutoff_sigmas=cutoff_sigmas,
    )
    density_on_grid = (
        density(grid) * val_units / x_units
    )  # returns ndarray or Quantity (values/x units)

    # Attach units to grid if needed
    grid_out = grid * x_units

    return SimpleNamespace(grid=grid_out, density=density_on_grid)


def _expand_zone_border(
    q_point: ureg.Quantity | np.ndarray,
) -> ureg.Quantity | np.ndarray:
    """
    Expand a q-point by adding periodic equivalents related by reciprocal lattice translations.

    This function generates a set of symmetry-related points lying at the borders of the Brillouin zone,
    by adding and subtracting ±1 in each reciprocal direction. This is useful in phonon or electron band
    structure calculations when points like (0.5, 0, 0) and (-0.5, 0, 0) are physically equivalent
    but not explicitly included in the star of q-points.

    Parameters
    ----------
    q_point : pint.Quantity | np.ndarray
        A 3-element q-point in fractional (crystal) coordinates. Or a np.ndarray of shape(N,3) with N
        q-points.

    Returns
    -------
    q_points : np.ndarray | pint.Quantity
        A (27*N, 3)-shaped array containing the original q-point and its ±1-shifted images
        along the three reciprocal directions. Units are preserved if input had units.

    Raises
    ------
    TypeError
        If `q_point` is a Quantity but not in crystal units (i.e., dimensionless reciprocal).
    """
    # Validate units if pint.Quantity
    if isinstance(q_point, ureg.Quantity):
        if (
            q_point.dimensionality != ureg.crystal.dimensionality
            and q_point.dimensionality != (1 / ureg.crystal).dimensionality
        ):
            raise TypeError(
                "If q_point has units, they must have crystal or 1/crystal dimensionality."
            )
        units = q_point.units
        q_point = q_point.magnitude
    else:
        q_point = np.array(q_point)
        units = 1

    if len(q_point.shape) == 1:
        output = np.array([q_point])
    else:
        output = np.array(q_point)
    for i in range(3):
        new_points = []
        for point in output:
            for delta in [-1, 1]:
                shifted = point.copy()
                shifted[i] += delta
                new_points.append(shifted)
        output = np.vstack([output, new_points])

    return output * units


def amplitude2order_parameter(
    amplitudes: ureg.Quantity | list[complex],
    masses: ureg.Quantity | list[float],
    displacements: list[np.ndarray],
) -> ureg.Quantity:
    """
    Convert amplitudes of normalized displacements into proper order parameters with mass scaling.

    Given phonon mode amplitudes applied to mass-normalized displacement vectors, this function
    returns the associated order parameters with physical units of length × sqrt(mass). The relation used is:

        q_s = A × sqrt(Σ_i M_i |ε_i^s|²)

    Parameters
    ----------
    amplitudes : ureg.Quantity or list[complex]
        Scalar amplitudes for each phonon distortion mode (with units of length or dimensionless).

    masses : ureg.Quantity or list[float]
        Atomic masses (usually in 2m_e units), one per atom in the unit cell.

    displacements : list[np.ndarray]
        Normalized displacement vectors for each mode. Each element has shape (N_atoms, 3)
        and is assumed to be mass-normalized (as in QE output).

    Returns
    -------
    ureg.Quantity
        Order parameters with units of length × sqrt(mass), one per mode.
    """
    # Strip units
    units = 1
    if isinstance(amplitudes, ureg.Quantity):
        units *= amplitudes.units
        amplitudes = amplitudes.magnitude
    if isinstance(masses, ureg.Quantity):
        units *= np.sqrt(1 * masses.units)
        masses = masses.magnitude

    # Formats amplitudes and displacements as np.ndarrays
    if isinstance(amplitudes, int | float | complex):
        amplitudes = np.array([amplitudes])
    else:
        amplitudes = np.array(amplitudes)
    if not isinstance(displacements, list):
        displacements = [displacements]

    # QE normalization constant: sum_i M_i * |ε_i|^2
    if len(amplitudes.shape) == 2:
        NN = [0] * amplitudes.shape[1]
    else:
        NN = [0] * len(amplitudes)

    for i in range(len(NN)):
        for j in range(len(masses)):
            NN[i] += masses[j] * (np.linalg.norm(displacements[i][j])) ** 2

    order_parameter = amplitudes * np.sqrt(NN)
    return order_parameter * units


def cumulative_integral(x: np.ndarray, y: np.ndarray):
    """
    Compute the cumulative integral of a function defined by discrete x and y values
    (handles pint units).

    Parameters
    ----------
    x : np.ndarray | ureg.Quantity
        1D array of x-values (must be increasing).
    y : np.ndarray | ureg.Quantity
        1D array of y-values evaluated at the x-points.

    Returns
    -------
    I : ndarray
        1D array of cumulative integral values at each x, starting from zero.

    Notes
    -----
    It uses scipy's cumulative_trapezoid method.
    """
    from scipy import integrate

    units = 1
    if isinstance(x, ureg.Quantity):
        units *= x.units
        x = x.magnitude
    if isinstance(y, ureg.Quantity):
        units *= y.units
        y = y.magnitude

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional arrays.")
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("x values must be strictly increasing.")

    Y = np.hstack([0, integrate.cumulative_trapezoid(y, x)])
    return Y * units


def _point_to_segment_distance(
    point: np.ndarray, endpoint_a: np.ndarray, endpoint_b: np.ndarray
) -> float:
    """
    Compute the Euclidean distance from a point to a line segment in 3D.

    The function returns the shortest distance from `point` to the line segment
    defined by the endpoints `endpoint_a` and `endpoint_b`. All inputs must be
    3-element numpy arrays representing Cartesian coordinates.

    Parameters
    ----------
    point : np.ndarray
        Cartesian coordinates of the point (shape: (3,)).
    endpoint_a : np.ndarray
        Cartesian coordinates of the first segment endpoint (shape: (3,)).
    endpoint_b : np.ndarray
        Cartesian coordinates of the second segment endpoint (shape: (3,)).

    Returns
    -------
    float
        The shortest distance from the point to the segment.
    """
    # Tangent vector of the segment, normalized
    segment_vec = endpoint_b - endpoint_a
    segment_dir = segment_vec / np.linalg.norm(segment_vec)

    # Signed projections onto the line extension
    dist_a = np.dot(endpoint_a - point, segment_dir)
    dist_b = np.dot(point - endpoint_b, segment_dir)

    # Clamp to segment endpoints if projection is outside
    parallel_dist = max(dist_a, dist_b, 0)

    # Orthogonal (perpendicular) component
    perpendicular_vec = np.cross(point - endpoint_a, segment_dir)

    return np.hypot(parallel_dist, np.linalg.norm(perpendicular_vec))


def wrap_fractional(
    coords: np.ndarray | ureg.Quantity,
    center: np.ndarray | float = 0.0,
) -> np.ndarray | ureg.Quantity:
    """
    Wrap fractional coordinates into a periodic unit cell centered at `center`.

    The wrapping interval is:
        [center - 0.5, center + 0.5)

    Parameters
    ----------
    coords : np.ndarray or pint.Quantity
        Fractional coordinates of shape (..., d).
    center : float or np.ndarray, optional
        Center of the wrapping interval. Can be a scalar or
        array broadcastable to `coords`. Default is 0.

    Returns
    -------
    np.ndarray or pint.Quantity
        Wrapped coordinates with same shape and units as input.
    """
    # Extract units if needed
    if isinstance(coords, ureg.Quantity):
        units = coords.units
        x = np.asarray(coords.magnitude, dtype=float)
    else:
        units = None
        x = np.asarray(coords, dtype=float)

    # Ensure center is array for broadcasting
    center = np.asarray(center, dtype=float)

    # Wrap to [center - 0.5, center + 0.5)
    x_wrapped = x - np.floor(x - center + 0.5)

    # Restore units
    if units is not None:
        return x_wrapped * units
    return x_wrapped


def find_little_group(
    kpoints: np.ndarray | ureg.Quantity,
    symmetries: list,
    tol: float = 1e-8,
    mod_G: bool = False,
) -> list[np.ndarray]:
    """
    Compute the little group for each input k-point.

    For each k (row vector), returns the indices i of symmetry operations R_i
    that leave k invariant, using the row-vector convention k' = R^{-T} k.

    Parameters
    ----------
    kpoints : np.ndarray | ureg.Quantity, shape (N, DIM)
        Input k-points as row vectors. If a Quantity, units are preserved internally.
        If `mod_G=True`, kpoints must be in crystal units (2π/crystal).
    symmetries : list
        List of symmetry objects, each with attribute `R` (DIM×DIM rotation matrix).
        Translations (if present) are ignored for k.
    tol : float, optional
        Numerical tolerance for invariance checks. Default 1e-8.
    mod_G : bool, optional
        If True, test invariance modulo reciprocal lattice vectors:
        (k @ inv(R)) = k (mod 1). If False, test direct equality
        (k @ inv(R) − k) = 0 within `tol`. Default is False.

    Returns
    -------
    little_group : list[np.ndarray]
        A list of length N. Each entry is an array of symmetry indices that
        leave the corresponding k-point invariant (under the chosen check).

    Raises
    ------
    ValueError
        If `kpoints` are not shape (N, 3), or if `mod_G=True` and units are not 2π/crystal.

    Notes
    -----
    - Row-vector convention: k' = k @ inv(R).
    - If `mod_G=True`, invariance is tested modulo 1 per component (crystal units).
    """
    # Units handling
    if isinstance(kpoints, ureg.Quantity):
        units = kpoints.units
        kpts = np.asarray(kpoints.magnitude, dtype=float)
    else:
        units = 1
        kpts = np.asarray(kpoints, dtype=float)

    if kpts.ndim == 1:
        kpts = np.asarray([kpoints], dtype=float)
    if kpts.ndim != 2:
        raise ValueError("kpoints must be of shape (N, DIM)")

    little_group = []

    for k in kpts:
        invariant_ops = []
        # Pre-wrap reference if using mod_G
        if mod_G:
            if units != 1 and units != ureg("_2pi/crystal"):
                raise ValueError(
                    "mod_G=True requires kpoints in crystal units (2π/crystal). "
                    "Convert your k-points before calling or set mod_G=False."
                )
            k_wr = wrap_fractional(k)

        for i, sym in enumerate(symmetries):
            R = np.asarray(sym.R, dtype=float)
            kR = rotate(k, R, covariant=True)
            if mod_G:
                kR_wr = wrap_fractional(kR)
                d = kR_wr - k_wr
                # modulo-1 closeness: subtract nearest integers
                d = d - np.round(d)
            else:
                d = kR - k
            ok = np.all(np.abs(d) <= tol)

            if ok:
                invariant_ops.append(i)
        little_group.append(np.asarray(invariant_ops, dtype=int))
    return little_group


def symmetry_orbit_kpoints(
    kpoints: np.ndarray | ureg.Quantity,
    symmetries: list,
    tol: float = 1e-6,
    mod_G: bool = True,
) -> SimpleNamespace:
    """
    Apply all symmetry rotations to a set of k-points (row vectors) and return
    the unique set.

    This treats k-points as row vectors and multiplies on the right by the
    rotation matrices (k' = k @ inv(R)). It preserves first occurrence so identity-
    generated points are kept when duplicates occur.

    Parameters
    ----------
    kpoints : np.ndarray | ureg.Quantity, shape (N, DIM)
        Input k-points (rows). If a Quantity, units are preserved. For mod_G=True,
        points are assumed in crystal units (2π/crystal).
    symmetries : list
        List of symmetry objects, each with attribute R (DIM×DIM rotation matrix).
        The first element must be the identity symmetry.
    tol : float, optional
        Numerical tolerance used for detecting duplicates (after rounding). Default 1e-8.
    mod_G : bool, optional
        If True (default), identify k ≡ k + G via wrapping to [-0.5, 0.5):
        k -> k - floor(k + 0.5). This maps -0.5 to +0.5 so boundary points are
        handled consistently.

    Returns
    -------
    orbit : SimpleNamespace
        - kpoints : np.ndarray | pint.Quantity, shape (M, 3)
            Unique symmetry-expanded k-points (within `tol`), same units as input.
        - sym : np.ndarray, shape (M,)
            Symmetry index (into `symmetries`) of the representative kept.
        - origin : np.ndarray, shape (M,)
            Original input k-point index from which the representative was generated.
        - weights : np.ndarray, shape (N,)
            Normalized weights per original k-point, computed as the fraction of orbit
            points whose representative originated from each input k-point (sums to 1).

    Raises
    ------
    ValueError
        If mod_G=True and units are not 2π/crystal.

    Notes
    -----
    - WARNING: This may generate more points than the ones on the grid, for that use
    `expand_irreducible_bz`.
    - Row-vector convention: k' = k @ inv(R).
    - First occurrence order is preserved when removing duplicates (identity wins).
    - Little group checks use full equivalence: (k @ R) ≈ k, not k + G.
    """

    # Units handling
    if isinstance(kpoints, ureg.Quantity):
        units = kpoints.units
        kpts = np.asarray(kpoints.magnitude, dtype=float)
    else:
        units = 1
        kpts = np.asarray(kpoints, dtype=float)

    if mod_G and units != 1 and units != ureg("_2pi/crystal"):
        raise ValueError(
            "mod_G=True requires kpoints in crystal units (2π/crystal). "
            "Convert your k-points before calling or set mod_G=False."
        )

    if kpts.ndim == 1:
        kpts = np.asarray([kpoints], dtype=float)
    if kpts.ndim != 2:
        raise ValueError("kpoints must be of shape (N, DIM)")

    # Expand via symmetries (identity first)
    Nk = len(kpts)
    expanded = []
    idx_pairs = []
    for i, sym in enumerate(symmetries):
        expanded.append(rotate(kpts, sym.R, covariant=True))
        idx_pairs.append(
            np.column_stack(
                (np.full(Nk, i), np.arange(Nk))  # i repeated Nk times  # j = 0 ... Nk-1
            )
        )
    expanded = np.concatenate(expanded)  # shape (S*N, DIM)
    idx_pairs = np.concatenate(idx_pairs)
    # Wrap to [-0.5,0.5)
    if mod_G:
        expanded = wrap_fractional(expanded)

    # Order-preserving uniqueness via rounding to tolerance
    rounded = np.round(expanded / tol) * tol
    # Wrap to [-0.5,0.5)
    if mod_G:
        rounded = wrap_fractional(
            rounded,
        )

    # Find unique values and store in keep_idx
    seen = set()
    keep_idx = []
    for idx, row in enumerate(rounded):
        key = tuple(row)
        if key not in seen:
            seen.add(key)
            keep_idx.append(idx)

    unique_kpts = expanded[keep_idx]
    idx_map = np.array([idx_pairs[i] for i in keep_idx], dtype=int)

    # Compute origin weights: count how many orbit points came from each original k-point
    N_orig = kpts.shape[0]
    origin_counts = np.bincount(idx_map[:, 1], minlength=N_orig)
    origin_weights = origin_counts / origin_counts.sum()

    return SimpleNamespace(
        kpoints=unique_kpts * units,
        sym=idx_map[:, 0],
        origin=idx_map[:, 1],
        weights=origin_weights,
    )


def expand_irreducible_bz(
    kpoints: np.ndarray | ureg.Quantity,
    grid: list[int],
    symmetries: list,
    tol: float = 1e-5,
) -> SimpleNamespace:
    """
    Expand a set of irreducible k-points to the full Brillouin zone using symmetry operations.

    Given a set of k-points in the irreducible Brillouin zone (IBZ), this function applies
    all symmetry operations to generate the full set of k-points on a uniform grid, matching
    them against a reference grid and avoiding duplicates.

    Parameters
    ----------
    kpoints : np.ndarray | pint.Quantity
        Array of shape (Nk, d) containing k-points in fractional (crystal) coordinates.
        If a Quantity, units must be compatible with reciprocal lattice units (2π/crystal).
    grid : list[int]
        Number of grid points along each reciprocal lattice direction, e.g. [Nx, Ny, Nz].
        Defines the full Brillouin zone sampling grid.
    symmetries : list
        List of symmetry operations. Each symmetry must have:
            - sym.R : rotation matrix (d × d)
            - sym.units : must be in crystal units (ureg.crystal)
    tol : float, optional
        Tolerance used to identify matching k-points modulo reciprocal lattice vectors.

    Returns
    -------
    SimpleNamespace
        Object with attributes:
            kpoints : np.ndarray or pint.Quantity
                Expanded k-points covering the full Brillouin zone.
            sym : np.ndarray
                Index of the symmetry operation used to generate each k-point.
            origin : np.ndarray
                Index of the original irreducible k-point from which each expanded point originates.

    Raises
    ------
    ValueError
        If symmetry operations are not expressed in crystal units.
    ValueError
        If kpoints are not in crystal reciprocal units (2π/crystal).
    ValueError
        Not all grid points are matched by symmetry expansion.

    Notes
    -----
    - It works in batches of 3000 target-grid k-points in order to avoid overuse of RAM.
    """

    # Units handling
    if isinstance(kpoints, ureg.Quantity):
        units = kpoints.units
        kpts = np.asarray(kpoints.magnitude, dtype=float)
    else:
        units = 1
        kpts = np.asarray(kpoints, dtype=float)

    if any(sym.units != ureg.crystal for sym in symmetries):
        raise ValueError("Symmetries must be in crystal units.")

    if units != 1 and units != ureg("_2pi/crystal"):
        raise ValueError("kpoints are required in crystal units (2π/crystal).")

    nkx, nky, nkz = grid
    grid_step = (1.0 / np.asarray(grid))[None, :]

    grid_points = grid_generator(grid, periodic=True)

    syms = np.zeros((len(grid_points),), dtype=np.int32)
    origins = np.zeros((len(grid_points),), dtype=np.int64)
    found = np.zeros_like(origins, dtype=bool)

    for i, sym in enumerate(symmetries):
        # Rk are the images of the IBZ points by the symmetry i
        Rk = rotate(kpts, sym.R, covariant=True)  # no need to wrap yet

        # Rk_snapped are the closestpoints nodes of the grid
        Rk_snapped = np.round(Rk / grid_step) * grid_step

        # True if the point is close enough to an infinite grid node
        mask = np.abs(Rk - Rk_snapped).max(axis=1) < tol

        Rk_match = Rk_snapped[mask]
        
        # Compute the integer coordinates
        Rk_match = np.mod(Rk_match, 1.0) # wrap to [0, 1)
        # convert to number of steps from 0
        ijk = np.round(Rk_match / grid_step).astype(int)

        # The grid is generated in the order of:
        # [(i, j, k) for j in range(ny) for i in range(nx) for k in range(nz)]
        # so the summation here reflect that
        indices = ijk.dot([nkz, nkx * nkz, 1])

        # only keep the new points in the mask, this way we always keep the
        # symmetry of the first match
        mask[mask] &= ~found[indices]  # mask = mask AND not found
        indices = indices[~found[indices]]

        origins[indices] = np.arange(0, len(Rk))[mask]
        syms[indices] = i
        found[indices] = True

        if found.sum() == nkx * nky * nkz:  # as soon as we hit all points we return
            return SimpleNamespace(
                kpoints=grid_points * units,
                sym=syms,
                origin=origins,
            )

    raise ValueError(
        f"Could only recover {found.sum()} out of the {nkx * nky * nkz} points of the grid from the reduced set of points"
    )


def auto_kgrid(
    lattice: np.ndarray | ureg.Quantity,
    delta_k: float | ureg.Quantity = 0.1 / ureg.ang,
    n_atoms: int = None,
    kppra: int | float = 9000,
    force_even: bool | list[bool] = False,
    force_odd: bool | list[bool] = False,
    verbose: bool = False,
) -> list[int]:
    """
    Suggest a Monkhorst–Pack grid [n1, n2, n3] using either:
      1) A target reciprocal-space spacing (delta_k), or
      2) A target KPPRA (k-points per reciprocal atom) product:
         Nk_total × n_atoms ≈ kppra  =>  Nk_total ≈ kppra / n_atoms.

    The grid is distributed proportionally to the magnitudes of the reciprocal
    lattice vectors (to keep the spacing as isotropic as possible). Optionally,
    each grid component can be coerced to even or odd after the estimate, either
    globally (bool) or per direction with a 3-element list of bools.

    Parameters
    ----------
    lattice : np.ndarray | pint.Quantity
        Direct lattice vectors in rows. If a Quantity is given, it must carry length units
        (e.g., angstrom). Unit handling is applied consistently to `delta_k`.  delta_k : float | pint.Quantity, optional
        Target k-point spacing in reciprocal space, with units of [1/length]. Default is 0.1 Å⁻¹.
        Used when `kpoints_per_atom` is None.
    delta_k : float | ureg.Quantity, optional
        Target reciprocal-space spacing. Default being 0.1 / ureg.ang.
    n_atoms : int | None, optional
        Number of atoms in the cell. If provided, KPPRA mode is used, with
        Nk_total ≈ kppra / n_atoms, instead of constant spacing.
    kppra : int | float, optional
        Target product Nk_total × n_atoms (k-points per reciprocal atom).
        Default 9000. Only used when `n_atoms` is provided (KPPRA mode).
    force_even : bool | list[bool], optional
        Postprocessing to coerce each grid component to even. If a single bool is given, it applies to all
        directions. If a 3-length list/tuple is given (e.g., [True, False, True]), it applies per direction.
        Cannot be True where `force_odd` is True in the same direction.
    force_odd : bool | list[bool], optional
        Postprocessing to coerce each grid component to odd. Same semantics as `force_even` (bool or 3-length).
        Cannot be True where `force_even` is True in the same direction.
    verbose : bool, optional
        If True, a final report is given. Default is False.

    Returns
    -------
    kgrid : list[int]
        Integer array [n1, n2, n3] with a minimum of 1 in each direction.

    Raises
    ------
    TypeError
        - If `lattice` has units but `delta_k` is unitless (or vice versa).
        - If `lattice` units are not length units.
        - If `kpoints_per_atom` is not None but `n_atoms` is None.
    ValueError
        - If `force_even` and `force_odd` are both True in any given direction.

    Notes
    -----
    - Constant Δk (default) is physically consistent across different cells/supercells.
    - KPPRA mode (Nk_total × n_atoms ≈ kppra) is a common consistency rule across materials datasets.
    """
    DIM = len(lattice)

    # Helper to broadcast bool or list[bool] to a 3-length list[bool]
    def _as_bool(x, name: str) -> list[bool]:
        if isinstance(x, bool):
            return [x, x, x]
        if not isinstance(x, list):
            raise TypeError(f"`{name}` must be a bool or a list[bool] of length {DIM}.")
        if len(x) != DIM:
            raise TypeError(f"`{name}` list must have length {DIM}.")
        if not all(isinstance(v, bool) for v in x):
            raise TypeError(f"All entries of `{name}` must be bool.")
        return x

    even = _as_bool(force_even, "force_even")
    odd = _as_bool(force_odd, "force_odd")
    if any(e and o for e, o in zip(even, odd)):
        raise ValueError(
            "`force_even` and `force_odd` cannot both be True in the same direction."
        )

    if isinstance(lattice, ureg.Quantity):
        # Lattice must carry length units
        if not lattice.check(ureg.meter):
            raise TypeError(
                "`lattice` must have length units if provided as a Quantity."
            )
        if not isinstance(delta_k, ureg.Quantity):
            raise TypeError(
                "`delta_k` must have units of 1/length when `lattice` has units."
            )
        # Convert delta_k to 1/(lattice units)
        delta_k = delta_k.to(1 / lattice.units).magnitude
        units = lattice.units
        lattice = lattice.magnitude
    else:
        units = 1
        if isinstance(delta_k, ureg.Quantity):
            delta_k = delta_k.magnitude

    if n_atoms is not None:
        target_kpts = kppra / n_atoms
    else:
        target_kpts = ((2 * np.pi) ** DIM / np.linalg.det(lattice)) / delta_k**DIM

    kmod = [np.linalg.norm(k) for k in reciprocal_basis(lattice).magnitude]
    kmod = kmod / (np.prod(kmod) ** (1 / DIM))
    kgrid = np.around(np.ones(DIM) * (target_kpts ** (1 / DIM)) * kmod).astype(int)
    # Force k to be at least 1
    kgrid = [int(k) if k > 0 else 1 for k in kgrid]

    # Postprocessing: enforce even/odd if requested (minimal addition)
    for i in range(DIM):
        if even[i]:
            if kgrid[i] % 2 != 0:
                kgrid[i] += 1
        elif odd[i]:
            if kgrid[i] % 2 == 0:
                # make odd, ensuring it stays >= 1
                kgrid[i] = kgrid[i] + 1 if kgrid[i] >= 1 else 1

    if verbose:
        total = np.prod(kgrid)
        Kvolume = (2 * np.pi) ** DIM / np.linalg.det(lattice)
        delta_k = (Kvolume / total) ** (1 / DIM) * (1 / units)
        print(f"Total number of k-points: {total}")
        print(f"Averge Δk distance: {delta_k}")
        if n_atoms is not None:
            print(f"K-points per atom: {float(total)/n_atoms}")
    return kgrid


def eigen_projection(
    spectra_in: np.ndarray,
    spectra_out: np.ndarray,
    eigenvalues_out: np.ndarray | ureg.Quantity = None,
) -> np.ndarray:
    """
    Projects initial spectra onto final spectra and optionally groups projections based
    on a set of eigenvalues.

    This function calculates the projection coefficients or their squared magnitudes
    between two sets of spectra expressed in orthonormal bases. If eigenvalues are provided,
    the squared projections of degenerate eigenvalues are summed and assigned to one of them.

    Parameters
    ----------
    spectra_in : np.ndarray
        Initial spectra (e.g., polarization vectors or waveform) to be projected, expressed in
        an orthonormal basis. Shape (N,...), where N is the total number of vectors in `spectra_in`.
    spectra_out : np.ndarray
        Final spectra onto which the initial spectra are projected, also expressed in an orthonormal
        basis. Shape (M,...) where M is the total number of vectors in `spectra_out`.
    eigenvalues_out : np.ndarray | ureg.Quantity, optional
        Set of eigenvalues corresponding to the final spectra. If provided, the projections
        for degenerate eigenvalues are first squared and summed, and after assigned to a representative
        of the degenerate subspace.

    Returns
    -------
    projections : np.ndarray(N,M)
        Projection coefficients (complex) if `eigenvalues_out` is None. Otherwise, the squared
        magnitudes of projection coefficients are returned.

    Notes
    -----
    * If `eigenvalues_out` is provided, the returned values are abs(coef)^2, not the coefficients
      themselves.
    * A sanity check is performed to ensure the projections' norm should be approximately 1.
    """
    if spectra_in.ndim == spectra_out.ndim:
        N = spectra_in.shape[0]  # Number of spectra vectors
    else:
        N = 1
        spectra_in = [spectra_in]  # Make it iterable

    # Initialize the projection matrix
    M = spectra_out.shape[0]
    PROJ = np.zeros((N, M), dtype=complex)

    # Calculate projection coefficients between initial and final spectra
    for i, si in enumerate(spectra_in):
        for j, so in enumerate(spectra_out):
            PROJ[i, j] = np.matmul(si.flatten(), so.flatten())

    # Perform a sanity check to ensure the projections have a norm approximately equal to 1
    for i, projection in enumerate(PROJ):
        norm_value = np.around(np.linalg.norm(projection), 4)
        if norm_value != 1:
            warnings.warn(
                f"Projections norm is {norm_value}. Not close to 1 beyond numerical error of 1E-4",
                UserWarning,
            )

    # If eigenvalues_out is provided, consolidate projections for degenerate eigenvalues
    if eigenvalues_out is not None:
        if isinstance(eigenvalues_out, ureg.Quantity):
            eigenvalues_out = eigenvalues_out.magnitude
        # Change format of the main output to squared magnitudes
        PROJ = np.abs(PROJ) ** 2
        # Identify and group degenerate eigenvalues
        degeneracies = [
            np.where(eigenvalues_out == x)[0] for x in np.unique(eigenvalues_out)
        ]
        for i in range(PROJ.shape[0]):
            for deg in degeneracies:
                if len(deg) > 1:
                    for j in deg[1:]:
                        PROJ[i, deg[0]] += PROJ[i, j]
                        PROJ[i, j] = 0  # Zero out contributions from repeated indices

    return PROJ
