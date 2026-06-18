"""
YAIV | yaiv.spectrum
====================

This module defines core classes for representing and plotting the eigenvalue spectrum
of periodic operators, such as electronic bands or phonon frequencies, across a set of
k-points. It also supports reciprocal lattice handling and coordinate transformations.

The classes in this module can be used independently or as output containers from
grepping functions.

Classes
-------
Spectrum
    General container for k-resolved eigenvalue spectra (e.g., bands, phonons).
    Supports plotting, DOS calculation, and band visualizations.
    Provides:
    - get_DOS(...): Computes the density of states via Gaussian or Methfessel–Paxton smearing.
    - plot(...): Plots the band structure along a cumulative k-path.
    - plot_fat(...): Fat-band style scatter plot for visualizing weights/projections over bands.
    - plot_color(...): Color-gradient line plot for weights/projections over bands.

ElectronBands
    Specialized `Spectrum` subclass for electronic band structures extracted from files.
    Adds Fermi level, number of electrons, and automatic parsing.

PhononBands
    Specialized `Spectrum` subclass for phonon spectra extracted from files.

Density
    General container for a scalar density defined on a 1D grid.
    Provides:
    - from_data(): Initialize a Density object with kernel-broadening from a set of discreate data.
    - integrate(): Computes the integral of the density or finds the intagration limit
                    corresponding to certain integral value.
    - plot(): Plots the density curve with optional fill and orientation options.

Private Utilities
-----------------
_Has_lattice
    Mixin that adds lattice handling capabilities.

_Has_kpath
    Mixin that adds support for k-path functionalities.
    Provides:
    - get_1Dkpath(self, patched=True): Provides a one dimensional Kpath

Examples
--------
>>> from yaiv.spectrum import ElectronBands
>>> bands = ElectronBands("data/qe/Si.bands.pwo")
>>> bands.eigenvalues.shape
(100, 32)
>>> bands.plot()
(Figure)

See Also
--------
yaiv.grep     : Low-level data extractors used to populate spectrum objects
yaiv.utils    : Basis universal utilities and vector transformations
yaiv.defaults : Configuration and default plotting values
"""

import warnings
from types import SimpleNamespace

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, LineCollection

from yaiv.defaults.config import ureg, defaults, apply_plot_defaults
from yaiv.defaults.config import plot_defaults as pdft

from yaiv import utils as ut

__all__ = ["Spectrum" "ElectronBands", "PhononBands", "Density"]


class _Has_lattice:
    """
    Mixin that provides lattice-related functionality:
    loading a lattice, computing its reciprocal basis, and keeping them syncd.

    Attributes
    ----------
    lattice : np.ndarray | ureg.Quantity
        3x3 matrix of direct lattice vectors in [length] units.
    k_lattice : np.ndarray | ureg.Quantity
        3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
    volume : np.ndarray | ureg.Quantity
        Volume of the cell defined by `lattice`.
    """

    def __init__(
        self,
        lattice: np.ndarray | ureg.Quantity = None,
        k_lattice: np.ndarray | ureg.Quantity = None,
        alat: np.ndarray | ureg.Quantity = None,
    ):
        """
        Initialize the _Has_lattice object from either the real or reciprocal space lattice.

        Parameters
        ----------
        lattice : np.ndarray | ureg.Quantity
            3x3 matrix of direct lattice vectors in [length] units.
        k_lattice : np.ndarray | ureg.Quantity
            3x3 matrix of reciprocal lattice vectors in [length]⁻¹ units.
        alat : np.ndarray | ureg.Quantity
            `alat` factor for conversions, dimensions of [length/alat].
        """
        self._lattice = self._k_lattice = None
        if lattice is not None:
            self._lattice = lattice
            self._k_lattice = ut.reciprocal_basis(self._lattice)
        elif k_lattice is not None:
            self._k_lattice = k_lattice
            self._lattice = ut.reciprocal_basis(self._k_lattice)
        self.alat = alat

    @property
    def lattice(self):
        return self._lattice

    @property
    def k_lattice(self):
        return self._k_lattice

    @property
    def volume(self):
        lat, units = ut._split_units(self.lattice)
        return np.linalg.det(lat) * (units**3)

    @lattice.setter
    def lattice(self, value):
        self._lattice = value
        self._k_lattice = ut.reciprocal_basis(value)

    @k_lattice.setter
    def k_lattice(self, value):
        self._k_lattice = value
        self._lattice = ut.reciprocal_basis(value)


class _Has_kpath:
    """
    Mixin that provides kpath-related functionality:

    Attributes
    ----------
    kpath : SimpleNamespace | np.ndarray
        A namespace with attributes `path`(ndarray) and `labels`(list)
        or just a ndarray.

    Methods
    -------
    get_1Dkpath()
        Computes the 1D cumulative k-path from the k-point coordinates.
    """

    def __init__(self, kpath: SimpleNamespace | np.ndarray = None):
        """
        Initialize the _Has_kpath object from a SimpleNamespace as given by
        `yaiv.grep.kpath`.

        Parameters
        ----------
        kpath : SimpleNamespace | np.ndarray
            A namespace with attributes `path`(ndarray) and `labels`(list)
            or just a ndarray.
        """
        self.kpath = kpath

    def get_1Dkpath(self, patched=True) -> np.ndarray:
        """
        Computes the 1D cumulative k-path from the k-point coordinates.

        Parameters
        ----------
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).

        Returns
        -------
        kpath : np.ndarray
            The 1D cumulative k-path from the k-point coordinates.
        """
        if self.kpoints is None:
            raise ValueError("kpoints are not defined.")

        # Strip units for math, retain them for reapplication later
        if isinstance(self.kpoints, ureg.Quantity):
            kpoints = self.kpoints
            if "crystal" in kpoints.units._units and self.k_lattice is not None:
                kpoints = ut.cryst2cartesian(self.kpoints, self.k_lattice)
                kpoints = kpoints.to("_2pi/ang")
            if (
                "alat" in kpoints.units._units
                and self.k_lattice is not None
                and self.alat is not None
            ):
                kpoints = kpoints / self.alat
                kpoints = kpoints.to("_2pi/ang")
            units = kpoints.units
            kpts_val = kpoints.magnitude
        else:
            units = 1
            kpts_val = self.kpoints

        # Compute segment lengths
        delta_k = np.diff(kpts_val, axis=0)
        segment_lengths = np.linalg.norm(delta_k, axis=1)
        if patched:
            # Define discontinuities as large jumps relative to minimum segment
            threshold = np.min(segment_lengths[segment_lengths >= 1e-5]) * 10
            segment_lengths = np.where(segment_lengths > threshold, 0, segment_lengths)
        kpath = np.concatenate([[0], np.cumsum(segment_lengths)])
        return kpath * units


class Spectrum(_Has_lattice, _Has_kpath):
    """
    General class for storing the eigenvalues of a periodic operator over k-points.

    This can represent band structures, phonon spectra, or eigenvalues of other operators.
    It is a subclass of `_Has_lattice` and `_Has_kpath` mixing classes.

    Attributes
    ----------
    eigenvalues : np.ndarray | ureg.Quantity, optional
        Array of shape (nkpts, neigs), e.g., energy or frequency values.
    kpoints : np.ndarray | ureg.Quantity, optional
        Array of shape (nkpts, 3) with k-points.
    weights : np.ndarray, optional
        Optional weights for each k-point.
    DOS : DOS, optional
        - vgrid : np.ndarray | pint.Quantity
            Array of shape (steps,) with the eigenvalue units.
        - DOS : np.ndarray
            Array of shape (steps,) with the corresponding DOS values.

    Methods
    -------
    get_DOS(...)
        Compute a density of states (DOS) for the set of eigenvalues.
    def plot(...)
        Plot the spectrum over a cumulative k-path.
    def plot_fat(...)
        Fat-band style plotting for weights over a cumulative k-path.
    def plot_color(...)
        Color gradient line-style for weights over a cumulative k-path.
    """

    def __init__(
        self,
        eigenvalues: np.ndarray | ureg.Quantity = None,
        kpoints: np.ndarray | ureg.Quantity = None,
        weights: list | np.ndarray = None,
        lattice: np.ndarray | ureg.Quantity = None,
        k_lattice: np.ndarray | ureg.Quantity = None,
        kpath: SimpleNamespace | np.ndarray = None,
        alat: np.ndarray | ureg.Quantity = None,
    ):
        """
        Initialize Spectrum object.

        Parameters
        ----------
        eigenvalues : np.ndarray | ureg.Quantity, optional
            Array of shape (nkpts, neigs), e.g., energy or frequency values.
        kpoints : np.ndarray | ureg.Quantity, optional
            Array of shape (nkpts, 3) with k-points.
        weights : np.ndarray, optional
            Optional weights for each k-point.
        lattice : np.ndarray | ureg.Quantity, optional
            3x3 matrix of direct lattice vectors in [length] units.
        k_lattice : np.ndarray | ureg.Quantity, optional
            3x3 matrix of reciprocal lattice vectors in 2π[length]⁻¹ units.
            Will be ignored when defining the spectrum if lattice is given.
        kpath : SimpleNamespace | np.ndarray, optional
            A namespace with attributes `path`(ndarray) and `labels`(list)
            or just a ndarray.
        alat : np.ndarray | ureg.Quantity
            `alat` factor for conversions, dimensions of [length/alat].
        """
        self.eigenvalues = eigenvalues
        self.kpoints = kpoints
        self.weights = weights
        _Has_lattice.__init__(self, lattice, k_lattice, alat)
        _Has_kpath.__init__(self, kpath)
        self.DOS = Density()

    def get_DOS(
        self,
        center: float | ureg.Quantity = None,
        window: float | list[float] | ureg.Quantity = None,
        smearing: float | ureg.Quantity = None,
        steps: int = None,
        order: int = 0,
        cutoff_sigmas: float = defaults.cutoff_sigmas,
    ):
        """
        Compute a density of states (DOS) using Gaussian, Fermi-Dirac or Methfessel-Paxton (MP)
        smearing of any order.

        This implementation uses a MP delta function to smear each eigenvalue and
        returns the total DOS over an eigenvalue grid. Since the default order is zero,
        it defaults to using a Gaussian distribution.

        Parameters
        ----------
        center : float | pint.Quantity, optional
            Center for the energy window (e.g., Fermi energy). Default is zero.
        window : float | list[float] | pint.Quantity, optional
            Value window for the DOS. If float, interpreted as symmetric [-window, window].
            If list, used as [Vmin, Vmax]. If None, the eigenvalue range (± smearing * cutoff_sigmas) is used.
        smearing : float | pint.Quantity, optional
            Smearing width in the same unit dimension as eigenvalues. Default is (window_size/200).
        steps : int, optional
            Number of grid points for DOS sampling. Default is 4 * (window_size/smearing).
        order : int, optional
            Kernel type: 0=Gaussian; >0=Methfessel–Paxton order; -1=Fermi-Dirac.
            Default is 0, which recovers a Gaussian smearing.
        cutoff_sigmas : float, optional
            Number of smearing widths to use for truncation (e.g., 3 means ±3σ).
            Default yaiv.defaults.defaults.cutoff_sigmas.

        Returns
        -------
        self.DOS : DOS
            - vgrid : np.ndarray | pint.Quantity
                Array of shape (steps,) with the eigenvalue units.
            - DOS : np.ndarray | pint.Quantity
                Array of shape (steps,) with the computed DOS values.

        Raises
        ------
        ValueError
            If eigenvalues shape is incorrect or weights do not match.
        """
        self.DOS = Density.from_data(
            x=self.eigenvalues,
            values=None,
            weights=self.weights,
            center=center,
            x_window=window,
            sigma=smearing,
            steps=steps,
            order=order,
            cutoff_sigmas=cutoff_sigmas,
        )

    def _pre_plot(
        self=None,
        ax=None,
        shift=None,
        bands=None,
        patched=True,
        weights=None,
        log=None,
        window=None,
    ):
        """
        Pre plotting tool to avoid code duplication.
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as colors

        apply_plot_defaults()

        # Handle units
        if shift is not None:
            quantities = [self.eigenvalues, shift]
            names = ["eigenvalues", "shift"]
            ut._check_unit_consistency(quantities, names)

        # Create fig if necessary
        if ax is None:
            fig, ax = plt.subplots()

        # Apply shift to eigenvalues
        eigen = self.eigenvalues - shift if shift is not None else self.eigenvalues
        kpath = self.get_1Dkpath(patched)
        length = kpath[-1]
        x = kpath / length

        band_indices = bands if bands is not None else range(eigen.shape[1])

        # Handle weights if present
        if weights is not None:
            W = weights.magnitude if isinstance(weights, ureg.Quantity) else weights
            window = (
                window.to(weights.units).magnitude
                if isinstance(window, ureg.Quantity)
                else window
            )
            if window is not None:
                vmin, vmax = window
            else:
                vals = W[:, band_indices]
                if not log:
                    vmin = np.min(vals)
                    vmax = np.max(vals)
                else:
                    positive = vals[vals > 0]
                    vmin = np.min(positive)
                    vmax = np.max(vals)
            if log:
                norm = colors.LogNorm(vmin=vmin, vmax=vmax)
            else:
                norm = colors.Normalize(vmin=vmin, vmax=vmax)
            return SimpleNamespace(
                ax=ax,
                x=x,
                eigen=eigen,
                band_indices=band_indices,
                weights=W,
                norm=norm,
            )
        else:
            return SimpleNamespace(
                ax=ax,
                x=x,
                eigen=eigen,
                band_indices=band_indices,
            )

    def plot(
        self,
        ax: Axes = None,
        shift: float | ureg.Quantity = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> (Axes, Axes):
        """
        Plot the spectrum over a cumulative k-path.

        Parameters
        ----------
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float | pint.Quantity, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
            Default is zero.
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).
        bands : list of int, optional
            Indices of the bands to plot. If None, all bands are plotted.
        **kwargs : dict
            Additional matplotlib arguments passed to `plot()`.

        Returns
        ----------
        ax : Axes
            The axes with the spectrum plot.
        """
        P = self._pre_plot(ax, shift, bands, patched)
        label = kwargs.pop("label", None)  # remove label from kwargs
        P.ax.plot(P.x, P.eigen[:, P.band_indices[0]], label=label, **kwargs)
        P.ax.plot(P.x, P.eigen[:, P.band_indices[1:]], **kwargs)

        P.ax.set_xlim(0, 1)
        return P.ax

    def plot_fat(
        self,
        weights: np.ndarray,
        window: list[float, float] | ureg.Quantity = None,
        ax: Axes = None,
        size_change: bool = False,
        alpha_change: bool = False,
        log: bool = False,
        shift: float | ureg.Quantity = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> (Axes, PathCollection):
        """
        Fat-band style plotting for weights over a cumulative k-path.

        These weights can represent projections over orbitals or other similar attributes.
        A point will be scattered at coordinates (k,E) with color, size, transparency related to the weights input.

        Parameters
        ----------
        weights : np.ndarray, pint.Quantity
            Array of shape (nkpts, neigs).
        window : list[float,float], optional
            Minimal and maximum values for the colormap of the weights.
            Default is minimal of maximal values for the set of weights.
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        size_change : bool, optional
            Whether the size of the dots should also change (linked to the window).
        alpha_change : bool, optional
            Whether the transparency (alpha) of the dots should also change (linked to the window).
        log : bool, optional
            If True logaritmic scaling is applied. Default is False.
        shift : float | pint.Quantity, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
            Default is zero.
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).
        bands : list of int, optional
            Indices of the bands to plot. If None, all bands are plotted.
        **kwargs : dict
            Additional matplotlib arguments passed to `scatter()`.

        Returns
        ----------
        ax : Axes
            The axes with the spectrum plot.
        scatter : matplotlib.collections.PathCollection
            The PathCollection for generating the colorbar.
        """
        P = self._pre_plot(
            ax=ax,
            shift=shift,
            bands=bands,
            patched=patched,
            weights=weights,
            log=log,
            window=window,
        )

        # Remove some labels from kwargs
        label = kwargs.pop("label", None)
        s = kwargs.pop("s", pdft.weights_s)
        alpha = kwargs.pop("alpha", 1)
        if alpha_change:
            alpha = np.clip(
                (P.weights - P.norm.vmin) / (P.norm.vmax - P.norm.vmin), 0, 1
            )
        else:
            alpha = np.ones(P.weights.shape)
        if size_change:
            s = (
                np.clip((P.weights - P.norm.vmin) / (P.norm.vmax - P.norm.vmin), 0, 1)
                * s
            )
        else:
            s = np.ones(P.weights.shape) * s

        scatter = P.ax.scatter(
            P.x,
            P.eigen[:, P.band_indices[0]],
            c=P.weights[:, P.band_indices[0]],
            s=s[:, P.band_indices[0]],
            alpha=alpha[:, P.band_indices[0]],
            norm=P.norm,
            label=label,
            edgecolors="none",
            **kwargs,
        )
        for i in P.band_indices[1:]:
            P.ax.scatter(
                P.x,
                P.eigen[:, i],
                c=P.weights[:, i],
                s=s[:, i],
                alpha=alpha[:, i],
                norm=P.norm,
                edgecolors="none",
                **kwargs,
            )

        P.ax.set_xlim(0, 1)
        return P.ax, scatter

    def plot_color(
        self,
        weights: np.ndarray,
        window: list[float, float] | ureg.Quantity = None,
        ax: Axes = None,
        log: bool = False,
        shift: float | ureg.Quantity = None,
        patched: bool = True,
        bands: list[int] = None,
        **kwargs,
    ) -> (Axes, LineCollection):
        """
        Color gradient line-style for weights over a cumulative k-path.

        These weights can represent projections over orbitals or other similar attributes.
        A LineCollection will be plotted with color related to the weights input.

        Parameters
        ----------
        weights : np.ndarray, pint.Quantity
            Array of shape (nkpts, neigs).
        window : list[float,float], optional
            Minimal and maximum values for the colormap of the weights.
            Default is minimal of maximal values for the set of weights.
        ax : Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        log : bool, optional
            If True logaritmic scaling is applied. Default is False.
        shift : float | pint.Quantity, optional
            A constant shift applied to the eigenvalues (e.g., Fermi level).
            Default is zero.
        patched : bool, optional
            If True, attempts to patch discontinuities in the k-path.
            This prevents artificially connected lines in band structure
            plots (e.g., flat bands across high-symmetry points).
        bands : list of int, optional
            Indices of the bands to plot. If None, all bands are plotted.
        **kwargs : dict
            Additional matplotlib arguments passed to `LineCollection()`.

        Returns
        ----------
        ax : Axes
            The axes with the spectrum plot.
        line : matplotlib.collections.LineCollection
            The LineCollection for generating the colorbar.
        """
        import matplotlib.pyplot as plt

        P = self._pre_plot(
            ax=ax,
            shift=shift,
            bands=bands,
            patched=patched,
            weights=weights,
            log=log,
            window=window,
        )

        # Remove some labels from kwargs
        label = kwargs.pop("label", None)
        linewidth = kwargs.pop("linewidth", pdft.gradcolor_w)

        # Plotting band by band
        points = np.array(
            [P.x.magnitude, P.eigen.magnitude[:, P.band_indices[0]]]
        ).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(
            segments,
            norm=P.norm,
            label=label,
            **kwargs,
        )
        lc.set_array(P.weights[:, P.band_indices[0]])
        lc.set_linewidth(linewidth)
        line = P.ax.add_collection(lc)
        for i in P.band_indices[1:]:
            points = np.array(
                [P.x.magnitude, P.eigen.magnitude[:, P.band_indices[i]]]
            ).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments,
                norm=P.norm,
                **kwargs,
            )
            lc.set_array(P.weights[:, P.band_indices[i]])
            lc.set_linewidth(linewidth)
            P.ax.add_collection(lc)

        P.ax.autoscale_view()
        P.ax.set_xlim(0, 1)
        return P.ax, line


class ElectronBands(Spectrum):
    """
    Dressed `Spectrum` subclass for handling electronic bandstructures and spectrums.

    Attributes
    ----------
    filepath : str
        Path to the file containing electronic structure output.
    electron_num : int
        Total number of electrons in the system.
    fermi : float
        Fermi energy (0 if not found).
    """

    def __init__(self, file: str = None):
        """
        Initialize ElectronBands object.

        Parameters
        ----------
        file : str
            File from which to extract the bands.
        """
        from yaiv import grep

        if file is not None:
            self.filepath = file
            self.electron_num = grep.electron_num(self.filepath)
            try:
                self.fermi = grep.fermi(self.filepath)
            except (NameError, NotImplementedError):
                self.fermi = None
            try:
                lattice = grep.lattice(self.filepath)
            except NotImplementedError:
                lattice = None
            try:
                alat = grep.alat(self.filepath)
            except NotImplementedError:
                alat = None
            spec = grep.kpointsEnergies(self.filepath)
            Spectrum.__init__(
                self,
                eigenvalues=spec.energies,
                kpoints=spec.kpoints,
                weights=spec.weights,
                lattice=lattice,
                alat=alat,
            )
        else:
            self.electron_num = self.fermi = None
            Spectrum.__init__(self)


class PhononBands(Spectrum):
    """
    Dressed `Spectrum` subclass for handling phonon bandstructures and spectrums.

    Attributes
    ----------
    filepath : str
        Path to the file containing phonon frequencies output.
    """

    def __init__(self, file: str = None):
        """
        Initialize PhononBands object.

        Parameters
        ----------
        file : str
            Path to the file containing phonon frequencies output.
        """
        from yaiv import grep

        if file is not None:
            self.filepath = file
            try:
                lattice = grep.lattice(self.filepath)
            except NotImplementedError:
                lattice = None
            try:
                alat = grep.alat(self.filepath)
            except NotImplementedError:
                alat = None
            spec = grep.kpointsFrequencies(self.filepath)
            Spectrum.__init__(
                self,
                eigenvalues=spec.frequencies,
                kpoints=spec.kpoints,
                lattice=lattice,
                alat=alat,
            )
        else:
            Spectrum.__init__(self)


class Density:
    """
    General container for a scalar density defined on a 1D grid.

    This class can represent any scalar density ρ(v) sampled on a grid v.

    Attributes
    ----------
    density : np.ndarray | ureg.Quantity
        Array of shape (N,) with the density values. Units typically are
        something per unit of `grid` (e.g., states / energy).
    grid : np.ndarray | ureg.Quantity
        Array of shape (N,) with the grid values (e.g., energies).

    Methods
    -------
    from_data(...)
        Construct a Density by kernel-broadening samples located at `x`.
        Supports Gaussian or Methfessel–Paxton kernels and returns an instance
        with computed `.grid` and `.density`.
    integrate(...)
        - If `amount` is None: integrate density from grid[0] to `limit`.
        - If `amount` is provided: find the grid value where the integral equals `amount`.
    plot(...)
        Plot the density against the grid with optional fill and axis switching.
    """

    def __init__(
        self,
        grid: np.ndarray | ureg.Quantity = None,
        density: np.ndarray | ureg.Quantity = None,
    ):
        """
        Initialize a Density object.

        Parameters
        ----------
        grid : np.ndarray | pint.Quantity, optional
            Array of shape (N,) with the grid values.
        density : np.ndarray | pint.Quantity, optional
            Array of shape (N,) with the density values.
        """
        self.density = density
        self.grid = grid

    @classmethod
    def from_data(
        cls,
        x: np.ndarray | ureg.Quantity,
        values: np.ndarray | ureg.Quantity = None,
        weights: np.ndarray | None = None,
        center: float | ureg.Quantity | None = None,
        x_window: float | list[float] | ureg.Quantity | None = None,
        sigma: float | ureg.Quantity | None = None,
        steps: int | None = None,
        order: int = 0,
        cutoff_sigmas: float = defaults.cutoff_sigmas,
    ):
        """
        Initialize a kernel-broadened density on a grid from samples located at `x`.

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
            Default is 0, which recovers a Gaussian smearing.
        cutoff_sigmas : float, optional
            Truncate kernel support to [-cutoff_sigmas * sigma, +cutoff_sigmas * sigma]
            when summing contributions. Default yaiv.defaults.config.defaults.cutoff_sigmas.

        Notes
        -----
        It uses the utility `yaiv.utils.kernel_density_on_grid`.
        """
        data = ut.kernel_density_on_grid(
            x=x,
            values=values,
            weights=weights,
            center=center,
            x_window=x_window,
            sigma=sigma,
            steps=steps,
            order=order,
            cutoff_sigmas=cutoff_sigmas,
        )
        return cls(grid=data.grid, density=data.density)

    def integrate(
        self,
        limit: float | ureg.Quantity = None,
        amount: float = None,
        bound: int = 100,
    ) -> (float | ureg.Quantity, float | ureg.Quantity):
        """
        Integrate the density up to a given limit, or invert the integral.

        Two use cases:
        1) Plain integration:
           Returns the integral of the density from grid[0] to `limit`.
           If `limit` is None, integrates up to grid[-1].

        2) Inverse problem:
           If `amount` is provided, finds X* such that:
                ∫_{grid[0]}^{X*} density(v) dv = amount
           Returns (X*, error). This can fail if density has negative values
           or is not well-behaved.

        Parameters
        ----------
        limit : float | pint.Quantity, optional
            Upper bound for the integration. If None, uses grid[-1].
        amount : float, optional
            Target integral value (dimensionless number in magnitude space).
            If provided, the method returns the grid value X* where the integral
            equals `amount` (within the integration error tolerance).
        bound : int, optional
            Upper bound in number of subintervals used in the integrate.quad algorithm.

        Returns
        -------
        (value, error) : tuple
            - If `amount` is None: (integral, estimated_error).
            - If `amount` is provided: (X_star, estimated_error).
              Units are handled consistently with `grid`.

        Raises
        ------
        RuntimeError
            If the inverse integral does not converge to the requested amount
            within 100 iterations.

        Notes
        -----
        - Integration uses cubic interpolation and scipy.integrate.quad.
        - The inverse integration (when `amount` is provided) uses a bisection-like search
          and can fail if the density is not strictly non-negative or not well-behaved.
        """
        from scipy import interpolate, integrate

        if limit is None:
            limit = self.grid[-1]

        # Unit consistency
        quantities = [self.density, self.grid, limit]
        names = ["density", "grid", "limit"]
        ut._check_unit_consistency(quantities, names)

        # Extract magnitudes for numeric work; track units to restore at the end
        if isinstance(self.density, ureg.Quantity):
            grid_units = self.grid.units
            integral_units = self.density.units * grid_units
            X = self.grid.magnitude
            Y = (self.density / integral_units).to(1 / grid_units).magnitude
            X_max = limit.to(grid_units).magnitude
        else:
            X = np.asarray(self.grid)
            Y = np.asarray(self.density)
            X_max = float(limit)
            integral_units = grid_units = 1

        # Interpolant of the density
        f = interpolate.interp1d(X, Y, kind="cubic", fill_value=0.0, bounds_error=False)

        if amount is None:
            # Plain integration from X[0] to X_max
            integral, error = integrate.quad(f, X[0], X_max, limit=bound)
            # Units: (density units) * (grid units) -> dimensionless if DOS
            return integral * integral_units, error * integral_units
        else:
            # Inverse integral: find X* such that ∫ density = amount
            X_low = X[0]
            X_high = X[-1]
            max_iter = 100

            for _ in range(max_iter):
                X_mid = 0.5 * (X_low + X_high)
                integral, error = integrate.quad(f, X[0], X_mid, limit=bound)
                if abs(integral - amount) < error:
                    # Converged
                    return X_mid * grid_units, error * grid_units
                if integral > amount:
                    X_high = X_mid
                else:
                    X_low = X_mid
            raise RuntimeError(
                "Inverse integration did not converge within 100 iterations."
            )

    def plot(
        self,
        ax: Axes = None,
        shift: float | ureg.Quantity = None,
        switchXY: bool = False,
        fill: bool = True,
        alpha: float = pdft.alpha,
        **kwargs,
    ) -> Axes:
        """
        Plot the density against the grid.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, a new figure and axes are created.
        shift : float | pint.Quantity, optional
            A constant shift applied to the grid (e.g., Fermi level for DOS).
        switchXY : bool, optional
            If True, plot the density along the x-axis (horizontal plot).
        fill : bool, optional
            Whether to fill under the curve. Default True.
        alpha : float, optional
            Opacity of the fill between 0 and 1. Default from defaults or 0.4.
        **kwargs : dict
            Additional matplotlib arguments forwarded to `plot()`.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        # Handle units
        import matplotlib.pyplot as plt

        apply_plot_defaults()

        if self.density is None or self.grid is None:
            raise ValueError("Both `density` and `grid` must be set to plot.")
        quantities = [self.density, self.grid, shift]
        names = ["density", "grid", "shift"]
        ut._check_unit_consistency(quantities, names)

        if ax is None:
            fig, ax = plt.subplots()

        # Apply optional shift on the grid
        X = self.grid if shift is None else (self.grid - shift)
        Y = self.density

        # zorder defaults (allow override via kwargs)
        z_line = kwargs.pop("zorder", 2)
        z_fill = z_line - 1

        if switchXY:
            # density on x-axis, grid on y-axis
            (line,) = ax.plot(Y, X, zorder=z_line, **kwargs)
            if fill:
                ax.fill_betweenx(
                    X, 0 * Y, Y, alpha=alpha, color=line.get_color(), zorder=z_fill
                )
            ax.set_ylim(np.min(X), np.max(X))
            ax.set_xlim(left=np.min(Y))
            if isinstance(Y, ureg.Quantity):
                ax.set_xlabel(f"density ({Y.units})")
            else:
                ax.set_xlabel("density")
        else:
            # grid on x-axis, density on y-axis
            (line,) = ax.plot(X, Y, zorder=z_line, **kwargs)
            if fill:
                ax.fill_between(
                    X, Y, alpha=alpha, color=line.get_color(), zorder=z_fill
                )
            ax.set_xlim(np.min(X), np.max(X))
            ax.set_ylim(bottom=np.min(Y))
            if isinstance(Y, ureg.Quantity):
                ax.set_ylabel(f"density ({Y.units})")
            else:
                ax.set_ylabel("density")
        return ax
