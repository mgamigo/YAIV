"""
YAIV | yaiv.plot
================

This module provides plotting utilities for visualizing eigenvalue spectra from periodic
systems. It supports electronic and vibrational spectra obtained from common ab initio
codes such as Quantum ESPRESSO and VASP.

Functions in this module are designed to work seamlessly with spectrum-like objects
(e.g., `Spectrum`, `ElectronBands`, `PhononBands`) and accept units-aware data.

The visualizations are based on `matplotlib`, and include options for:

- Plotting band structures and phonon spectra
- Automatically shifting eigenvalues (e.g., Fermi level)
- Detecting and patching discontinuities in the k-path
- Annotating high-symmetry points from KPOINTS or bands.in

Functions
---------
get_HSP_ticks(kpath, k_lattice=None)
    Computes tick positions and LaTeX labels for high-symmetry points along a k-path.

kpath(ax, kpath, k_lattice=None)
    Plots vertical lines and labels at high-symmetry points in a matplotlib Axes.

bands(electronBands, ax=None, ...)
    Plots the electronic band structure for one or multiple systems.

phonons(phononBands, ax=None, ...)
    Plots the phonon band structure for one or multiple systems.

DOS(spectra, ax=None, ...)
    Plots the density of states (DOS) for a single or multiple eigenvalue spectra.

bandsDOS(electronBands, fig=None, axes=None, ...)
    Plots a band structure and its corresponding DOS side-by-side.

phononDOS(phononBands, fig=None, axes=None, ...)
    Plots a phonon band structure and its corresponding DOS side-by-side.

arrow3D(ax, vector, ...)
    Add a 3D arrow to a plot.

brillouinZone(lattice, ...)
    Plots a 3D Brillouin zone using lattice vectors.

Private Utilities
-----------------
_compare_spectra(spectra, ax, ...)
    Internal utility to overlay multiple spectra on the same Axes with legends and formatting.

_spectra_DOS(spectra, plot_func, ...)
    Internal helper to produce spectrum + DOS panels for electronic or vibrational bands.

_Arrow3D(FancyArrowPatch):
    A class to create 3D arrows in matplotlib plots.

_get_wigner_seitz(cell):
    Generates the Wigner-Seitz cell for a given lattice.

_axisEqual3D(ax):
    Adjusts the 3D plot axes to have equal scale, ensuring uniform aspect ratios.

Examples
--------
>>> from yaiv.spectrum import ElectronBands
>>> from yaiv import plot
>>> bands = ElectronBands("OUTCAR")
>>> plot.bands(bands)

See Also
--------
yaiv.spectrum : Base class for storing and plotting eigenvalue spectra
yaiv.grep     : Low-level data extractors used to populate spectrum objects
yaiv.defaults : Configuration and default plotting values
"""

from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch

from yaiv.defaults.config import ureg
from yaiv.defaults.config import plot_defaults as pdft
from yaiv.defaults.config import apply_plot_defaults
from yaiv import utils as ut
from yaiv import spectrum as spec


__all__ = [
    "get_HSP_ticks",
    "kpath",
    "bands",
    "phonons",
    "DOS",
    "bandsDOS",
    "phononDOS",
    "arrow3D",
    "brillouinZone",
]

apply_plot_defaults()


def get_HSP_ticks(
    kpath: SimpleNamespace | np.ndarray,
    k_lattice: np.ndarray = None,
    grid: list[int] = None,
) -> SimpleNamespace:
    """
    Compute tick positions and labels for high-symmetry points (HSPs) along a k-path.
    And optionally also the ticks for the grid points that lie in the path.

    Parameters
    ----------
    kpath : SimpleNamespace or np.ndarray
        A k-path object as given by yaiv.grep.kpath()
    k_lattice : np.ndarray, optional
        3x3 matrix of reciprocal lattice vectors in rows (optional).
        If provided, the high-symmetry points are converted from crystal to Cartesian coordinates.
    grid : list[int], optional
        Γ centred grid to show in the path.

    Returns
    -------
    ticks : SimpleNamespace
        Object with the following attributes:
        - ticks : np.ndarray
            Normalized cumulative distance for each high-symmetry point.
        - labels : list of str or None
            Corresponding labels for the ticks, or None if not available.
        - grid : np.ndarray
            Normalized cumulative distance for each grid point in the k-path.
    """
    if isinstance(kpath, SimpleNamespace):
        path_array = kpath.path
        label_list = kpath.labels
    else:
        path_array = kpath
        label_list = None
    if grid is not None:
        grid = ut.grid_generator(grid, periodic=True) * ureg("_2pi/crystal")
        grid = ut._expand_zone_border(grid)

    # Handle units
    quantities, names = [path_array, k_lattice], ["kpath", "k_lattice"]
    ut._check_unit_consistency(quantities, names)
    path_array, units = ut._split_units(path_array)

    segment_counts = [int(n) for n in path_array[:, -1]]
    hsp_coords = path_array[:, :3] * units

    # Convert to Cartesian coordinates if lattice is provided
    if k_lattice is not None:
        hsp_coords = ut.cryst2cartesian(hsp_coords, k_lattice).magnitude
        if grid is not None:
            grid = ut.cryst2cartesian(grid, k_lattice).magnitude
    else:
        hsp_coords = hsp_coords.magnitude

    # Ticks positions
    x_coord, grid_coord = [0.0], []
    for i, s in enumerate(segment_counts):
        if s != 1:
            length = np.linalg.norm(hsp_coords[i + 1] - hsp_coords[i])
            x_coord.append(x_coord[-1] + length)
            if grid is not None:
                for g in grid:
                    seg_distance = ut._point_to_segment_distance(
                        g, hsp_coords[i], hsp_coords[i + 1]
                    )
                    if np.around(seg_distance, decimals=3) == 0:
                        if (
                            np.around(np.linalg.norm(g - hsp_coords[i]), decimals=3)
                            == 0
                        ):
                            grid_coord.append(x_coord[-2])
                        elif (
                            np.around(
                                np.linalg.norm(g - [hsp_coords[i + 1]]), decimals=3
                            )
                            == 0
                        ):
                            grid_coord.append(x_coord[-1])
                        else:
                            lenght = np.linalg.norm(g - hsp_coords[i])
                            grid_coord.append(x_coord[-2] + lenght)
    x_coord = np.array(x_coord)
    # Normalize to [0, 1]
    grid_coord /= x_coord[-1]
    x_coord /= x_coord[-1]

    # Merge labels at discontinuities (where N=1)
    if label_list is not None:
        merged_labels = []
        for i, label in enumerate(label_list):
            label = label.strip()
            latex_label = r"$\Gamma$" if label.lower() == "gamma" else rf"${label}$"
            if i != 0 and segment_counts[i - 1] == 1:
                merged_labels[-1] = merged_labels[-1][:-1] + "|" + latex_label[1:]
            else:
                merged_labels.append(latex_label)
    else:
        merged_labels = None
    ticks = SimpleNamespace(ticks=x_coord, labels=merged_labels, grid=grid_coord)
    return ticks


def kpath(
    ax: Axes,
    kpath: SimpleNamespace | np.ndarray,
    k_lattice: np.ndarray = None,
    grid: list[int] = None,
):
    """
    Plots the high-symmetry points (HSPs) along a k-path in a given ax. And optionally,
    also the ticks for the grid points that lie in the path.

    Parameters
    ----------
    ax : Axes
        Axes to plot on. If None, a new figure and axes are created.
    kpath : SimpleNamespace or np.ndarray
        A k-path object as given by yaiv.grep.kpath()
    k_lattice : np.ndarray, optional
        3x3 matrix of reciprocal lattice vectors in rows (optional).
        If provided, the high-symmetry points are converted from crystal to Cartesian coordinates.
    grid : list[int], optional
        Γ centred grid to show in the path.
    """
    ticks = get_HSP_ticks(kpath, k_lattice, grid)
    for tick in ticks.ticks:
        ax.axvline(
            tick,
            color=pdft.vline_c,
            linewidth=pdft.vline_w,
            linestyle=pdft.vline_s,
        )
    for tick in ticks.grid:
        ax.axvline(
            tick,
            color=pdft.grid_c,
            linewidth=pdft.grid_w,
            linestyle=pdft.vline_s,
        )
    if ticks.labels is not None:
        ax.set_xticks(ticks.ticks, ticks.labels)
    else:
        ax.set_xticks(ticks.ticks)
    ax.xaxis.label.set_visible(False)


def _compare_spectra(
    spectra: list[spec.Spectrum],
    ax: Axes,
    patched: bool = True,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[list[int]] = None,
    **kwargs,
) -> Axes:
    """
    Plot and compare multiple spectra on a shared axes object.

    Parameters
    ----------
    spectra : list[spec.Spectrum]
    A list of spectrum objects to be plotted. Each spectrum must implement
        a `.plot()` method compatible with the plotting interface.
    ax : Axes
        Axes to plot on.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    colors : list[str], optional
        Colors to use when plotting multiple bands.
    labels : list[str], optional
        Labels to assign to each band in multi-plot case.
    grid : list[list[int]], optional
        Γ centred grids to show in the path.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """
    from itertools import cycle

    cycle_iter = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])

    if len(np.shape(grid)) == 2 and np.shape(grid)[0] == len(spectra):
        GRID = True
    else:
        GRID = False
    for i, S in enumerate(spectra):
        color = (
            colors[i] if colors is not None and i < len(colors) else next(cycle_iter)
        )

        label = (
            labels[i] if labels is not None and i < len(labels) else f"Spectrum {i+1}"
        )
        ax = S.plot(
            ax=ax,
            shift=getattr(S, "fermi", None),
            patched=patched,
            color=color,
            label=label,
            **kwargs,
        )
        if GRID:
            ticks = get_HSP_ticks(spectra[-1].kpath, spectra[-1].k_lattice, grid[i])
            for tick in ticks.grid:
                ax.axvline(
                    tick,
                    color=color,
                    linewidth=pdft.grid_w,
                    linestyle=pdft.vline_s,
                )
    ax.legend()
    return ax


def bands(
    electronBands: spec.ElectronBands | list[spec.ElectronBands],
    ax: Axes = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = [-1, 1] * ureg("eV"),
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot electronic band structures for one or multiple systems.

    Parameters
    ----------
    electronBands : ElectronBands or list of ElectronBands
        Band structure objects to plot.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : list[float] | float | ureg.Quantity, optional
        Energy window to be shown, default is [-1,1] eV.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """

    # Pop user-level styling
    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    if not isinstance(electronBands, list):
        band = electronBands
        indices = list(range(band.eigenvalues.shape[1]))
        # For non-spin orbit weights might add up to 2.
        valence_bands = round(band.electron_num / np.sum(band.weights))
        # plot valence bands
        ax = band.plot(
            ax=ax,
            shift=band.fermi,
            patched=patched,
            bands=indices[:valence_bands],
            color=user_color or pdft.valence_c,
            label=user_label,
            **kwargs,
        )
        # plot conduction bands
        ax = band.plot(
            ax=ax,
            shift=band.fermi,
            patched=patched,
            bands=indices[valence_bands:],
            color=user_color or pdft.conduction_c,
            **kwargs,
        )
    else:
        ax = _compare_spectra(electronBands, ax, patched, colors, labels, **kwargs)
        band = electronBands[0]

    if band.kpath is not None:
        kpath(ax, band.kpath, band.k_lattice)

    if band.fermi is not None:
        ax.axhline(y=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)

    # Handle units and setup window
    window = (
        window.to(band.eigenvalues.units).magnitude
        if isinstance(window, ureg.Quantity)
        else window
    )
    if isinstance(window, int) or isinstance(window, float):
        window = [-window, window]
    ax.set_ylim(window[0], window[1])

    plt.tight_layout()
    return ax


def phonons(
    phononBands: spec.PhononBands | list[spec.PhononBands],
    ax: Axes = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = None,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[int] | list[list[int]] = None,
    **kwargs,
) -> Axes:
    """
    Plot electronic band structures for one or multiple systems.

    Parameters
    ----------
    phononBands : spec.PhononBands | list[spec.PhononBands]
        Phonon band objects to plot.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : list[float] | float | ureg.Quantity, optional
        Frequency window to be shown, default is the whole spectra.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    grid : list[int] | list[list[int]], optional
        Γ centred grid (or grids) to show in the path.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """

    # Pop user-level styling
    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    if not isinstance(phononBands, list):
        band = phononBands
        ax = band.plot(
            ax,
            patched=patched,
            color=user_color or pdft.valence_c,
            label=user_label,
            **kwargs,
        )
    else:
        ax = _compare_spectra(
            phononBands, ax, patched, colors, labels, grid=grid, **kwargs
        )
        band = phononBands[0]

    if band.kpath is not None:
        if grid is None or len(np.shape(grid)) == 1:
            kpath(ax, band.kpath, band.k_lattice, grid)
        else:
            kpath(ax, band.kpath, band.k_lattice)

    # Handle units and setup window
    if window is not None:
        window = (
            window.to(band.eigenvalues.units).magnitude
            if isinstance(window, ureg.Quantity)
            else window
        )
        if isinstance(window, int) or isinstance(window, float):
            window = [-window, window]
        ax.set_ylim(window[0], window[1])

    ax.axhline(y=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)

    plt.tight_layout()
    return ax


def DOS(
    spectra: spec.ElectronBands | spec.PhononBands | spec.Spectrum,
    ax: Axes = None,
    window: float | list[float] | ureg.Quantity = None,
    smearing: float | ureg.Quantity = None,
    steps: int = None,
    order: int = 0,
    cutoff_sigmas: float = 3.0,
    switchXY: bool = False,
    fill: bool = True,
    alpha: float = pdft.alpha,
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> Axes:
    """
    Plot the density of states (DOS) for a single or list of spectra.

    Parameters
    ----------
    spectra : spec.ElectronBands | spec.PhononBands | spec.Spectrum
        Spectra from which to plot the DOS.
    ax : Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    window : float | list[float] | ureg.Quantity, optional
        Value window for the DOS. If float, interpreted as symmetric [-window, window].
        As a default and if a Fermi level is present it will compute the DOS in a [-1,1] eV window
        centered at E_f, otherwise it defaults to the whole eigenvalue range.
    smearing : float | ureg.Quantity, optional
        Gaussian smearing width in the same units as eigenvalues. Default is (window_size/200).
    steps : int, optional
        Number of grid points for DOS sampling. Default is 4 * (window_size/smearing).
    order : int, optional
        Order of the Methfessel-Paxton expansion. Default is 0, which recovers a Gaussian smearing.
    cutoff_sigmas : float, optional
        Number of smearing widths to use for truncation (e.g., 3 means ±3σ).
    switchXY : bool, optional
        Whether to plot the DOS along the x-axis (horizontal plot). Default is False.
    fill : bool, optional
        Whether to fill the area under the curve. Default is True.
    alpha : float, optional
        Opacity of the fill (0 = transparent, 1 = solid).
    colors : list[str], optional
        Colors to use when plotting multiple DOS.
    labels : list[str], optional
        Labels to assign to each DOS in multi-plot case.
    **kwargs : dict
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        Axes containing the plot.
    """
    from itertools import cycle

    # Extract first spectrum for defaults
    S = spectra[0] if isinstance(spectra, list) else spectra

    # Default window based on presence of Fermi level
    if window is None:
        window = [-1, 1] * ureg.eV if hasattr(S, "fermi") else None

    # Pop user-level styling
    user_color = kwargs.pop("color", None)  # user-defined color overrides all
    user_label = kwargs.pop("label", None)  # user-defined label

    if not isinstance(spectra, list):
        # Single plot
        S.get_DOS(
            center=getattr(S, "fermi", None),
            window=window,
            smearing=smearing,
            steps=steps,
            order=order,
            cutoff_sigmas=cutoff_sigmas,
        )
        ax = S.DOS.plot(
            ax,
            shift=getattr(S, "fermi", None),
            switchXY=switchXY,
            fill=fill,
            alpha=alpha,
            color=user_color or (pdft.DOS_c if hasattr(S, "fermi") else None),
            label=user_label,
            **kwargs,
        )
    else:
        # Multi plot
        cycle_iter = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        zorder = 2
        for i, S in enumerate(spectra):
            color = (
                colors[i]
                if colors is not None and i < len(colors)
                else next(cycle_iter)
            )
            label = (
                labels[i] if labels is not None and i < len(labels) else f"DOS {i+1}"
            )
            S.get_DOS(
                center=getattr(S, "fermi", None),
                window=window,
                smearing=smearing,
                steps=steps,
                order=order,
                cutoff_sigmas=cutoff_sigmas,
            )
            ax = S.DOS.plot(
                ax,
                shift=getattr(S, "fermi", None),
                switchXY=switchXY,
                fill=fill,
                alpha=alpha,
                color=color,
                label=label,
                zorder=zorder,
                **kwargs,
            )
            zorder += 2
        ax.legend()

    if hasattr(S, "fermi"):
        if switchXY == True:
            ax.axhline(y=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)
        else:
            ax.axvline(x=0, color=pdft.fermi_c, linewidth=pdft.fermi_w)
    # Labels
    if switchXY:
        if isinstance(S.DOS.density, ureg.Quantity):
            ax.set_xlabel(f"DOS ({S.DOS.density.units})")
        else:
            ax.set_xlabel("DOS")
    else:
        if isinstance(S.DOS.density, ureg.Quantity):
            ax.set_ylabel(f"DOS ({S.DOS.density.units})")
        else:
            ax.set_ylabel("DOS")

    plt.tight_layout()
    return ax


def _spectra_DOS(
    spectra: spec.ElectronBands | spec.PhononBands | list,
    plot_func: callable,
    fig: Figure = None,
    axes: list[Axes] = None,
    patched: bool = True,
    window: float | list[float] | ureg.Quantity = None,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[int] | list[list[int]] = None,
    **kwargs,
) -> tuple[Axes, Axes]:
    """
    Internal helper to plot a spectrum and its corresponding DOS.

    Parameters
    ----------
    spectra : spec.ElectronBands | spec.PhononBands | list,
        List of spectrum objects (e.g., ElectronBands or PhononBands).
    plot_func : callable
        Function to plot the band structure (e.g., `bands()` or `phonons()`).
    fig : Figure, optional
        Optional figure object to plot into. If not provided, a new one is created.
    axes : list of Axes, optional
        Optional list or tuple of two axes [ax_band, ax_DOS] to use instead of creating new ones.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : float | list[float] | Quantity, optional
        Energy/frequency window to show. Interpreted symmetrically if float.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    grid : list[int] | list[list[int]], optional
        Γ centred grid (or grids) to show in the path.
    **kwargs
        Additional keyword arguments passed to `plot_func()` and `DOS()`.

    Returns
    -------
    ax : Axes
        Axis with the band or phonon structure.
    ax_DOS : Axes
        Axis with the horizontal DOS plot.
    """
    if axes is not None:
        ax, ax_DOS = axes
        fig = ax.figure
    else:
        fig = fig or plt.figure()
        gs = fig.add_gridspec(
            1,
            2,
            hspace=0,
            wspace=0,
            width_ratios=[1 - pdft.bandsDOS_ratio, pdft.bandsDOS_ratio],
        )
        ax, ax_DOS = gs.subplots(sharex="col", sharey="row")

    user_color = kwargs.pop("color", None)
    user_label = kwargs.pop("label", None)

    if grid is None:
        plot_func(
            spectra,
            ax=ax,
            patched=patched,
            window=window,
            colors=colors,
            labels=labels,
            color=user_color,
            label=user_label,
            **kwargs,
        )
    else:
        plot_func(
            spectra,
            ax=ax,
            patched=patched,
            window=window,
            colors=colors,
            labels=labels,
            color=user_color,
            label=user_label,
            grid=grid,
            **kwargs,
        )

    DOS(
        spectra,
        ax=ax_DOS,
        switchXY=True,
        window=window,
        colors=colors,
        labels=labels,
        color=user_color,
        label=user_label,
        **kwargs,
    )

    # Clean up DOS axis
    ax_DOS.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    legend = ax_DOS.get_legend()
    if legend is not None:
        legend.remove()
    for name, spine in ax_DOS.spines.items():
        if name not in ["bottom", "left"]:
            spine.set_visible(False)
    ax_DOS.set_xlabel("DOS")
    plt.tight_layout()
    return ax, ax_DOS


def bandsDOS(
    electronBands: spec.ElectronBands | list[spec.ElectronBands],
    fig: Figure = None,
    axes: list[Axes] = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = [-1, 1] * ureg("eV"),
    colors: list[str] = None,
    labels: list[str] = None,
    **kwargs,
) -> tuple[Axes, Axes]:
    """
    Plot a band structure and its corresponding density of states (DOS) side-by-side.

    Parameters
    ----------
    electronBands : spec.ElectronBands | list[spec.ElectronBands]
        A spectrum or list of spectra representing electronic band structures.
    fig : Figure, optional
        Optional figure object to plot into. If not provided, a new figure is created.
    axes : list[Axes], optional
        Optional list or tuple of two axes [ax_band, ax_DOS] to use instead of creating new ones.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : float | list[float] | Quantity, optional
        Energy window to be shown, default is [-1,1] eV.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    **kwargs
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        The axis containing the band structure plot.
    ax_DOS : Axes
        The axis containing the DOS plot.
    """

    return _spectra_DOS(
        spectra=electronBands,
        plot_func=bands,
        fig=fig,
        axes=axes,
        patched=patched,
        window=window,
        colors=colors,
        labels=labels,
        **kwargs,
    )


def phononsDOS(
    phononBands: spec.PhononBands | list[spec.PhononBands],
    fig: Figure = None,
    axes: list[Axes] = None,
    patched: bool = True,
    window: list[float] | float | ureg.Quantity = None,
    colors: list[str] = None,
    labels: list[str] = None,
    grid: list[int] | list[list[int]] = None,
    **kwargs,
) -> tuple[Axes, Axes]:
    """
    Plot a phonon band structure and its corresponding density of states (DOS) side-by-side.

    Parameters
    ----------
    phononBands : spec.PhononBands | list[spec.PhononBands]
        A spectrum or list of spectra representing phonon band structures.
    fig : Figure, optional
        Optional figure object to plot into. If not provided, a new figure is created.
    axes : list[Axes], optional
        Optional list or tuple of two axes [ax_band, ax_DOS] to use instead of creating new ones.
    patched : bool, optional
        Whether to patch k-path discontinuities. Default is True.
    window : float | list[float] | Quantity, optional
        Energy window to be shown, default is [-1,1] eV.
    colors : list of str, optional
        Colors to use when plotting multiple bands.
    labels : list of str, optional
        Labels to assign to each band in multi-plot case.
    grid : list[int] | list[list[int]], optional
        Γ centred grid (or grids) to show in the path.
    **kwargs
        Additional keyword arguments passed to `plot()`.

    Returns
    -------
    ax : Axes
        The axis containing the band structure plot.
    ax_DOS : Axes
        The axis containing the DOS plot.
    """

    ax, ax_DOS = _spectra_DOS(
        spectra=phononBands,
        plot_func=phonons,
        fig=fig,
        axes=axes,
        patched=patched,
        window=window,
        colors=colors,
        labels=labels,
        grid=grid,
        **kwargs,
    )
    ax.autoscale(), ax.set_xlim([0, 1])

    plt.tight_layout()
    return ax, ax_DOS


class _Arrow3D(FancyArrowPatch):
    """
    A class to create 3D arrows in matplotlib plots.

    This class extends FancyArrowPatch to allow the creation of arrows in 3D
    space by using projections. It maintains the vertices in 3D and computes
    their 2D projection for rendering.

    Parameters
    ----------
    xs, ys, zs : array-like or ureg.Quantity
        Coordinates of the arrow's start and end points in 3D space.
    *args, **kwargs :
        Additional arguments and keyword arguments passed to FancyArrowPatch.

    Methods
    -------
    do_3d_projection(renderer=None)
        Projects 3D coordinates into 2D space and sets the arrow positions
        accordingly.
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """
        Project 3D coordinates onto the plot's 2D surface.

        Uses the current matplotlib axes' transform matrix to compute 2D
        positions from 3D coordinates, updating the arrow accordingly.

        Parameters
        ----------
        renderer : object, optional
            A rendering object that handles drawing operations. Uses the
            renderer associated with the current plot if None.

        Returns
        -------
        float
            The minimum z-coordinate of the projected vertices, used for depth
            ordering.
        """
        from mpl_toolkits.mplot3d import proj3d

        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def arrow3D(
    ax: Axes3D,
    vector: np.ndarray | ureg.Quantity,
    origin: np.ndarray | ureg.Quantity = None,
    **kwargs,
):
    """
    Add a 3D arrow to a plot.

    This function plots a 3D arrow on the given axis, originating from a
    specified point and extending toward a given vector. It handles unit
    consistency if provided through pint.

    Parameters
    ----------
    ax : Axes3D
        The 3D axis on which to draw the arrow.
    vector : np.ndarray or ureg.Quantity
        The direction and length of the arrow. If using ureg.Quantity, units are
        ensured to be consistent with the origin.
    origin : np.ndarray or ureg.Quantity, optional
        The start point for the arrow. Defaults to the origin of the 3D space.
    **kwargs :
        Additional styling parameters like 'color', 'mutation_scale', 'linewidth',
        and 'arrowstyle' for appearance customization.
    """
    # Handle units
    quantities = [vector, origin]
    names = ["vector", "origin"]
    ut._check_unit_consistency(quantities, names)
    if isinstance(vector, ureg.Quantity) and origin is not None:
        units = vector.units
        origin = origin.to(units)
    else:
        units = 1
    if origin is None:
        origin = np.zeros(3) * units
    else:
        vector = vector + origin

    # Pop user-level styling
    color = kwargs.pop("color", "black")
    mutation_scale = kwargs.pop("mutation_scale", 10)
    linewidth = kwargs.pop("linewidth", 1)
    arrowstyle = kwargs.pop("arrowstyle", "-|>")

    a = _Arrow3D(
        [origin[0], vector[0]],
        [origin[1], vector[1]],
        [origin[2], vector[2]],
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        arrowstyle=arrowstyle,
        color=color,
    )
    ax.add_artist(a)


def _get_wigner_seitz(cell: np.ndarray | ureg.Quantity) -> tuple:
    """
    Generates the Wigner-Seitz cell for a given lattice.

    This function constructs the Wigner-Seitz cell of the lattice specified by
    the input matrix. The Wigner-Seitz cell is formed using a Voronoi decomposition
    of points in space that define the lattice. For a reciprocal lattice, this cell
    is equivalent to the Brillouin zone.

    Parameters
    ----------
    cell : np.ndarray | ureg.Quantity
        A 3x3 matrix representing the lattice vectors. Each row corresponds to a
        lattice vector in 3D space.

    Returns
    -------
    tuple
        A tuple containing:
        - vertices : np.ndarray
          The vertices of the Wigner-Seitz cell.
        - ridges : list of np.ndarray
          The ridge lines of the Wigner-Seitz cell.
        - facets : list of np.ndarray
          The facet faces of the Wigner-Seitz cell.

    Notes
    -----
    - The function uses a Voronoi decomposition, which divides space into regions
      closest to each point in a given set. The function specifically identifies
      facets and ridges that are relevant for the origin-centered Voronoi cell of the
      provided lattice.
    - The point [0, 0, 0] is considered the center of the Wigner-Seitz cell in the
      Voronoi diagram, and computation relies on identifying boundaries formed
      orthogonally to lattice vectors.
    - The number 13 corresponds to the central point in the generated grid that
      represents [0, 0, 0] after indexing mgrid.
    """
    cell = np.asarray(cell, dtype=float)

    if cell.shape != (3, 3):
        raise ValueError("Input cell must be a 3x3 matrix.")
    from scipy.spatial import Voronoi

    # Generate grid points within a 3x3x3 cube centered at the origin.
    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    # Compute the Voronoi diagram for the set of points.
    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        if pid[0] == 13 or pid[1] == 13:
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    # Deduplicate vertex indices.
    bz_vertices = list(set(bz_vertices))
    return vor.vertices[bz_vertices], bz_ridges, bz_facets


def _axisEqual3D(ax):
    """
    Adjusts the 3D plot axes to have equal scale, ensuring uniform aspect ratios.

    This function modifies the axis limits to make the scales equal across all
    three dimensions in a 3D plot. This adjustment aims to maintain true
    proportional representation of the data and geometric relationships within
    the plot space.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.Axes3DSubplot
        The 3D subplot axis object from Matplotlib to be adjusted. The axis should
        have preconfigured limits that may need equalization.
    """
    # Calculate current extents for x, y, z dimensions
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]  # Size of each dimension
    centers = np.mean(extents, axis=1)  # Center points of each dimension
    maxsize = max(abs(sz))  # Determine max size for uniform scaling
    r = maxsize / 2  # Half of max size for symmetric limits
    # Apply symmetric limits centered at computed centers
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def brillouinZone(
    lattice: np.ndarray | ureg.Quantity,
    axis: Axes3D = None,
    basis: bool = True,
    sides: bool = True,
    facecolors: str = "cyan",
    **kwargs,
) -> Axes3D:
    """
    Plots a 3D Brillouin zone using lattice vectors.

    This function visualizes the Brillouin zone based on given lattice vectors,
    which may be specified either in real space or reciprocal space units. It
    supports rendering on a specified matplotlib 3D axis, with customizable
    styling for facets and basis vectors.

    Parameters
    ----------
    lattice : ureg.Quantity or np.ndarray (shape=(3,3))
        Lattice vectors used as input. If provided as a ureg.Quantity, units
        must be consistent with spatial or reciprocal measures.
    axis : matplotlib.axes._subplots.Axes3DSubplot, optional
        The axis on which the plot will be rendered. If None, a new figure and
        axis are created.
    basis : bool, optional
        If True, plots the basis vectors of the lattice.
    sides : bool, optional
        If True, plots the facet sides of the Brillouin Zone.
    facecolors : str or tuple, optional
        Color specification for the sides of the 3D facets. Defaults to "cyan".
    **kwargs :
        Additional keyword styling arguments passed to ridge plots,
        such as 'color' and 'linewidth'.

    Raises
    ------
    ValueError
        Raised if the units of the lattice are incompatible with plotting the
        Brillouin zone.

    Returns
    -------
    Axes3D
        The axis object containing the Brillouin zone plot.

    Notes
    -----
    - Handles unit conversion if the input is a ureg.Quantity, ensuring that
      lattice vectors are appropriately transformed in reciprocal space.
    - Uses a default unit of "au" if lattice is provided as np.ndarray.
    - Incorporates a utility to normalize axis scale after plotting.
    """
    if isinstance(lattice, ureg.Quantity):
        if lattice.dimensionality in [
            ureg.m.dimensionality,
            ureg.crystal.dimensionality,
            ureg.alat.dimensionality,
        ]:
            lattice = ut.reciprocal_basis(lattice)
            units = lattice.units
            K_vec = lattice.magnitude
        elif lattice.dimensionality in [
            (1 / ureg.m).dimensionality,
            (1 / ureg.crystal).dimensionality,
            (1 / ureg.alat).dimensionality,
        ]:
            units = lattice.units
            K_vec = lattice.magnitude
        else:
            raise ValueError(
                f"Invalid units ({lattice.units}) for plotting Brillouin zone"
            )
    else:
        units = "au"
        K_vec = np.asarray(lattice)

    if axis is None:
        plt.figure()
        ax = plt.axes(projection="3d")
    else:
        ax = axis

    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # Style settings from user input
    color = kwargs.pop("color", "black")
    linewidth = kwargs.pop("linewidth", 1.2)

    v, r, f = _get_wigner_seitz(K_vec)
    for xx in r:
        ax.plot(
            xx[:, 0], xx[:, 1], xx[:, 2], color=color, linewidth=linewidth, **kwargs
        )
    if sides:
        ax.add_collection3d(
            Poly3DCollection(r, facecolors=facecolors, linewidths=0, alpha=0.05)
        )

    if basis:
        arrow3D(ax, K_vec[0], color="red", linewidth=1.5, mutation_scale=12)
        arrow3D(ax, K_vec[1], color="green", linewidth=1.5, mutation_scale=12)
        arrow3D(ax, K_vec[2], color="blue", linewidth=1.5, mutation_scale=12)

    ax.set_xlabel(f"$k_x$ ({units})")
    ax.set_ylabel(f"$k_y$ ({units})")
    ax.set_zlabel(f"$k_z$ ({units})")
    if axis is None:
        plt.show()
    _axisEqual3D(ax)
    plt.tight_layout()
    return ax
