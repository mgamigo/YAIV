"""
YAIV | yaiv.grep
================

This module provides text-scraping utilities for extracting (grepping) structural and spectral
information from first-principles calculation outputs. It supports common DFT packages
such as Quantum ESPRESSO and VASP.

The functions in this module perform low-level parsing (i.e., grepping) of data such as:

- Electronic eigenvalues and k-points
- Phonon frequencies and paths
- Lattice vectors and stress tensors
- Number of electrons and total energies
- Fermi level and reciprocal space paths
- Computational cost (runtime, RAM)
- DFT parameters as cutoff, smearing or K-grid

Supported formats include:
- Quantum ESPRESSO output/input: `pw.x`, `ph.x`, `bands.in`, `projwfc.x`, `matdyn.x`, `.xml`
- VASP output: `OUTCAR`, `EIGENVAL`, `KPOINTS`, `PROCAR`

Functions
---------
electron_num(file)
    Greps the number of electrons from a QE or VASP output file.

alat(file)
    Extract the lattice parameter (alat) from an input file.

lattice(file)
    Extracts the lattice vectors from outputs. Optionally in internal units (alat).

fermi(file)
    Extracts the Fermi energy from an output file.

total_energy(file, decomposition=False)
    Extracts the total free energy, optionally returning its internal decomposition (QE).

stress_tensor(file)
    Extracts the total stress tensor from a QE or VASP output file.

kpath(file, labels=True)
    Extracts the k-point path in reciprocal space, optionally with high-symmetry labels.

kpointsEnergies(file)
    Greps the k-points, energies, kpoint-weights  and orbital projections (if available) for different file kinds.

kpointsFrequencies(file)
    Extracts k-points and phonon frequencies from Quantum ESPRESSO phonon outputs.

dyn_file(file)
    Reads a QE `.dyn` file and returns vibrational data (q-point (2π/Å), lattice (Å), frequencies, ...)

dyn_q(q_cryst, results_ph_path, qe_format=True)
    Locates and reads `.dyn*` file for a given q-point, returning the full dynamical matrix (3Nx3N).

symmetries(file)
    Grep symmetry operations and return them as rotation/translation pairs.

cutoff(file)
    Greps the cutoff energy from a variety of filetypes.

smearing(file)
    Greps the smearing used in calculation from a variety of filetypes.

runtime(file)
    Greps the computational runtime from a variety of filetypes.

ram(file)
    Greps the RAM needed in the computation from a variety of filetypes.

k_grid(file)
    Greps the k-grid from a variety of filetypes.

atomic_forces(file)
    Greps the atomic forces from a variety of filetypes.

Private Utilities
-----------------
_filetype(file)
    Heuristically determines the type of simulation output file (QE, VASP, etc.).

_OrbitalProjectionContainer:
    Container for orbital-resolved projection matrices.

_Symmetry:
    Container for symmetry operations.

_Qe_xml:
    Minimal reader for Quantum ESPRESSO XML output files.

_find_dyn_file(q_cryst, results_ph_path)
    Internal helper that searches `.dyn*` files matching the specified q-point.

Examples
--------
>>> from yaiv grep import grep
>>> spectrum = grep.kpointsEnergies("OUTCAR")
>>> spectrum.eigenvalues.shape
(100, 32)

>>> from yaiv grep import grep
>>> a = grep.lattice("qe.out")
>>> a.shape
(3, 3)

See Also
--------
yaiv.spectrum : Data container and plotter for eigenvalue spectra
yaiv.utils    : Basis universal utilities
"""

import re
import warnings
from types import SimpleNamespace

import numpy as np

from yaiv.defaults.config import ureg
from yaiv import utils as ut

__all__ = [
    "electron_num",
    "alat",
    "lattice",
    "fermi",
    "total_energy",
    "stress_tensor",
    "kpath",
    "kpointsEnergies",
    "kpointsFrequencies",
    "dyn_file",
    "dyn_q",
    "symmetries",
    "cutoff",
    "smearing",
    "k_grid",
    "atomic_forces",
    "runtime",
    "ram",
]


def _filetype(file: str) -> str:
    """
    Detects the filetype of the provided file.

    Parameters
    ----------
    file : str
        Filepath for the file to analyze.

    Returns
    -------
    filetype : str
        Detected filetype (None if not filetype is detected).
    """
    filetype = None
    with open(file, "r") as lines:
        for line in lines:
            line = line.strip().lower()
            if re.search(r"calculation.*scf|calculation.*nscf", line):
                filetype = "qe_scf_in"
                break
            elif "program pwscf" in line:
                filetype = "qe_scf_out"
                break
            elif "program phonon" in line:
                filetype = "qe_ph_out"
                break
            elif "calculation" in line and "bands" in line:
                filetype = "qe_bands_in"
                break
            elif "flfrc" in line:
                filetype = "matdyn_in"
                break
            elif "dynamical matrix" in line:
                filetype = "qe_dyn"
                break
            elif "&plot nbnd=" in line:
                filetype = "qe_freq_out"
                break
            elif "projwave" in line:
                filetype = "qe_proj_out"
                break
            elif "procar" in line:
                filetype = "procar"
                break
            elif "vasp" in line:
                filetype = "outcar"
                break
            elif len(line.split()) == 4 and all(x.isdigit() for x in line.split()):
                filetype = "eigenval"
                break
            elif "line-mode" in line:
                filetype = "kpath"
                break
            elif ("direct" in line and "directory" not in line) or "cartesian" in line:
                filetype = "poscar"
                break
            elif "espresso xml" in line or "espresso xsi" in line:
                filetype = "qe_xml"
                break
    return filetype


class _OrbitalProjectionContainer:
    """
    Container for orbital-resolved projection matrices.

    Supports projections stored from:
      - POSCAR-style (VASP-like): keys (ion, l, ml, M)
      - QE (Quantum ESPRESSO) SOC projwfc: keys (ion, l, j, mj, wfc)
      - QE (Quantum ESPRESSO) scalar projwfc: keys (ion, l, m, wfc)

    Each entry maps a quantum-number tuple → projection matrix of shape (nkpts, nbnds).

    Attributes
    ----------
    _data : dict
        Internal mapping from quantum-number tuples to projection matrices.
        - POSCAR format: (ion, l, m, M) → np.ndarray (nkpts, nbnds)
        - QE SOC format:     (ion, l, j, mj, wfc) → np.ndarray (nkpts, nbnds)
        - QE scalar format:  (ion, l, m, wfc) → np.ndarray (nkpts, nbnds)
    _code : str
        Code according to which the code is formated. Options are "quantum_espresso" and "vasp".

    Methods
    -------
    add_poscar(ion, l, m, M, matrix)
        Store a projection matrix in POSCAR format.
    add_qe(ion, l, wfc, matrix, m, j, mj)
        Store a projection matrix in QE projwfc format.
    __call__(...)
        Sum over matching entries with flexible indexing across both formats.
    __repr__()
        Compact summary of stored channels (POSCAR or QE).
    """

    def __init__(self, code=str):
        """
        Initialize an empty orbital projection container.

        Parameters
        ----------
        code : str
            Code according to which the code is formated. Options are "quantum_espresso" and "vasp".
        """
        self._data = {}
        self._code = code

    def add_poscar(self, ion: int, l: int, m: int, M: int, matrix: np.ndarray):
        """
        Store a projection matrix for a specific channel in POSCAR format.

        Parameters
        ----------
        ion : int
            Ion index the projection refers to.
        l : int
            Orbital angular momentum (0=s, 1=p, 2=d, ...).
        m : int
            Magnetic quantum number (−l..+l).
        M : int
            Magnetization channel: 0 for total (|M|), 1,2,3 for x,y,z components.
        matrix : np.ndarray, shape (nkpts, nbnds)
            Projection matrix.
        """
        self._data[(ion, l, m, M)] = matrix

    def add_qe(
        self,
        ion: int,
        l: int,
        wfc: int,
        matrix: np.ndarray,
        m: int = None,
        j: float = None,
        mj: float = None,
    ):
        """
        Store a projection matrix for a specific QE projwfc channel.

        Parameters
        ----------
        ion : int
            Ion index the projection refers to.
        l : int
            Orbital angular momentum (0=s, 1=p, 2=d, ...).
        wfc : int
            Wavefunction index (QE projwfc channel identifier).
        matrix : np.ndarray, shape (nkpts, nbnds)
            Projection matrix.
        m : int, optional
            QE real-harmonic component index (1..2*l+1). Not a magnetic quantum number.
            Used in scalar mode.
        j : float, optional
            Total angular momentum (e.g., 0.5, 1.5, ...). Used in SOC mode.
        mj : float, optional
            Projection of total angular momentum (−j..+j, step 1). Used in SOC mode.

        Raises
        ------
        ValueError
            If an invalid combination of (m, j, mj) is provided or values are out of range.
        """
        scalar_mode = (m is not None) and (j is None) and (mj is None)
        soc_mode = (m is None) and (j is not None) and (mj is not None)

        if scalar_mode:
            self._data[(ion, l, m, wfc)] = matrix
        elif soc_mode:
            self._data[(ion, l, j, mj, wfc)] = matrix
        else:
            raise ValueError(
                "Invalid QE projection description. "
                "Provide either (m) for scalar calculations "
                "or (j, mj) for spin–orbit calculations."
            )

    def __repr__(self) -> str:
        keys = np.array(tuple(self._data.keys()), dtype=object)
        if self._code == "quantum_espresso":
            ions = int(max(int(k[0]) for k in keys) + 1)
            l_by_ion = {
                ion: sorted({int(k[1]) for k in keys if int(k[0]) == ion})
                for ion in range(ions)
            }
            # QE_SOC: keys of length 5 -> (ion, l, j, mj, wfc)
            if len(keys[0]) == 5:
                return (
                    f"_OrbitalProjectionContainer("
                    f"ions={ions}, \n"
                    f"l={l_by_ion} \n"
                    f"shape=(ion, l, j, m_j, wfc))"
                )
            # QE_scalar: keys of length 5 -> (ion, l, m, wfc)
            elif len(keys[0]) == 4:
                return (
                    f"_OrbitalProjectionContainer("
                    f"ions={ions}, \n"
                    f"l={l_by_ion} \n"
                    f"shape=(ion, l, m[1,1+2l], wfc)"
                )
        # POSCAR: keys of length 4 -> (ion, l, m, M)
        elif self._code == "vasp":
            keys = np.array(keys, dtype=int)
            ions = np.max(keys[:, 0]) + 1
            l = sorted(set(int(x) for x in keys[:, 1]))
            Mmax = int(np.max(keys[:, 3]))
            Mstr = f"[0-{Mmax}]" if Mmax > 0 else "0"
            return (
                f"_OrbitalProjectionContainer(ions={ions}, l={l}, m_l=[-l,l], M={Mstr}\n"
                f"shape=(ion, l, m_l, M))"
            )
        else:
            return f"_OrbitalProjectionContainer(keys_shape={len(keys[0])}, entries={len(keys)})"

    def __call__(
        self,
        ion: int | range | tuple | list | slice = slice(None),
        l: int | range | tuple | list | slice = slice(None),
        m: int | range | tuple | list | slice = slice(None),
        M: int | range | tuple | list | slice = 0,
        j: float | tuple | list | slice = slice(None),
        mj: float | tuple | list | slice = slice(None),
        wfc: int | range | tuple | list | slice = slice(None),
    ) -> np.ndarray:
        """
        Sum all projection matrices matching the query.

        This method supports both POSCAR-style and QE-style keys transparently.
        Provide the subset of quantum numbers you want to match; any argument can be
        a scalar, list/tuple/range, or slice(None) to select all.

        Parameters
        ----------
        ion : int | list | range | slice, optional
            Ion index or indices. Default selects all ions.
        l : int | list | range | slice, optional
            Orbital angular momentum(s). Default selects all.
        m : int | list | range | slice, optional
            Magnetic quantum number(s) (−l..+l), for POSCAR data.
            Real-harmonic component index [1,1+2l], for scalar QE.
        M : int | list | range | slice, optional (POSCAR)
            Magnetization channel (0 total, 1,2,3 for x,y,z). Default 0.
        j : float | list | tuple | slice, optional (QE)
            Total angular momentum(s). Only used for QE data.
        mj : float | list | tuple | slice, optional (QE)
            m_j values (projection of J). Only used for QE data.
        wfc : int | list | range | slice, optional (QE)
            Wavefunction channel indices. Only used for QE data.

        Returns
        -------
        np.ndarray
            Sum of all projection matrices matching the query.

        Raises
        ------
        KeyError
            If no entries match the given query.

        Examples
        --------
        # POSCAR-style:
        proj(ion=0, l=1, m=0, M=1)               # single matrix
        proj(ion=[0,1,2], l=1, m=0, M=1)         # sum over several ions
        proj(ion=slice(None), l=1, m=0, M=1)     # sum over all ions

        # QE-style:
        proj(ion=0, l=1, j=1.5, mj=0.5, wfc=3)   # single matrix
        proj(ion=0, l=1, j=slice(None))          # sum over all J
        proj(ion=[0,1], l=[0,1], j=[0.5,1.5])    # sum over subsets
        """
        keys = np.array(tuple(self._data.keys()), dtype=object)
        # QE keys length 5
        if self._code == "quantum_espresso":
            if len(keys[0]) == 5:
                query = (ion, l, j, mj, wfc)
            elif len(keys[0]) == 4:
                query = (ion, l, m, wfc)
        elif self._code == "vasp":
            # POSCAR keys length 4
            query = (ion, l, m, M)
        else:
            raise KeyError("Unknown code in container")

        def match(val, key):
            if isinstance(val, slice):
                return True
            elif isinstance(val, (list, tuple, set, range)):
                return key in val
            else:
                return key == val

        # Direct (single) match returns the stored matrix
        if all(not isinstance(q, (list, tuple, set, range, slice)) for q in query):
            return self._data[query]

        # Sum all matching matrices
        selected = [
            mat
            for key, mat in self._data.items()
            if all(match(q, k) for q, k in zip(query, key))
        ]
        if not selected:
            raise KeyError(f"No projection matches query: {query}")
        return sum(selected)


class _Symmetry:
    def __init__(
        self,
        R: np.ndarray,
        t: np.ndarray = None,
        units: ureg.Quantity = ureg.dimensionless,
    ):
        """
        Container for symmetry operations with rotation and (optional) translation in a given basis.

        Parameters
        ----------
        R : np.ndarray
            Rotation matrix in the current coordinate system (matches `units`).
        t : np.ndarray, optional
            Translation (shift) vector in the current coordinate system. Default is zero.
        units : ureg.Quantity, optional.
            Units of `t` (and the basis in which `R` is expressed). Typically
            crystal (e.g., ureg('crystal')) or Cartesian length (e.g., ureg.meter).

        Methods
        -------
        to_cartesian(lattice)
            Convert this symmetry from crystal to Cartesian coordinates.
        to_crystal(lattice)
            Convert this symmetry from Cartesian to crystal coordinates.

        Notes
        -----
        - This container does not assume or enforce a specific convention
          (row/column) beyond storing R and t consistently in the same basis.
        - Use `to_cartesian` / `to_crystal` to convert both R and t between bases.
        """

        self.R = R
        if t is None:
            t = np.zeros(R.shape[0])
        self.t = t
        self.units = units

    def __repr__(self) -> str:
        return f"_Symmetry \n {self.R}, {self.t}, {self.units}"

    # ------------- change of basis ---------
    def to_cartesian(self, lattice: np.ndarray | ureg.Quantity) -> "_Symmetry":
        """
        Convert _Symmetry object to Cartesian coordinates.

        Parameters
        ----------
        lattice : np.ndarray | ureg.Quantity
            Direct lattice vectors (rows) in Cartesian units (e.g., Å). If
            given as a Quantity, units are used; otherwise assumed Cartesian length

        Returns
        -------
        Sym_new : _Symmetry
            New _Symmetry object in Cartesian coord.

        Raises
        ------
        ValueError
            If the current `units` are not crystal units.
        """
        if self.units.dimensionality != ureg.crystal.dimensionality:
            raise ValueError(
                f"Only can chage to Cartesian coord from crystal units. Not {self.units} units."
            )
        t = ut.cryst2cartesian(self.t * self.units, lattice)
        lattice, _ = ut._split_units(lattice)
        cryst2cartesian = lattice.T
        cartesian2cryst = ut.invQ(cryst2cartesian)
        R = cryst2cartesian @ self.R @ cartesian2cryst
        return _Symmetry(R, t.magnitude, t.units)

    def to_crystal(self, lattice: np.ndarray | ureg.Quantity) -> "_Symmetry":
        """
        Convert _Symmetry object to crystal coordinates.

        Parameters
        ----------
        lattice : np.ndarray | ureg.Quantity
            Direct lattice vectors (rows) in Cartesian units (e.g., Å). If
            given as a Quantity, units are used; otherwise assumed Cartesian length

        Returns
        -------
        Sym_new : _Symmetry
            New _Symmetry object in crystal coord.

        Raises
        ------
        ValueError
            If the current `units` are not in lenght dimnesion.
        """
        if self.units.dimensionality != ureg.meter.dimensionality:
            raise ValueError(
                f"Only can chage to crystal coord from length units. Not {self.units} units."
            )
        t = ut.cartesian2cryst(self.t * self.units, lattice)
        lattice, _ = ut._split_units(lattice)
        cryst2cartesian = lattice.T
        cartesian2cryst = ut.invQ(cryst2cartesian)
        R = cartesian2cryst @ self.R @ cryst2cartesian
        return _Symmetry(R, t.magnitude, t.units)


class _Qe_xml:
    """
    Minimal reader for Quantum ESPRESSO XML output files.

    Provides utilities to extract common physical quantities.

    Notes
    -----
    - Units: numerical values in the XML are in Hartree atomic units unless
      otherwise specified. Returned values are wrapped in `ureg.Quantity`.
    """

    def __init__(self, file):
        """
        Initialize a QE XML reader.

        Parameters
        ----------
        file : str or Path
            Path to the Quantum ESPRESSO XML file.

        Raises
        ------
        NotImplementedError
            If the file type is not recognized as a QE XML file.
        """
        import xml.etree.ElementTree as ET

        if _filetype(file) == "qe_xml":
            tree = ET.parse(file)
            self.root = tree.getroot()
        else:
            raise NotImplementedError("Unsupported filetype")

    def electron_num(self) -> int:
        """
        Greps the number of electrons.

        Returns
        -------
        num_elec : int
            Number of electrons.
        """
        elec = self.root.find(".//nelec")
        return int(float(elec.text))

    def alat(self) -> ureg.Quantity:
        """
        Extract the lattice parameter (alat) from an input file.

        Returns
        -------
        alat : ureg.Quantity
            Lattice parameter `alat` with units.
        """
        alat = float(self.root.find(".//atomic_structure").attrib["alat"]) * ureg(
            "bohr/alat"
        )
        return alat

    def lattice(self) -> np.ndarray:
        """
        Greps the lattice vectors.

        Returns
        -------
        lattice : np.ndarray
            3x3 array of lattice vectors with attached units (ureg.Quantity).
        """
        cell = self.root.find(".//cell")
        lattice = []
        for line in cell:
            v = [float(x) for x in line.text.split()]
            lattice += [v]
        lattice = np.array(lattice) * ureg.bohr
        return lattice

    def fermi(self) -> float:
        """
        Greps the Fermi energy.

        Returns
        -------
        E_f : float
            Fermi energy with attached units (ureg.Quantity).
        """
        fermi = self.root.find(".//fermi_energy")
        return float(fermi.text) * ureg.hartree

    def total_energy(self, decomposition: bool = False) -> float | SimpleNamespace:
        """
        Greps the total free energy or it's decomposition.

        Parameters
        ----------
        decomposition : bool, optional
            If True an energy decomposition is returned instead. Default is False.

        Returns
        -------
        energy : float | SimpleNamespace
            If decomposition is False a single float with the free energy is returned.
            If decomposition is True a namespace with the following attributes is returned:
                -  F            -> Total Free energy
                - -TS           -> Smearing contribution
                -  U (= F+TS)   -> Internal energy
                    -  U_one_electron
                    -  U_hartree
                    -  U_exchange-correlational
                    -  U_ewald
        """
        lines = self.root.find(".//total_energy")
        etot = float(lines.find(".//etot").text) * ureg.hartree
        eband = float(lines.find(".//eband").text) * ureg.hartree
        ehart = float(lines.find(".//ehart").text) * ureg.hartree
        vtxc = float(lines.find(".//vtxc").text) * ureg.hartree
        etxc = float(lines.find(".//etxc").text) * ureg.hartree
        ewald = float(lines.find(".//ewald").text) * ureg.hartree
        demet = float(lines.find(".//demet").text) * ureg.hartree
        energy = SimpleNamespace(
            F=etot,
            TS=demet,
            U=etot - demet,
            U_one_electron=etot - demet - ehart - etxc - ewald,
            U_hartree=ehart,
            U_xc=etxc,
            U_ewald=ewald,
        )
        if decomposition:
            return energy
        else:
            return energy.F

    def kpointsEnergies(self) -> SimpleNamespace:
        """
        Grep the kpoints, energies and kpoint-weights.

        Returns
        -------
        SimpleNamespace : SimpleNamespace
            SimpleNamespace class with the following attributes:
            - energies : np.ndarray
                List of energies, each row corresponds to a particular k-point.
            - kpoints : np.ndarray
                List of k-points.
            - weights : np.ndarray
                List of kpoint-weights.
        """
        KPOINTS, WEIGHTS, ENERGIES = [], [], []
        ks_energies = self.root.findall(".//ks_energies")
        for elem in ks_energies:
            # Get kpoint and weights
            kpoint = elem.find(".//k_point")
            w = float(kpoint.attrib["weight"])
            k = [float(x) for x in kpoint.text.split()]
            KPOINTS += [k]
            WEIGHTS += [w]
            # Get energies
            E = [float(x) for x in elem.find(".//eigenvalues").text.split()]
            ENERGIES += [E]
        return SimpleNamespace(
            energies=ENERGIES * ureg.hartree,
            kpoints=KPOINTS * ureg("_2pi/alat") / self.alat(),
            weights=np.array(WEIGHTS),
        )

    def symmetries(self) -> list[SimpleNamespace]:
        """Grep symmetry operations from the QE XML and return them as rotation/translation pairs.

        This reads all <symmetry> elements, extracts the 3×3 rotation matrix from
        <rotation> and the fractional translation from <fractional_translation>,
        and returns a list of objects with fields:
          - R: np.ndarray shape (3, 3)
          - t: lenght-3 translation vector
          - units: ureg.Quantity with the units in which {R|t} is given.

        Returns
        -------
        symmetries : list[SimpleNamespace]
            Each entry contains R (3×3 rotation matrix) and t (translation
            vector) and units.
        """
        OUT = []
        symmetries = self.root.findall(".//symmetry")
        for elem in symmetries:
            rotation = elem.find(".//rotation")
            R = np.fromstring(rotation.text, sep=" ").reshape(3, 3)
            translation = elem.find(".//fractional_translation")
            if translation is not None:
                t = np.array([float(x) for x in translation.text.split()])
            else:
                t = np.zeros(3)
            OUT.append(_Symmetry(R=R, t=t, units=ureg.crystal))
        return OUT

    def cutoff(self) -> ureg.Quantity:
        """
        Greps the cutoff energy.

        Returns
        -------
        cutoff : ureg.Quantity
            Cutoff energy with attached units (ureg.Quantity).
        """
        cutoff = float(self.root.find(".//ecutwfc").text) * ureg.Ry
        return cutoff

    def smearing(self) -> ureg.Quantity:
        """
        Greps the smearing.

        Returns
        -------
        smearing : ureg.Quantity
            Smearing with attached units (ureg.Quantity).
        """
        smearing = float(self.root.find(".//smearing").attrib["degauss"]) * ureg.Ry
        return smearing

    def runtime(self) -> ureg.Quantity:
        """
        Greps the computational runtime.

        Returns
        -------
        runtime : ureg.Quantity
            Computational runtime with attached units (ureg.Quantity).
        """
        lines = self.root.find(".//timing_info")
        runtime = float(lines.find(".//total").find(".//cpu").text)
        return runtime * ureg.second

    def k_grid(self) -> ureg.Quantity:
        """
        Greps the k-grid from a variety of filetypes.

        Returns
        -------
        k_grid : list(int)
            K-grid used in the computation.
        """
        lines = self.root.find(".//k_points_IBZ").find(".//monkhorst_pack")
        try:
            kgrid = [
                int(lines.attrib["nk1"]),
                int(lines.attrib["nk2"]),
                int(lines.attrib["nk3"]),
            ]
        except AttributeError:
            kgrid = None
        return kgrid


def electron_num(file: str) -> int:
    """
    Greps the number of electrons.

    It supports different filetypes as Quantum Espresso or VASP outputs.

    Parameters
    ----------
    file : str
        File from which to extract the electron number, it currently supports:
        - QuantumEspresso `xml` or pw.x output.
        - VASP OUTCAR.

    Returns
    -------
    num_elec : int
        Number of electrons.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The number of electrons was not found in the provided file.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in lines:
                if "number of electrons" in line:
                    num_elec = int(float(line.split()[4]))
                    break
        elif filetype == "qe_xml":
            num_elec = _Qe_xml(file).electron_num()
        elif filetype == "outcar":
            for line in lines:
                if "NELECT" in line:
                    num_elec = int(float(line.split()[2]))
                    break
        elif filetype == "eigenval":
            for line in lines:
                if len(line.split()) == 3:
                    num_elec = int(line.split()[0])
                    break
        else:
            raise NotImplementedError("Unsupported filetype")
        if "num_elec" not in locals():
            raise NameError("Number of electrons not found.")
    return num_elec


def alat(file: str) -> ureg.Quantity:
    """
    Extract the lattice parameter (alat) from an input file.

    The function reads a file produced by supported electronic-structure codes
    and returns the lattice parameter `alat` as a Pint quantity.

    Parameters
    ----------
    file : str
        Path to the input/output file.

    Returns
    -------
    ureg.Quantity
        Lattice parameter `alat` with units.

    Raises
    ------
    NotImplementedError
        If the file type is not supported.
    NameError
        If the lattice parameter `alat` cannot be found in the file.
    """
    filetype = _filetype(file)

    if filetype == "qe_xml":
        alat = _Qe_xml(file).alat()
        return alat
    elif filetype == "qe_scf_out":
        with open(file, "r") as lines:
            for line in lines:
                if "lattice parameter (alat)" in line:
                    alat = float(line.split()[-2]) * ureg("bohr/alat")
                    return alat
    elif filetype == "qe_ph_out":
        with open(file, "r") as lines:
            for line in lines:
                if "lattice parameter" in line:
                    line = line.split()
                    alat = float(line[4]) * ureg("bohr/alat")
                    return alat
    elif filetype == "qe_dyn":
        with open(file, "r") as lines:
            for line in lines:
                if len(line.split()) == 9:
                    alat = float(line.split()[3]) * ureg("bohr/alat")
                    return alat
    else:
        raise NotImplementedError(f"Unsupported filetype: {filetype}")

    if "alat" not in locals():
        raise NameError("alat not found.")


def lattice(file: str) -> ureg.Quantity:
    """
    Greps the lattice vectors from various outputs.

    Parameters
    ----------
    file : str
        File from which to extract the lattice.

    Returns
    -------
    lattice : np.ndarray
        3x3 array of lattice vectors with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """
    from ase import io

    filetype = _filetype(file)
    READ = False

    if filetype == "qe_xml":
        lattice = _Qe_xml(file).lattice()
        return lattice
    elif filetype == "qe_ph_out":
        with open(file, "r") as lines:
            for line in lines:
                if re.search("crystal axes", line, flags=re.IGNORECASE):
                    READ = True
                    lattice = []
                elif READ:
                    values = line.split()
                    vec = np.array(
                        [float(values[3]), float(values[4]), float(values[5])]
                    )
                    lattice.append(vec)
                    if len(lattice) == 3:
                        break
        lattice = np.array(lattice) * ureg.alat * alat(file)
        return lattice

    elif filetype == "qe_dyn":
        with open(file, "r") as lines:
            for line in lines:
                if "Basis vectors" in line:
                    READ = True
                    lattice = []
                elif READ:
                    vec = [float(x) for x in line.split()]
                    lattice.append(vec)
                    if len(lattice) == 3:
                        lattice = np.array(lattice) * ureg.alat * alat(file)
                        return lattice

    elif filetype == "qe_proj_out":
        raise NotImplementedError(
            "Unsupported filetype: ASE is not handling it correctly"
        )
    else:
        # Fallback to ASE
        try:
            lattice = io.read(file).cell  # (3, 3) in Å
        except io.formats.UnknownFileTypeError:
            raise NotImplementedError(
                "Unsupported filetype: ASE is not handling it correctly"
            )
        return lattice * ureg.angstrom


def fermi(file: str) -> ureg.Quantity:
    """
    Greps the Fermi energy from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the Fermi energy.

    Returns
    -------
    E_f : ureg.Quantity
        Fermi energy with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The Fermi energy was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            E_f = _Qe_xml(file).fermi()
        elif filetype == "qe_scf_out":
            for line in reversed(list(lines)):
                # If smearing is used
                if "Fermi energy is" in line:
                    E_f = float(line.split()[4])
                    break
                # If fixed occupations is used
                if "highest occupied" in line:
                    if "unoccupied" in line:
                        split = line.split()
                        E1, E2 = float(split()[6]), float(split()[7])
                        # Fermi level between the unoccupied and occupied bands
                        E_f = E1 + (E2 - E1) / 2
                    else:
                        E_f = float(line.split()[4])
                    break
            E_f *= ureg("eV")
        elif filetype == "outcar":
            for line in reversed(list(lines)):
                if "E-fermi" in line:
                    E_f = float(line.split()[2]) * ureg("eV")
                    break
        else:
            raise NotImplementedError("Unsupported filetype")
    if "E_f" not in locals():
        raise NameError("Fermi energy not found.")
    return E_f


def total_energy(file: str, decomposition: bool = False) -> float | SimpleNamespace:
    """
    Greps the total free energy or it's decomposition.

    Parameters
    ----------
    file : str
        File from which to extract the energy.
    decomposition : bool, optional
        If True an energy decomposition is returned instead. Default is False.

    Returns
    -------
    energy : float | SimpleNamespace
        If decomposition is False a single float with the free energy is returned.
        If decomposition is True a namespace with the following attributes is returned:
            -  F            -> Total Free energy
            - -TS           -> Smearing contribution
            -  U (= F+TS)   -> Internal energy
                -  U_one_electron
                -  U_hartree
                -  U_exchange-correlational
                -  U_ewald
                -  U_paw -> One center paw contribution
                -  U_correction -> Empirical correction (VdW for example.)

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The energy was not found in the provided file.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            energy = _Qe_xml(file).total_energy(decomposition)
        elif filetype == "qe_scf_out":
            data = {}
            for line in reversed(list(lines)):
                if "!" in line:
                    data["F"] = float(line.split()[4]) * ureg("Ry")
                    break
                elif "smearing contrib" in line:
                    data["TS"] = float(line.split()[4]) * ureg("Ry")
                elif "internal energy" in line:
                    data["U"] = float(line.split()[4]) * ureg("Ry")
                elif "one-electron" in line:
                    data["U_one_electron"] = float(line.split()[3]) * ureg("Ry")
                elif "hartree contribution" in line:
                    data["U_hartree"] = float(line.split()[3]) * ureg("Ry")
                elif "xc contribution" in line:
                    data["U_xc"] = float(line.split()[3]) * ureg("Ry")
                elif "ewald" in line:
                    data["U_ewald"] = float(line.split()[3]) * ureg("Ry")
                elif "Dispersion Correction" in line:
                    data["U_correction"] = float(line.split()[3]) * ureg("Ry")
                elif "one-center paw" in line:
                    data["U_paw"] = float(line.split()[4]) * ureg("Ry")
                elif "convergence NOT achieved" in line:
                    raise NameError(f"Convergence not achieved in {file}")
            if decomposition:
                energy = SimpleNamespace(**data)
            else:
                energy = data["F"]
        elif filetype == "outcar":
            for line in reversed(list(lines)):
                if "sigma->" in line:
                    l = line.split()
                    energy = float(l[-1])
                    break
            energy = energy * ureg("eV").to("Ry")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "energy" not in locals():
        raise NameError("Total energy not found.")
    return energy


def stress_tensor(file: str) -> ureg.Quantity:
    """
    Greps the total stress tensor.

    Parameters
    ----------
    file : str
        File from which to extract the stress tensor.

    Returns
    -------
    stress : ureg.Quantity
        Stress tensor with attached units.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The energy was not found in the provided file.
    """
    filetype = _filetype(file)
    READ = False
    stress = []
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in lines:
                if READ == True:
                    vec = np.array([float(x) for x in line.split()[:3]])
                    stress.append(vec)
                    if np.shape(stress) == (3, 3):
                        break
                elif re.search("total.*stress", line):
                    READ = True
            stress = np.array(stress) * (ureg("Ry") / ureg("bohr") ** 3).to("kbar")
        elif filetype == "outcar":
            for line in lines:
                if "in kB" in line:
                    l = [float(x) for x in line.split()[2:]]
                    voigt = np.array([l[0], l[1], l[2], l[4], l[5], l[3]])
                    stress = ut.voigt2cartesian(voigt) * ureg("kbar")
                    warnings.warn(
                        "According to VASP this is kB units, but when comparing to QE it appears to be GPa.",
                        UserWarning,
                    )
        else:
            raise NotImplementedError("Unsupported filetype")
        lines.close()
    if "stress" not in locals():
        raise NameError("Stress tensor not found.")
    return stress


def kpath(file: str, labels: bool = True) -> SimpleNamespace | np.ndarray:
    """
    Greps the coordinates, labels and number of poiints from the path in reciprocal space.

    Currently supports:
    - QuantumEspresso: qe_bands_in, matdyn_in.
    - VASP: KPATH (KPOINTS in line mode).

    The code expects the labels to be after the high-symmetry points commented with a `!` as:
    ...
    0   0   0   ! Gamma
    0   0.5 0   ! X
    ...

    Parameters
    ----------
    file : str
        File from which to extract the kpath.
    labels : bool, optional
        Whether labels for the high-symmetry points are extracted. Default is True.

    Returns
    -------
    kpath : SimpleNamespace | np.ndarray
        If labels is True, a namespace with attributes `path` and `labels` is returned.
        Otherwise, the kpath is returned as an ndarray.


    Raises
    ------
    NameError:
        If label or kpath is not found.
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """

    def read_qe_path(line_iter):
        kpath, k_names, N = [], [], None
        for line in line_iter:
            if N is None:
                N = int(line.split()[0])
            else:
                if labels:
                    try:
                        kpoint, label = line.split("!")
                    except ValueError:
                        raise NameError("Label not found, try using labels=False.")
                else:
                    kpoint = line
                # Grep K point
                kpoint = [float(x) for x in kpoint.split()]
                kpath.append(kpoint)
                # Grep K point label
                if labels:
                    new_name = label.split()[0]
                    k_names.append(new_name)
                # Check if path is complete
                if len(kpath) == N:
                    break
        return np.array(kpath), k_names

    filetype = _filetype(file)
    READ = EVEN = False
    kpath = k_names = N = None

    with open(file, "r") as lines:
        # QE input format
        if filetype in ["qe_bands_in", "matdyn_in"]:
            line_iter = iter(lines)
            for line in line_iter:
                if re.search("K_POINTS.*crystal_b", line, flags=re.IGNORECASE) or (
                    filetype == "matdyn_in" and re.search("/", line.split()[0])
                ):
                    kpath, k_names = read_qe_path(line_iter)
                    break
        # VASP KPATH format
        elif filetype == "kpath":
            for line in lines:
                # Grep number of points for each subpath
                if N is None:
                    try:
                        N = int(line.split()[0])
                    except ValueError:
                        pass
                elif re.search("Reciprocal", line, flags=re.IGNORECASE):
                    READ = True
                # Read path and labels
                elif READ:
                    if labels:
                        try:
                            kpoint, label = line.split("!")
                        except ValueError:
                            raise NameError("Label not found, try using labels=False.")
                    else:
                        kpoint = line
                    kpoint = [float(x) for x in kpoint.split()]
                    if kpath is None:
                        kpath = np.array([kpoint + [N]])
                        if labels:
                            k_names = [label.split()[0]]
                    # If points is different from previous in the KPATH file
                    elif (kpoint[:3] != kpath[-1][:3]).any():
                        if EVEN:
                            kpath = np.vstack([kpath, kpoint + [1]])
                        else:
                            kpath = np.vstack([kpath, kpoint + [N]])
                        if labels:
                            k_names = k_names + [label.split()[0]]
                    # If point is repeated and odd
                    elif not EVEN:
                        kpath[-1, -1] = N
                    EVEN = EVEN is False
        else:
            raise NotImplementedError("Unsupported filetype")
    if kpath is None:
        raise NameError("Kpath not found.")
    kpath = kpath * ureg._2pi / ureg.crystal
    if labels:
        # Post-process labels
        [l.replace("Gamma", r"\Gamma") for l in k_names]
        return SimpleNamespace(path=kpath, labels=k_names)
    else:
        return kpath


def kpointsEnergies(file: str) -> SimpleNamespace:
    """
    Grep the kpoints, energies, kpoint-weights  and orbital projections
    (if available) for different file kinds.

    Energies are given in eV and kpoints in reciprocal crystal units.
    Currently supports:
    - QuantumEspresso: qe_scf_out, projwfc_out, `.xml` files.
    - VASP: OUTCAR, EIGENVAL, PROCAR.

    Parameters
    ----------
    file : str
        File from which to extract the quantities.

    Returns
    -------
    SimpleNamespace : SimpleNamespace
        SimpleNamespace class with the following attributes:
        - energies : np.ndarray
            List of energies, each row corresponds to a particular k-point.
        - kpoints : np.ndarray
            List of k-points.
        - weights : np.ndarray
            List of kpoint-weights.
        - projections : _OrbitalProjectionContainer
            Container for orbital-resolved projection matrices.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    IndexError:
        If the number of weights does not coincide with the number of bands.
    """

    filetype = _filetype(file)
    READ_energies = READ_kpoints = RELAX_calc = RELAXED = OCCUPATIONS = False
    KPOINTS, ENERGIES, WEIGHTS, E, PROJ = [], [], [], [], []
    PROJECTIONS = None
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            return _Qe_xml(file).kpointsEnergies()
        elif filetype == "qe_scf_out":
            for line in lines:
                # Grep number of bands
                if "number of Kohn-Sham" in line:
                    num_bands = int(line.split("=")[1])
                elif "number of k points" in line:
                    num_points = int(line.split("=")[1].split()[0])
                elif " cryst. coord." in line:
                    READ_kpoints = True
                elif "force convergence" in line:
                    RELAX_calc = True
                elif "Final scf calculation at the relaxed" in line:
                    RELAXED = True
                elif re.search("End of .* calculation", line):
                    if (RELAX_calc == False) or (
                        RELAX_calc == True and RELAXED == True
                    ):
                        READ_energies = True
                elif READ_kpoints:
                    k = [float(x) for x in line.split("(")[2].split(")")[0].split()]
                    w = float(line.split()[-1])
                    KPOINTS.append(k)
                    WEIGHTS.append(w)
                    if len(WEIGHTS) == num_points:
                        READ_kpoints = False
                elif READ_energies:
                    if line.lstrip().startswith("k ="):
                        OCCUPATIONS = False
                    elif OCCUPATIONS:
                        # Currently not implemented
                        pass
                    elif line.strip() != "":
                        for e in re.findall(r"[-+]?\d*\.\d+|\d+", line):
                            E.append(float(e))
                        if len(E) == num_bands:
                            ENERGIES.append(E)
                            E = []
                            OCCUPATIONS = True
            # Recover crystal units
            KPOINTS = np.array(KPOINTS) * (ureg._2pi / ureg.crystal)
            ENERGIES *= ureg("eV")
        elif filetype == "qe_proj_out":
            STATES, PROJECTIONS = [], []
            for line in lines:
                if "natomwfc" in line:
                    num_states = int(line.split()[-1])
                if "nbnd" in line:
                    num_bands = int(line.split()[-1])
                if "nkstot" in line:
                    num_points = int(line.split()[-1])
                # Scrape STATES info
                if "state #" in line:
                    l = line.split()
                    if "j=" in line:
                        # atom, l, j, mj, wfc
                        state = [
                            int(l[4]) - 1,
                            int(l[9][3:]),
                            float(l[10][2:]),
                            float(l[-1].split("=")[-1][:-1]),
                            int(l[8]) - 1,
                        ]
                        STATES.append(state)
                        soc_mode = True
                    if "m=" in line:
                        # atom, l, m, wfc
                        state = [
                            int(l[4]) - 1,
                            int(l[9][3:]),
                            int(l[-1][:-1]),
                            int(l[8]) - 1,
                        ]
                        STATES.append(state)
                        soc_mode = False
                if "k =" in line:
                    k = [float(x) for x in line.split()[2:]]
                    KPOINTS.append(k)
                elif "eV ===" in line:
                    energy = float(line.split()[-3])
                    E.append(energy)
                    P = np.zeros(num_states)
                    if len(E) == num_bands:
                        ENERGIES.append(E)
                        E = []
                elif "*[#" in line:
                    l = line.split("=")[-1].split("]")[:-1]
                    for x in l:
                        split = x.split("*[#")
                        try:
                            proj = float(split[0])
                        except ValueError:
                            print(f"ValueError with {split[0]} at {k}, defaulted to 0.")
                            proj = 0
                        index = int(split[1]) - 1
                        P[index] = proj
                if "|psi|" in line:
                    PROJ.append(P)
                    if len(PROJ) == num_bands:
                        PROJECTIONS.append(PROJ)
                        PROJ = []
            KPOINTS = np.array(KPOINTS) * (ureg._2pi / ureg.alat)
            ENERGIES *= ureg("eV")
            # Save projections into the proper container
            PROJ = np.array(PROJECTIONS)
            STATES = np.array(STATES)
            PROJECTIONS = _OrbitalProjectionContainer(code="quantum_espresso")
            N = 0
            if soc_mode:
                for i, S in enumerate(STATES):
                    PROJECTIONS.add_qe(
                        ion=S[0],
                        l=S[1],
                        wfc=S[4],
                        matrix=PROJ[:, :, i],
                        j=S[2],
                        mj=S[3],
                    )
            else:
                for i, S in enumerate(STATES):
                    PROJECTIONS.add_qe(
                        ion=S[0], l=S[1], wfc=S[3], matrix=PROJ[:, :, i], m=S[2]
                    )
        elif filetype == "eigenval":
            for i, line in enumerate(lines):
                l = line.split()
                if i == 5:
                    num_points, num_bands = int(l[1]), int(l[2])
                    READ_kpoints = READ_energies = True
                elif READ_kpoints:
                    # Kpoint line
                    if len(l) == 4:
                        k = [float(x) for x in l[:3]]
                        w = float(l[-1])
                        KPOINTS.append(k)
                        WEIGHTS.append(w)
                    # Energy line
                    elif len(l) == 3:
                        E.append(float(l[1]))
                        if len(E) == num_bands:
                            ENERGIES.append(E)
                            E = []
            KPOINTS *= ureg._2pi / ureg.crystal
            ENERGIES *= ureg("eV")
        elif filetype == "outcar":
            for line in lines:
                if "NBANDS" in line:
                    num_bands = int(line.split()[-1])
                elif "Coordinates" in line and len(KPOINTS) == 0:
                    READ_kpoints = True
                elif "band No." in line:
                    READ_energies = True
                elif READ_kpoints:
                    l = line.split()
                    if len(l) != 0:
                        k = [float(x) for x in l[:3]]
                        w = float(l[-1])
                        KPOINTS.append(k)
                        WEIGHTS.append(w)
                    else:
                        num_points = len(KPOINTS)
                        READ_kpoints = False
                elif READ_energies:
                    l = line.split()
                    if len(l) == 3:
                        E.append(float(l[1]))
                        if len(E) == num_bands:
                            ENERGIES.append(E)
                            E = []
                            if len(ENERGIES) == num_points:
                                break
            KPOINTS *= ureg._2pi / ureg.crystal
            ENERGIES *= ureg("eV")
        elif filetype == "procar":
            for line in lines:
                if "k-points" in line:
                    l = line.split()
                    num_points = int(l[3])
                    num_bands = int(l[7])
                    num_ions = int(l[-1])
                elif "k-point" in line:
                    numbers = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
                    k = numbers[1:4]
                    w = numbers[-1]
                    KPOINTS.append(k)
                    WEIGHTS.append(w)
                elif "energy" in line:
                    E.append(float(line.split()[4]))
                    if len(E) == num_bands:
                        ENERGIES.append(E)
                        E = []
                elif "tot" not in line and len(line.split()) == 11:
                    proj = [float(x) for x in line.split()[1:-1]]
                    PROJ.append(proj)
            KPOINTS *= ureg._2pi / ureg.crystal
            ENERGIES *= ureg("eV")
            # Save projections into the proper container
            PROJ = np.array(PROJ)
            M = round(PROJ.shape[0] / (np.prod(np.shape(ENERGIES)) * num_ions))
            PROJECTIONS = _OrbitalProjectionContainer(code="vasp")
            for i in range(num_ions):
                for l in range(3):
                    for m in range(-l, l + 1):
                        for mag in range(M):
                            C = l + l * l + m
                            matrix = PROJ[
                                i + num_ions * mag :: num_ions * M, C
                            ].reshape(np.shape(ENERGIES))
                            PROJECTIONS.add_poscar(i, l, m, mag, matrix)
        else:
            raise NotImplementedError("Unsupported filetype")
    WEIGHTS = np.array(WEIGHTS)
    if WEIGHTS.shape[0] != ENERGIES.shape[0] and len(WEIGHTS) != 0:
        raise IndexError(
            f"There should be as many `weights` ({WEIGHTS.shape[0]}) as bands ({ENERGIES.shape[0]})."
        )
    return SimpleNamespace(
        energies=ENERGIES,
        kpoints=KPOINTS,
        weights=WEIGHTS,
        projections=PROJECTIONS,
    )


def kpointsFrequencies(file: str) -> SimpleNamespace:
    """
    Grep the kpoints and frequencies from phonon ouputs.

    Frequencies are given in cm-1 and kpoints in reciprocal alat.
    Currently supports:
    - QuantumEspresso: qe_freq_out.

    Parameters
    ----------
    file : str
        File from which to extract the spectrum.

    Returns
    -------
    SimpleNamespace : SimpleNamespace
        SimpleNamespace class with the following attributes:
        - frequencies : np.ndarray
            List of frequencies, each row corresponds to a particular k-point.
        - kpoints : np.ndarray
            List of k-points.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """

    filetype = _filetype(file)
    KPOINTS, FREQS, F = [], [], []
    READ_freqs = False
    with open(file, "r") as lines:
        if filetype == "qe_freq_out":
            for line in lines:
                l = line.split()
                if "nbnd" in line:
                    num_bands = int(l[2][:-1])
                    num_points = int(line.split("nks=")[-1][:-2])
                elif len(l) == 3:
                    k = [float(x) for x in l]
                    KPOINTS.append(k)
                    READ_freqs = True
                elif READ_freqs:
                    for f in l:
                        F.append(float(f))
                    if len(F) == num_bands:
                        FREQS.append(F)
                        F = []
        else:
            raise NotImplementedError("Unsupported filetype")
    # Give proper units
    FREQS = np.array(FREQS) * ureg("c") / ureg("cm")
    KPOINTS = np.array(KPOINTS) * ureg("_2pi/alat")
    return SimpleNamespace(frequencies=FREQS, kpoints=KPOINTS)


def dyn_file(file: str) -> SimpleNamespace:
    """
    Parse a dynamical matrix file and extract phonon mode information.

    This function extracts:
    - the lattice vectors,
    - the atomic species and their masses,
    - the atom types and positions,
    - the q-point at which the phonon modes are computed,
    - the phonon frequencies (in cm⁻¹),
    - and the polarization vectors (phonon eigenvectors).

    Parameters
    ----------
    file : str
        Path to the dynamical matrix file (e.g. QE `.dyn` or `.dynmat` file).

    Returns
    -------
    SimpleNamespace
        A container with the following fields:
        - q : ureg.Quantity, shape (3,)
            The q-point in where the calculation was performed.
        - lattice : ureg.Quantity, shape (3, 3)
            The lattice vectors of the unit cell.
        - freqs : ureg.Quantity, shape (n_modes,)
            Array of vibrational frequencies (in cm⁻¹).
        - displacements : np.ndarray, shape(n_modes,n_atoms,3)
            An (n_modes, n_atoms, 3) array of complex normalized displacement vectors for each mode.
        - positions : ureg.Quantity, shape (n_atoms, 3)
            Atomic positions.
        - elements : list of str
            Chemical symbol for each atom.
        - masses : ureg.Quantity, shape (n_atoms,)
            Atomic mass in atomic units for each atom.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """
    filetype = _filetype(file)
    if filetype != "qe_dyn":
        raise NotImplementedError("Unsupported filetype")
    lat = lattice(file)
    n_atoms = n_types = freqs = vec = None
    vec, freqs = [], []
    species = []
    atoms = []
    displacements = []
    read_modes = False
    with open(file, "r") as lines:
        for line in lines:
            l = line.split()
            if n_atoms is None and len(l) == 9:
                # Get number of species and atoms
                l = line.split()
                n_types, n_atoms = int(l[0]), int(l[1])
            elif n_types != 0 and len(l) == 4:
                # Get species
                new = [int(l[0]), l[1][1:], float(l[-1])]
                species.append(new)
                n_types -= 1
            elif n_atoms != 0 and len(l) == 5:
                # Get atomic positions
                new = [int(l[1])] + [float(x) for x in l[2:]]
                atoms.append(new)
                n_atoms -= 1
            elif "Diagonalizing" in line:
                read_modes = True
            elif read_modes:
                # Read modes
                if "q = (" in line:
                    q_point = np.array([float(x) for x in l[3:6]])
                elif "freq" in line:
                    freqs.append(float(l[-2]))
                elif len(l) == 8:
                    nums = l[1:-1]
                    new = [
                        complex(float(nums[i]), float(nums[i + 1]))
                        for i in range(0, 6, 2)
                    ]
                    vec.append(new)
                    if len(vec) == len(atoms):
                        displacements.append(vec)
                        vec = []

    # Attach units and get positions, elmenets and masses.
    ALAT = alat(file)
    positions = np.array(atoms)[:, 1:] * ureg("alat") * ALAT
    indices = np.array(atoms)[:, 0].astype(int) - 1
    elements = [species[x][1] for x in indices]
    masses = np.array([species[x][2] for x in indices]) * ureg._2m_e
    displacements = np.array(displacements)
    q_point = q_point * ureg("_2pi/alat") / ALAT
    freqs = np.array(freqs) * ureg("c/cm")

    return SimpleNamespace(
        q=q_point,
        lattice=lat,
        alat=ALAT,
        freqs=freqs,
        displacements=displacements,
        positions=positions,
        elements=elements,
        masses=masses,
    )


def _find_dyn_file(q_cryst: np.ndarray | ureg.Quantity, results_ph_path: str) -> str:
    """
    Search for the Quantum ESPRESSO `.dyn` file containing a specified q-point
    in crystalline coordinates.

    This function compares the requested q-point with those found in each
    `dyn*` file, accounting for equivalence under reciprocal lattice translations.

    Parameters
    ----------
    q_cryst : np.ndarray | ureg.Quantity
        The q-point to locate, expressed in crystalline (reduced) coordinates.
    results_ph_path : str
        Path to the folder where the phonon (`ph.x`) output `.dyn*` files are stored.

    Returns
    -------
    str
        The full path to the `.dyn` file that contains the matching q-point.

    Raises
    ------
    FileNotFoundError
        If no `.dyn*` file or no matching q-point is found in any of the `.dyn` files
    """
    from glob import glob

    # Locate a reference .dyn file to extract lattice
    dyn1 = glob(results_ph_path + "/*dyn1") + glob(results_ph_path + "/*dyn")
    if not dyn1:
        raise FileNotFoundError(
            "No 'dyn1' or 'dyn' file found in the specified folder."
        )

    # Read lattice and convert to alat units
    dyn = dyn_file(dyn1[0])
    lattice = (dyn.lattice / dyn.alat).to("alat")
    k_basis = ut.reciprocal_basis(lattice)

    # Scan all .dyn* files (excluding matdyn if present)
    dyn_files = glob(results_ph_path + "/*.dyn*")
    dyn_files = [f for f in dyn_files if "results_matdyn" not in f]

    for file in dyn_files:
        with open(file, "r") as f:
            for line in f:
                if "q = (" in line:
                    q_point = np.array([float(x) for x in line.split()[3:6]]) * ureg(
                        "_2pi/alat"
                    )
                    q_crys_from_file = ut.cartesian2cryst(q_point, k_basis)

                    # Account for periodic images using symmetry
                    for q_equiv in ut._expand_zone_border(q_crys_from_file):
                        if np.allclose(q_cryst, q_equiv, atol=1e-4):
                            return file

    raise FileNotFoundError(
        f"No `.dyn*` file found containing q = {q_cryst} in crystalline coordinates."
    )


def dyn_q(
    q_cryst: np.ndarray | ureg.Quantity, results_ph_path: str, qe_format: bool = True
) -> SimpleNamespace:
    """
    Reads the Quantum ESPRESSO `.dyn*` file corresponding to a given q-point.

    This function locates the `.dyn*` file generated by `ph.x` that corresponds to a
    desired q-point (in reduced crystalline coordinates), extracts the corresponding dynamical
    matrix, and optionally converts it to the real physical dynamical matrix in units
    of 1 / [time]².

    Parameters
    ----------
    q_cryst : np.ndarray | ureg.Quantity
        The q-point of interest, expressed in reduced crystalline coordinates (fractions of reciprocal lattice vectors).
        If not a `Quantity`, it is assumed to be in `_2pi/crystal` units.

    results_ph_path : str
        Path to the directory containing the Quantum ESPRESSO `ph.x` output `.dyn*` files.

    qe_format : bool, optional
        If True (default), returns the raw QE dynamical matrix (includes sqrt(m_i m_j) mass factors).
        If False, converts the dynamical matrix to true physical form in 1 / [time]² units (Ry/h)^2.

    Returns
    -------
    system : SimpleNamespace
        A container with the following fields:
        - q : ureg.Quantity, shape (3,)
            The q-point in where the calculation was performed.
        - lattice : ureg.Quantity, shape (3, 3)
            The lattice vectors of the unit cell.
        - freqs : ureg.Quantity, shape (n_modes,)
            Array of vibrational frequencies (in cm⁻¹).
        - positions : ureg.Quantity, shape (n_atoms, 3)
            Atomic positions.
        - elements : list of str
            Chemical symbol for each atom.
        - masses : ureg.Quantity, shape (n_atoms,)
            Atomic mass in atomic units for each atom.
        - dyn: ureg.Quantity
            The (3N × 3N) complex dynamical matrix (units depend on `qe_format`).

    Notes
    -----
    - The dynamical matrix read from QE includes a sqrt(m_i m_j) prefactor for each (3×3) subblock.
      This must be removed to obtain the physical matrix for diagonalization (ω² in Ry²/ħ²).
    - The units of the returned matrix are `_2m_e * Ry^2 / planck_constant^2` in QE format.
    """
    from yaiv.phonon import _QEdyn2Realdyn

    # Normalize units
    if isinstance(q_cryst, ureg.Quantity):
        q_cryst = q_cryst.to("_2pi/crystal")
    else:
        q_cryst = q_cryst * ureg("_2pi/crystal")

    file = _find_dyn_file(q_cryst, results_ph_path)

    system = dyn_file(file)
    dim = 3 * len(system.elements)
    dyn_mat = np.zeros((dim, dim), dtype=complex)

    # Lattice and reciprocal basis
    lattice = (system.lattice / system.alat).to("alat")
    k_basis = ut.reciprocal_basis(lattice)

    READ_dynmat = False
    with open(file, "r") as lines:
        for line in lines:
            if "q = (" in line:
                q_point = np.array([float(x) for x in line.split()[3:6]]) * ureg(
                    "_2pi/alat"
                )
                q_crys_from_file = ut.cartesian2cryst(q_point, k_basis)

                for q_equiv in ut._expand_zone_border(q_crys_from_file):
                    if np.allclose(q_cryst.magnitude, q_equiv.magnitude, atol=1e-4):
                        READ_dynmat = True
                        break

            if READ_dynmat:
                l = line.split()
                if len(l) == 2:  # matrix block index line
                    n, m = int(l[0]), int(l[1])
                    num = 0
                elif len(l) == 6:  # submatrix row
                    row = np.array(
                        [
                            complex(float(l[0]), float(l[1])),
                            complex(float(l[2]), float(l[3])),
                            complex(float(l[4]), float(l[5])),
                        ]
                    )
                    sub_mat = row if num == 0 else np.vstack([sub_mat, row])
                    num += 1
                    if num == 3:
                        i, j = 3 * (n - 1), 3 * (m - 1)
                        dyn_mat[i : i + 3, j : j + 3] = sub_mat
                if re.search("Dynamical", line) or re.search("Diagonalizing", line):
                    break

    # Clean output
    system.q = q_cryst
    delattr(system, "displacements")
    system.dyn = dyn_mat * ureg("_2m_e * Ry^2 / planck_constant^2")
    if not qe_format:
        system.dyn = _QEdyn2Realdyn(system.dyn, system.masses)

    return system


def symmetries(file: str) -> list[SimpleNamespace]:
    """
    Grep symmetry operations and return them as rotation/translation pairs.

    This reads all symmetry elements, extracts the 3×3 rotation matrix and
    a fractional translation:
      - R: np.ndarray shape (3, 3) in crystal coord.
      - t: ureg.Quantity length-3 vector in units of 2π/crystal

    Parameters
    ----------
    file : str
        Path to the output file containing symmetry information (e.g., QE XML).

    Returns
    -------
    symmetries : list[SimpleNamespace]
        Each entry contains R (3×3 rotation matrix) and t (fractional translation
        vector in 2π/crystal units).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    """
    filetype = _filetype(file)
    if filetype == "qe_xml":
        symmetries = _Qe_xml(file).symmetries()
    else:
        raise NotImplementedError("Unsupported filetype")
    return symmetries


def cutoff(file: str) -> ureg.Quantity:
    """
    Greps the cutoff energy from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the cutoff energy.

    Returns
    -------
    cutoff : ureg.Quantity
        Cutoff energy with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The cutoff energy was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            cutoff = _Qe_xml(file).cutoff()
        elif filetype == "qe_scf_out":
            for line in lines:
                # If smearing is used
                if "kinetic-energy cutoff" in line:
                    cutoff = float(line.split()[-2])
                    break
            cutoff *= ureg("Ry")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "cutoff" not in locals():
        raise NameError("Cutoff energy not found.")
    return cutoff


def smearing(file: str) -> ureg.Quantity:
    """
    Greps the smearing from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the smearing.

    Returns
    -------
    smearing : ureg.Quantity
        Smearing with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The smearing was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            smearing = _Qe_xml(file).smearing()
        elif filetype == "qe_scf_out":
            for line in lines:
                if "smearing, width" in line:
                    smearing = float(line.split()[-1])
                    break
            smearing *= ureg("Ry")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "smearing" not in locals():
        raise NameError("Smearing energy not found.")
    return smearing


def k_grid(file: str) -> list[int]:
    """
    Greps the k-grid from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the k-grid.

    Returns
    -------
    k_grid : list(int)
        K-grid used in the computation.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The k-grid was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            kgrid = _Qe_xml(file).k_grid()
        elif filetype == "qe_scf_in":
            READ = False
            for line in lines:
                if "K_POINTS" in line:
                    READ = True
                elif READ:
                    l = line.split()
                    if len(l) >= 3:
                        kgrid = [int(x) for x in l[:3]]
                        break
        elif filetype == "qe_scf_out":
            # Try to find the corresponding input file
            kgrid = k_grid(file[:-1] + "i")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "kgrid" not in locals():
        raise NameError("K-grid not found.")
    return kgrid


def atomic_forces(file: str) -> SimpleNamespace:
    """
    Greps the atomic forces from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the atomic forces.

    Returns
    -------
    forces : SimpleNamespace
        SimpleNamespace class with the following attributes:
        SimpleNamespace with atomic forces with attached units (ureg.Quantity).
        - per_atom : np.ndarray | ureg.Quantity
            List of foces per atom with shape (N,3) where N is the number of atoms.
        - total : np.ndarray | ureg.Quantity
            Total average force.

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The atomic forces were not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            READ, WRITE = False, False
            forces = []
            for line in lines:
                if "Forces acting on atoms" in line:
                    READ, WRITE = True, True
                elif READ:
                    l = line.split()
                    if "atom" in line and WRITE:
                        f = [float(x) for x in l[6 : 6 + 3]]
                        forces.append(f)
                    elif "Total force" in line:
                        total_force = float(l[3])
                        break
                    elif len(forces) != 0:
                        WRITE = False
            forces = np.asarray(forces) * ureg("Ry/bohr")
            total_force *= ureg("Ry/bohr")
        else:
            raise NotImplementedError("Unsupported filetype")
    if "forces" not in locals():
        raise NameError("Atomic forces not found.")
    return SimpleNamespace(per_atom=forces, total=total_force)


def runtime(file: str) -> ureg.Quantity:
    """
    Greps the computational runtime from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the computational runtime.

    Returns
    -------
    runtime : ureg.Quantity
        Computational runtime with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The computational runtime was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_xml":
            runtime = _Qe_xml(file).runtime()
        elif filetype == "qe_scf_out":
            for line in reversed(list(lines)):
                # If smearing is used
                if "PWSCF" in line:
                    h, m, s = 0, 0, 0
                    runtime = line.split()[2]
                    if "h" in runtime:
                        h = int(runtime.split("h")[0])
                        runtime = runtime.split("h")[1]
                    if "m" in runtime:
                        m = int(runtime.split("m")[0])
                        runtime = runtime.split("m")[1]
                    if "s" in runtime:
                        s = float(runtime.split("s")[0])
                    runtime = s * ureg.second + m * ureg.minute + h * ureg.hour
                    break
        else:
            raise NotImplementedError("Unsupported filetype")
    if "runtime" not in locals():
        raise NameError("Computation runtime not found.")
    return runtime


def ram(file: str) -> ureg.Quantity:
    """
    Greps the RAM needed in the computation from a variety of filetypes.

    Parameters
    ----------
    file : str
        File from which to extract the RAM.

    Returns
    -------
    RAM : ureg.Quantity
        RAM with attached units (ureg.Quantity).

    Raises
    ------
    NotImplementedError:
        The function is not currently implemeted for the provided filetype.
    NameError:
        The RAM was not found.
    """
    filetype = _filetype(file)
    with open(file, "r") as lines:
        if filetype == "qe_scf_out":
            for line in lines:
                if "total dynamical RAM" in line:
                    RAM = float(line.split()[5])
                    units = ureg(line.split()[6])
                    break
            RAM *= units
        else:
            raise NotImplementedError("Unsupported filetype")
    if "RAM" not in locals():
        raise NameError("RAM not found.")
    return RAM
