#PYTHON module with code developed to analyze the Quantum Geometry

import numpy as np

import yaiv.utils as ut
import yaiv.experimental.cell_analyzer as cell
import yaiv.experimental.phonon as ph


def __generate_satellites(KPOINTS,displacement=0.001):
    """
    General for a list of points of any dimensionality:
    
    KPOINTS = List of K_points (in carterian coord)
    displacement = Displacement to be applied in all cartesian coordinates to the points

    returns a new list of points with the original points and the satelites.
    """
    shifts = np.identity(KPOINTS.shape[1])*displacement  # Identity matrix gives unit vectors for each axis

    for K in KPOINTS:
        satellite_points =  [K] + \
                           [list(K + shift) for shift in shifts] + \
                           [list(K - shift) for shift in shifts]
        try:
            SATELLITES=np.vstack([SATELLITES,satellite_points])
        except NameError:
            SATELLITES=satellite_points
    return SATELLITES

def Kpath_generator(KPATH,lattice=None,crystal=True,satellites=True,displacement=0.001):
    """
    Generates a list of Kpoints replicating the desired KPATH

    KPATH = List of High Symmetry Points defining the path in QE format:
            [kx ky kz num_points
             ... ... .. ... ...]
    lattice = Lattice vectors
    crystal = (Boolean) If True the output will be in crystal coordinates (*2pi)
    satellites = Whether to create salite points around each point.
        displacement = Displacement to be applied in all cartesian coordinates to the points.

    returns a list of K-points
    """
    #Reading / converting input
    if np.any(lattice) == None:
        lattice = np.identity(3)
    Kbasis=ut.K_basis(lattice)*2*np.pi

    #Process:
    for i,K in enumerate(KPATH[:-1]):
        k0,N=K[0:3],int(K[3])
        k1=KPATH[i+1][0:3]
        path=np.linspace(k0,k1,N+1)
        try:
            PATH=np.vstack([PATH,path[1:]]) #The [1:] is for avoiding duplicate points
        except NameError:
            PATH=path
    #Generate satellites
    if satellites == True:
        PATH=ut.cryst2cartesian(PATH,Kbasis,list_of_vec=True)
        PATH=__generate_satellites(PATH,displacement)
        PATH=ut.cartesian2cryst(PATH,Kbasis,list_of_vec=True)
    if crystal==False:
        PATH=ut.cryst2cartesian(PATH,Kbasis,list_of_vec=True)
    return PATH


def Kgrid_generator(grid,lattice=None,crystal=True,satellites=True,displacement=0.001):
    """
    Generates a list of Kpoints replicating a desired K-mesh

    grid = [N1,N2,N3...] describing your grid in any dimension
    lattice = Lattice vectors
    crystal = (Boolean) If True the output will be in crystal coordinates (*2pi)
    satelites = Whether to create salite points around each point.
        displacement = Displacement to be applied in all cartesian coordinates to the points.

    returns a list of K-points
    """
    #Reading / converting input
    if np.any(lattice) == None:
        lattice = np.identity(len(grid))
    Kbasis=ut.K_basis(lattice)*2*np.pi

    #Process:
    GRID=ph.__grid_generator(grid,from_zero=True)
    #Generate satellites
    if satellites == True:
        GRID=ut.cryst2cartesian(GRID,Kbasis,list_of_vec=True)
        GRID=__generate_satellites(GRID,displacement)
        GRID=ut.cartesian2cryst(GRID,Kbasis,list_of_vec=True)
    if crystal==False:
        GRID=ut.cryst2cartesian(GRID,Kbasis,list_of_vec=True)
    return GRID


def nnkp_generator(grid):
    """
    Generates a list of nnkp neighbours for computing the desired overlaps.
    It assumes that the list of kpoints is given in groups of 7, with the first being the center and the rest satellites.

    grid = [N1,N2,N3] describing your grid in 3D

    returns a list of nnkp-neighbours
    """

    N=np.prod(grid)
    for n in range(N):
        START=7*n+1
        END=7*n+8
        for i in range(START,END):
            for j in range(START,END):
                if i!=j:
                    line=np.array([i,j,0,0,0],dtype=int)
                    try:
                        OUT=np.vstack((OUT,line))
                    except NameError:
                        OUT=line
    return OUT
