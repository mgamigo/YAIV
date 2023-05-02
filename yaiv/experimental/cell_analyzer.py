import numpy as np
import matplotlib.pyplot as plt
import spglib as spg
import re
import os

from ase.io import read, write
from ase.visualize import view
from ase import Atoms

from crystal_toolkit.renderables import StructureGraph
from pymatgen.analysis.local_env import MinimumDistanceNN
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from crystal_toolkit.core.legend import Legend
from crystal_toolkit.renderables.structuregraph import get_structure_graph_scene

from yaiv.experimental.matview.visualizers.crystal import CrystalVisualizer


#from ase_notebook import AseView, ViewConfig 


def ase2spglib(crystal_ase): #not really needed since spglib reads the ase atoms type
    lattice=np.array(crystal_ase.get_cell())
    positions=crystal_ase.get_scaled_positions()
    numbers=crystal_ase.get_atomic_numbers()
    spg_crystal=(lattice,positions,numbers)
    return spg_crystal

def spglib2ase(spglib_crystal):
    lattice=spglib_crystal[0]
    positions=spglib_crystal[1]
    numbers=spglib_crystal[2]
    ase_crystal=Atoms(scaled_positions=positions,numbers=numbers,cell=lattice)
    return ase_crystal

def get_spacegroup(crystal,symprec=1e-5,silent=False):
    """Get the spacegroup of almost any file using spglib
    symprec = Symmetry thushold as defined by spglib
    silent = bolean (whether to print the result)
    """
    if type(crystal)==str:   #We are loading a file
        ASE=read(crystal)
        SPG=ase2spglib(ASE)
    elif type(crystal)==Atoms:  #We have an ase structure
        SPG=ase2spglib(crystal)
    elif type(crystal)==tuple:   #spglib structure
        SPG=crystal
    else:
        print('Cannot print! Don\'t understand format')
    SG=spg.get_spacegroup(SPG,symprec=symprec)
    if silent==False:
        print('SpaceGroup =',SG)
    return SG

#OLD DEPRECATED.... NEW VERSION OVERWRITING
def visualize(crystal,gui=False,svg=False,repeat_uc=(1,1,1),miller_planes=None,center_in_uc=False,conventional=False,rotations='-130x,-130y,40z'):
    """
    crystal = Either a file, an ase atoms objetct or an spglib object
    svg = bolean (displays an static picture)
    gui = bolean (opens a window with the usual ASE visualizer)
    miller_planes = list of miller planes as: [(1,1,0),(0,0,1)]
    center_in_uc = False (shift atoms to the unit cell)
    conventional = False (whether to draw the conventional cell)
    """
    config = ViewConfig(
    rotations=rotations, #initial rotation
    atom_font_size=12,
    canvas_size=(700, 400),
    zoom=1.3,
    show_bonds=True,
    atom_show_label=True,
    show_uc_repeats=True,
    show_axes=True,
    axes_uc=True,
    axes_length=50,
    uc_dash_pattern=(.6,.4)
    )
    ase_view = AseView(config)

    if type(crystal)==str:   #We are loading a file
        SPG=read_spg(crystal)
        ASE=spglib2ase(SPG)
    elif type(crystal)==Atoms:  #We have an ase structure
        ASE=crystal
        SPG=ase2spglib(ASE)
    elif type(crystal)==tuple:   #spglib structure
        SPG=crystal
        ASE=spglib2ase(SPG)
    else:
        print('Cannot print! Don\'t understand format')

    if conventional==True:
        SPG=spg.standardize_cell(SPG)
        ASE=spglib2ase(SPG)

    atoms_number=len(SPG[1])
    get_spacegroup(crystal)
    print(atoms_number,"atoms")
    if miller_planes!=None:
        for p in miller_planes:
            ase_view.add_miller_plane(p[0],p[1],p[2],color='pink')

    if gui==True:
        gui=ase_view.make_gui(ASE,repeat_uc=repeat_uc,center_in_uc=center_in_uc)
    elif svg==True:
        gui=ase_view.make_svg(ASE,repeat_uc=repeat_uc,center_in_uc=center_in_uc)
    else:
        gui=ase_view.make_render(ASE,repeat_uc=repeat_uc,center_in_uc=center_in_uc)
    return gui


def visualize(crystal,matview=True,local_env=True,neighbours=True,conventional=False):
    """
    crystal = Either a file, an ase atoms objetct or an spglib object
    matview = Boolean if you want the interface from matview
    local_env = Boolean controling if want to show local enviroment
    neighbours = Boolean controling wheter to show bonded sites outside the unit cell
    conventional = False (whether to draw the conventional cell)
    """
    if type(crystal)==str:   #We are loading a file
        SPG=read_spg(crystal)
        ASE=spglib2ase(SPG)
    elif type(crystal)==Atoms:  #We have an ase structure
        ASE=crystal
        SPG=ase2spglib(ASE)
    elif type(crystal)==tuple:   #spglib structure
        SPG=crystal
        ASE=spglib2ase(SPG)
    else:
        print('Cannot print! Don\'t understand format')

    if conventional==True:
        SPG=spg.standardize_cell(SPG)
        ASE=spglib2ase(SPG)

    atoms_number=len(SPG[1])
    get_spacegroup(crystal)
    print(atoms_number,"atoms")
    structure=AseAtomsAdaptor.get_structure(ASE)
    
    if matview==True:
        view = CrystalVisualizer(structure)
        return view.show()
    else:
        StructureGraph.get_scene = lambda x: get_structure_graph_scene(
        x,
        bond_radius=0.1, 
        legend=Legend(structure, color_scheme="VESTA"),
        bonded_sites_outside_unit_cell=neighbours
        )
        if local_env==True:
            graph = StructureGraph.with_local_env_strategy(structure, MinimumDistanceNN())
        else:
            graph = StructureGraph.with_empty_graph(structure)
        return graph.get_scene()

def read_spg(file):
    cryst=read(file)
    spgcryst=ase2spglib(cryst)
    return spgcryst

def get_sym_info(crystal,symprec=1e-5):
    """A simple report of main symmetry asepects
    crystal = Either QE/VASP file, spglib or ase object.
    """
    if type(crystal)==str:   #We are loading a file
        ASE=read(crystal)
        SPG=ase2spglib(ASE)
    elif type(crystal)==Atoms:  #We have an ase structure
        ASE=crystal
        SPG=ase2spglib(ASE)
    elif type(crystal)==tuple:   #spglib structure
        ASE=spglib2ase(crystal)
        SPG=crystal
    else:
        print('Cannot print! Don\'t understand format')
    
    print(ASE.get_chemical_formula())
    dataset=spg.get_symmetry_dataset(SPG,symprec=symprec)
    print('SpaceGroup =',dataset["international"],'('+str(dataset["number"])+')')
    print()
    print('ATOMS:')
    print(ASE.get_chemical_symbols())
    print('Wyckoffs:')
    print(dataset["wyckoffs"])
    print('Equivalent positions:')
    print(dataset["equivalent_atoms"])
    print('Site symmetry simbols:')
    print(dataset["site_symmetry_symbols"])
    print()
    print('SYMMETRY OPERATIONS:')
    print()
    symmetry=[(r, t) for r, t in zip(dataset['rotations'], dataset['translations'])]
    for i in range(len(symmetry)):
        rot=symmetry[i][0]
        t=np.around(symmetry[i][1],decimals=3)
        __rot_name(rot,SPG)
        print(rot,'+',t)
        print()

def __rot_name(rot,SPG):
    """gives names to symmetry operations"""
    E=np.identity(3)
    r=rot
    [eigvalues,eigvectors]=np.linalg.eig(rot)
    d=1
    while (r!=E).any():
        r=np.matmul(r,rot)
        d=d+1
    det=np.linalg.det(rot)
    if det==1:
        if d==1:
            print('E')
        else:
            index = np.where(np.isclose(eigvalues,1))[0][0]
            direction=eigvectors[:,index].real
            directionxyz=np.matmul(direction,SPG[0])
            print('C'+str(d),'// [a,b,c]=',np.around(direction/np.linalg.norm(direction),decimals=3),
                 ' // [x,y,z]=',np.around(directionxyz/np.linalg.norm(directionxyz),decimals=3))
    elif det==-1:
        index = np.where(np.isclose(eigvalues,-1))[0][0]
        direction=eigvectors[:,index].real
        directionxyz=np.matmul(direction,SPG[0])
        if d==2:
            if (rot==-E).all():
                print('I')
            else:
                print('m','// [a,b,c]=',np.around(direction/np.linalg.norm(direction),decimals=3),
                     ' // [x,y,z]=',np.around(directionxyz/np.linalg.norm(directionxyz),decimals=3))
        else:
            if d==3:
                print('S3','// [a,b,c]=',np.around(direction/np.linalg.norm(direction),decimals=3),
                     ' // [x,y,z]=',np.around(directionxyz/np.linalg.norm(directionxyz),decimals=3))
            elif d==6:
                print('S6','// [a,b,c]=',np.around(direction/np.linalg.norm(direction),decimals=3),
                     ' // [x,y,z]=',np.around(directionxyz/np.linalg.norm(directionxyz),decimals=3))
    else:
        print('sym element not detected')   


def store_structure_QE_pwi(structure,filename,template=None):
    """Writes a QE input based on an ase structure:
    
    structure = your ase or spglib structure
    filename = name of your input
    template = A template input you want to copy
    """
    if type(structure)==tuple:   #spglib structure
        structure=spglib2ase(structure)
    if template==None:
        filename=filename
        write(filename,structure,format='espresso-in')
    else:
        write('.tmp.pwi',structure,format='espresso-in')
        
        #process relevant data of new structure
        basis=[]
        pos=[]
        cell_line=-4
        pos_line=-999999
        nat=0
        file=open('.tmp.pwi','r')
        for n,line in enumerate(file):
            if re.search('nat',line):
                nat=int(line.split()[2])
            if n-cell_line<=3:
                basis=basis+[line]
            if n-pos_line<=nat:
                pos=pos+[line]
            if re.search('CELL_PARAMETERS',line):
                cell_line=n
            if re.search('ATOMIC_POSITIONS',line):
                pos_line=n
        file.close()
        os.remove('.tmp.pwi')
        
        #open template and change input accordingly
        write_nat=True
        write_pos=True
        write_basis=True
        K_points=False
        temp=open(template,'r')
        output=open(filename,'w')
        for line in temp:
            if re.search('ibrav',line):
                if '0' not in line:
                    print('ERROR: Your template has not ibrav=0')
                    return ERROR
            elif re.search('pseudo_dir',line):
                line="  pseudo_dir = 'pseudo',\n"
            elif re.search('outdir',line):
                line="  outdir = './tmp',\n"
            elif re.search('nat*=',line) and write_nat==True:
                line='  nat='+str(nat)+',\n'
                write_nat=False
            elif re.search('POSITIONS',line,re.IGNORECASE) and write_pos==True:
                line='ATOMIC_POSITIONS {angstrom}\n'
                output.write(line)
                for line in pos:
                    output.write(line)
                write_pos=False
            elif  re.search('POINTS',line,re.IGNORECASE):
                K_points=True
            elif  re.search('CELL',line,re.IGNORECASE):
                line='CELL_PARAMETERS {angstrom}\n'
                output.write(line)
                for line in basis:
                    output.write(line)
                K_points=False
            if write_pos==True or K_points==True:
                output.write(line)
        temp.close()
        output.close()

def wyckoff_positions(crystal):
    """Reads a QE/VASP, ASE or spglib structures and returns the independent Wyckoff positions

    indep_symb = A list with the symbols of the independent WP elements
    indep_WP = A list with the independent WP
    positions = A list with the positions of each of the indep WP
    indices = A list with the indices of the atoms corresponding to each of the WP

    return indep_symb,indep_WP,positions,indices
    """
    if type(crystal)==str:   #We are loading a file
        ASE=read(crystal)
        cryst=ase2spglib(ASE)
    elif type(crystal)==Atoms:  #We have an ase structure
        cryst=ase2spglib(crystal)
    elif type(crystal)==tuple:   #spglib structure
        cryst=crystal
    else:
        print('Cannot print! Don\'t understand format')
    ASE=spglib2ase(cryst)
    sym_dataset=spg.get_symmetry_dataset(cryst)
    equiv=sym_dataset["equivalent_atoms"]
    WP=sym_dataset["wyckoffs"]
    symb=ASE.get_chemical_symbols()

    positions=[]
    indep_WP=[]
    indep_symb=[]
    indices=[]

    repeated=[]
    for i,n in enumerate(equiv):
        if n not in repeated:
            positions=positions+[cryst[1][i,:]]
            indep_WP=indep_WP+[WP[i]]
            indep_symb=indep_symb+[symb[i]]
            indices=indices+[[i]]
            repeated=repeated+[equiv[i]]
        else:
            index=repeated.index(equiv[i])
            positions[index]=np.vstack((positions[index],cryst[1][i,:]))
            indices[index]=indices[index]+[i]
    return indep_symb,indep_WP,positions,indices

def write_struc(crystal,file,primitive=True,conventional=False,silent=True):
    """Output a structure in a sensible readable way
    crystal = Either QE/VASP file, spglib or ase object.
    file = File name for your output
    primitive = bolean (whether you want the primitive cell)
    conventional = bolean (whether you want the conventional cell)
    silent = bolean (whether you want some sort of printed output)
    """
    print(type(crystal))
    if type(crystal)==str:   #We are loading a file
        ASE=read(crystal)
        SPG=ase2spglib(ASE)
    elif type(crystal)==Atoms:  #We have an ase structure
        SPG=ase2spglib(crystal)
    elif type(crystal)==tuple:   #spglib structure
        SPG=crystal
    else:
        print('Cannot print! Don\'t understand format')
    if primitive==True:
        SPG=spg.find_primitive(SPG)
    elif conventional==True:
        SPG=spg.standardize_cell(SPG)
    ASE=spglib2ase(SPG)
    CELL=np.array(ASE.get_cell())
    POS=np.array(ASE.get_scaled_positions())
    SYM=ASE.get_chemical_symbols()
    #create(positions)
    for i,s in enumerate(SYM):
        new=np.hstack([s,np.array(POS[i],dtype=object)])
        try:
            positions=np.vstack((new,positions))
        except NameError:
            positions=new
    #print(positions)
    if silent==True:
        np.savetxt(file, CELL, fmt='%14.9f',header='CELL (Anstrom)')
        fmt='%-2s %14.9f %14.9f %14.9f'
        with open(file,"ab") as f:
            np.savetxt(f,positions,header='\n Atomic Positions (crystal)',fmt=fmt)
    else:
        print(CELL)
        positions[:,1:]=np.around(np.array(positions[:,1:],dtype=float),decimals=8)
        print(positions)
