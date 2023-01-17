#PYTHON module with nice tu have utilitary tools, mostly focussed on text scraping

import numpy as np
import re
from ase import io

import yaiv.constants as const

# GREPPING utilities----------------------------------------------------------------

class file:
    """A class for file scraping, depending on the filetype a different set of attributes will initialize.
    The filetype should be automatically detected, but can be manually introduced:
    QuantumEspresso: qe_scf_in, qe_scf_out, qe_bands_in, qe_ph_out, matdyn_in
    VASP: POSCAR, OUTCAR, KPATH (KPOINTS in line mode), EIGENVAL
    """
    def __init__(self,file,filetype=None):
        self.file = file
        #Define file type
        if filetype == None:
            self.filetype = grep_filetype(file)
        else:
            self.filetype = filetype.lower()
        #Read attributes:
        if self.filetype in ['qe_scf_out','qe_scf_in','qe_bands_in','qe_ph_out','outcar','poscar']:
            self.lattice = grep_lattice(self.file,filetype=self.filetype)
        if self.filetype in ['qe_scf_out','outcar']:
            self.electrons = grep_electrons(file,filetype=self.filetype)
        if self.filetype in ['qe_scf_out','outcar']:
            self.fermi = grep_fermi(file,filetype=self.filetype)
        if self.filetype == 'kpath':
            self.path,self.labels = grep_ticks_labels_KPATH(file)
        if self.filetype in ['qe_bands_in','matdyn_in']:
            self.path = grep_ticks_QE(self.file,self.filetype)
    def __str__(self):
        return str(self.filetype) + ':\n' + self.file
    def grep_lattice(self,alat=False):
        """Check grep_lattice function"""
        self.lattice = grep_lattice(self.file,filetype=self.filetype,alat=alat)
        return self.lattice
    def reciprocal_lattice(self,alat=False):
        """Check K_basis function"""
        if hasattr(self, 'lattice'):
            return K_basis(self.lattice,alat=alat)
        else:
            print('No lattice data in order to compute reciprocal lattice')
    def grep_ph_grid_points(self,expanded=False,decimals=3):
        """Check grep_ph_grid_points function"""
        if self.filetype != 'qe_ph_out':
            print('This method if for ph.x outputs, which this is not...')
            print('Check the documentation for grep_ph_grid_points function')
        else:
            grid = grep_ph_grid_points(self.file,expanded=expanded,decimals=decimals)
            self.ph_grid_points = grid
            return grid
    def grep_total_energy(self,meV=False):
        """Returns the total energy in (Ry). Check grep_total_energy"""
        out= grep_total_energy(self.file,meV=meV,filetype=self.filetype)
        self.total_energy = out
        return out

def grep_filetype(file):
    """Returns the filetype, currently it supports:
    QuantumEspresso: qe_scf_in, qe_scf_out, qe_bands_in, qe_ph_out, matdyn_in
    VASP: POSCAR, OUTCAR, KPATH (KPOINTS in line mode), EIGENVAL
    Anything else is considered a general 'data' type
    """
    lines = open(file)
    counter=0
    for line in lines:
        if re.search('calculation.*scf.*',line,re.IGNORECASE) or re.search('calculation.*nscf.*',line,re.IGNORECASE):
            filetype='qe_scf_in'
            break
        elif re.search('Program PWSCF',line,re.IGNORECASE):
            filetype='qe_scf_out'
            break
        elif re.search('Program PHONON',line,re.IGNORECASE):
            filetype='qe_ph_out'
            break
        elif re.search('calculation.*bands.*',line,re.IGNORECASE):
            filetype='qe_bands_in'
            break
        elif re.search('flfrc',line,re.IGNORECASE):
            filetype='matdyn_in'
            break
        elif re.search('direct',line,re.IGNORECASE) or re.search('cartesian',line,re.IGNORECASE):
            filetype='poscar'
            break
        elif re.search('vasp',line,re.IGNORECASE):
            filetype='outcar'
            break
        elif len(line.split()) == 4 and all([x.isdigit() for x in line.split()]):
            filetype='eigenval' 
            break
        elif re.search('line.mode',line,re.IGNORECASE):
            filetype='kpath' 
            break
        else:
            filetype='data'
    return filetype

def grep_lattice(file,alat=False,filetype=None):
    """Greps the lattice vectors from a variety of outputs (it uses ase)

    alat = Bolean controling if you want your lattice normalized (mod(a0) = 1, alat units)
   
    The filetype should be given by the function grep_filetype(file)

    OUTPUT= np.array([vec1,vec2,vec3])
    """
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype = filetype.lower()
    if filetype=='qe_ph_out':
        lattice_lines=False
        lines=open(file,'r')
        for line in lines:
            if re.search('lattice parameter',line):
                line=line.split()
                alat_au=float(line[4])
            elif lattice_lines==True:
                X=float(line.split()[3])
                Y=float(line.split()[4])
                Z=float(line.split()[5])
                vec=np.array([X,Y,Z])
                try:
                    lattice=np.vstack((lattice,vec))            
                except NameError:
                    lattice=vec
                if lattice.shape == (3,3):
                    break
            elif re.search('crystal axes',line,flags=re.IGNORECASE):
                lattice_lines=True
        if alat == True:
            return lattice
        else:
            return lattice*alat_au*const.au2ang
    else:
        import warnings
        warnings.filterwarnings("ignore", message="Non-collinear spin is not yet implemented. Setting magmom to x value.")
        try:
            data=io.read(file)
            lattice=np.array(data.cell)
            if alat == True:
                lattice = lattice/np.linalg.norm(lattice[0])
        except:
            lattice=None
            print('No lattice data found')
    return lattice

def grep_fermi(file,filetype=None):
    """Greps the Fermi level from a variety of filetypes and returns it in eV
    The filetype should be detected automatically, but it supports:
    qe_scf_out (Quantum Espresso), OUTCAR (VASP)
    """
    E_f=None
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype=filetype.lower()
    if filetype[:2]=='qe':
        scf_out=open(file,'r')
        for line in scf_out:
            if re.search('Fermi energy is',line):
                E_f=float(line.split()[4])
            if re.search('highest occupied',line):
                if re.search('unoccupied',line):
                    E1=float(line.split()[6])
                    E2=float(line.split()[7])
                    print('The gap is',(E2-E1)*1000,'meV')
                    E_f=E1+(E2-E1)/2
                else:
                    E_f=float(line.split()[4])
    elif filetype=='outcar':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('E-fermi',line):
                E_f=float(line.split()[2])
    return E_f

def grep_electrons(file,filetype=None):
    """Greps the number of electrons from a scf.pwo or OUTCAR file.
    The filetype should be detected automatically, but it supports:
    qe_scf_out (Quantum Espresso), OUTCAR (VASP)
    """
    num_elec=None
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype=filetype.lower()
    if filetype[:2]=='qe':
        scf_out=open(file,'r')
        for line in scf_out:
            if re.search('number of electrons',line):
                num_elec=int(float(line.split()[4]))
    elif filetype=='outcar':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('NELECT',line):
                num_elec=int(float(line.split()[2]))
    return num_elec

def grep_ticks_labels_KPATH(file):
    """Greps ticks and labels of the ticks from a KPATH file of VASP.
    It expects the file to have the structure:
    0 0 0 !GM
    0 0.5 0 !X
    0 0.5 0 !X
    0.5 0.5 0 !

    It outputs two variables, the PATH:
    np.array([K-point1, # of points to next],
             [K-point2, # of points to next],
              ...]


    and a list of LABELS:
    [label1, label2, label3 ...]
    """
    KPATH=open(file,'r')
    ticks=np.zeros(0)
    q0=np.zeros(0)
    repeat=False
    labels=[]
    path_section=False
    for line in KPATH:
        if path_section==True:
            if len(line.split())!=0:
                X=float(line.split()[0])
                Y=float(line.split()[1])
                Z=float(line.split()[2])
                q1=np.array([X,Y,Z,100])
                label=line.split('!')[1].split()[0]
                if not np.array_equal(q0,q1):
                    labels=labels+[label]
                    if len(ticks)==0:
                        ticks=q1
                    else:
                        if repeat==False:
                            if len(ticks.shape)>1:
                                if ticks[len(ticks)-2,3]!=1:
                                    ticks[len(ticks)-1,3]=1
                            ticks=np.vstack((ticks,q1))
                            q0=q1
                        else:
                            ticks=np.vstack((ticks,q1))
                            q0=q1
                            repeat=False
                else:
                    repeat=True
        if re.search('Reciprocal',line,flags=re.IGNORECASE):
            path_section=True
    ticks[len(ticks)-1,3]=1

    num_q=len(ticks[:,0])

    num_labels=num_q
    path=True
    for i in range(num_q):
        if path==False:
            diff=ticks[i,:3]-ticks[i-1,:3]
            labels[i-1]=labels[i-1]+'|'+labels[i]
            labels[i]='000'
        if ticks[i,3]==1:
            path=False
            num_labels=num_labels-1
        else:
            path=True
    while labels.count('000')>0:
        labels.remove('000')
    for i in range(len(labels)):
        if 'Gamma' in labels[i]:
            labels[i]=labels[i].split('Gamma')[0]+'\Gamma'+labels[i].split('Gamma')[1]
        labels[i]='$'+labels[i]+'$'
    return ticks, labels

def grep_ticks_QE(file,filetype=None,silent=True):
    """Greps the K-path from a qe_bands_in or matdyn.in Quantum Espresso files.
    OUTPUT= np.array([K-point1, # of points to next],
                     [K-point2, # of points to next],
                     ...]
    """
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype=filetype.lower()
    KPATH=open(file)
    ticks=np.zeros(0)
    num_q=0
    path_section=False
    for line in KPATH:
        if path_section==True:
            if num_q==0:
                num_q=int(line)
                i=0
            elif i<num_q:
                if len(line.split())!=0:
                    X=float(line.split()[0])
                    Y=float(line.split()[1])
                    Z=float(line.split()[2])
                    points=int(line.split()[3])
                    q1=np.array([X,Y,Z,points])
                    if len(ticks)==0:
                        ticks=q1
                    else:
                        ticks=np.vstack((ticks,q1))
                i=i+1
            else:
                break
        elif re.search('K_POINTS.*crystal_b',line,flags=re.IGNORECASE):
            path_section=True
        elif re.search('/',line) and filetype=='matdyn_in':
            path_section=True
    num_labels=num_q
    path=True
    for i in range(num_q):
        if ticks[i,3]==1:
            path=False
            num_labels=num_labels-1
        else:
            path=True
    if silent == False:
        print("you need to introduce",num_labels+1,"labels")
    return ticks

def grep_ph_grid_points(file,expanded=False,decimals=3):
    """Greps the grid points from a ph.pwo file, it reads the points and the star of those
    points given in the QE ouput and expresses them in reciprocal space lattice vectors. This grid
    can be further expanded to equivalent points line (0,0,0.5) and (0,0,-0.5).

    file='material.ph.pwo'
    """
    vectors=grep_lattice(file,alat=True,filetype='qe_ph_out')
    text=open(file)
    read_text=False
    for line in text:
        if read_text==True and line.split()[0]==str(i):
            split=(line.split())
            point=np.array([float(split[1]),float(split[2]),float(split[3])])
            point=np.matmul(point,vectors.transpose())
            try:
                grid_points=np.vstack([grid_points,point])
            except NameError:
                grid_points=point
            i=i+1
            if i > num_star:
                read_text=False
        elif re.search('Number of q in the star',line):
            num_star=int(line.split()[7])
            i=1
            read_text=True
        elif re.search('In addition there is the',line):
            i=1
            read_text=True
    grid_points=np.around(grid_points,decimals=decimals) #Fix for detecting the grid in the paths
    
    if expanded==True:
        initial_grid=grid_points
        for point in initial_grid:
            expanded_star=__expand_star(point)
            try:
                grid=np.vstack([grid,expanded_star])
            except NameError:
                grid=expanded_star
    else:
        grid=grid_points
    return grid

def __expand_star(q_point):
    """Expands the "star" of each point to equivalent points inside (111) (related by lattice) in the border of the BZ.
    This is usefull when a High Sym point is the (0.5,0,0), this generates (-0.5,0,0), which is not in the star
    since is it the same point.    
    """
    output=[q_point]
    for i in range(3):
        for point in output:
            related1=np.array(point)
            related2=np.array(point)
            related1[i]=point[i]+1
            related2[i]=point[i]-1
            output=np.vstack([output,related1,related2])
    return output

def grep_total_energy(file,meV=False,filetype=None):
    """Greps the total energy (in Ry or meV) from a Quantum Espresso (.pwo) or VASP (OUTCAR)file.
    returns either the energy or a False boolean if the energy was not found"""
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype=filetype.lower()
    lines=open(file,'r')
    energy=False
    if filetype[:2] == 'qe':
        for line in reversed(list(lines)):
            if re.search('!',line):
                l=line.split()
                energy=float(l[4])
                break
    elif filetype == 'outcar':
        for line in reversed(list(lines)):
            if re.search('sigma->',line):
                l=line.split()
                energy=float(l[-1])
                break
        energy=energy/const.Ry2eV
    if meV==True:
        energy=energy*const.Ry2eV*1000
    return energy

# Transformation tools----------------------------------------------------------------

def K_basis(lattice,alat=False):
    """With basis_vec being in rows, it returns the reciprocal basis vectors in rows
    and units of 2pi"""
    if alat == True:
        lattice=lattice/np.linalg.norm(lattice[0])
    K_vec=np.linalg.inv(lattice).transpose() #reciprocal vectors in rows
    return K_vec

def cartesian2cryst(cartesian,cryst_basis,list_of_vec=False):
    """Goes from cartesian units to cryst units. Either vectors or matrices (also list of vectors)
    cartesian: coordinates or matrix in cartesian units
    cryst_basis: crystaline basis written by rows
    return crystal_coord"""
    if len(np.shape(cartesian))==1 or list_of_vec==True:
        crystal_coord=np.matmul(cartesian,np.linalg.inv(cryst_basis))
    elif len(np.shape(cartesian))==2:
        inv=np.linalg.inv(cryst_basis)
        crystal_coord=np.matmul(np.transpose(inv),cartesian)
        crystal_coord=np.matmul(crystal_coord,np.transpose(cryst_basis))
    return crystal_coord

def cryst2cartesian(cryst,cryst_basis,list_of_vec=False):
    """Goes from crystaling units to cartesian units. Either vectors or matrices (also list of vectors)
    cryst: coordinates or  matrix in cryst units
    cryst_basis: crystaline basis written by rows
    return cartesian_coord"""
    if len(np.shape(cryst))==1 or list_of_vec==True:
        cartesian_coord=np.matmul(cryst,cryst_basis)
    elif len(np.shape(cryst))==2:
        cartesian_coord=np.matmul(np.transpose(cryst_basis),cryst)
        inv=np.linalg.inv(cryst_basis)
        cartesian_coord=np.matmul(cartesian_coord,np.transpose(inv))
    return cartesian_coord

def cartesian2spherical(xyz,degrees=False):
    """From cartesian to spherical coord
    mod, theta(z^x), phi(x^y)"""
    ptsnew = np.zeros(3)
    xy = xyz[0]**2 + xyz[1]**2
    ptsnew[0] = np.sqrt(xy + xyz[2]**2)
    ptsnew[1] = np.arctan2(np.sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    #ptsnew[1] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[2] = np.arctan2(xyz[1], xyz[0])
    if degrees==True:
        ptsnew[1:]=ptsnew[1:]*180/np.pi
    return ptsnew

def cryst2spherical(cryst,cryst_basis,degrees=False):
    """From crystal coord to spherical (usefull for SKEAF)
    mod, theta(z^x), phi(x^y)"""
    xyz=cryst2cartesian(cryst,cryst_basis)
    spherical=cartesian2spherical(xyz,degrees)
    return spherical
