#PYTHON module with nice tu have utilitary tools, mostly focussed on text scraping

import numpy as np
import re
from ase import io

import yaiv.constants as const
import yaiv.plot as plot

#& GREPPING utilities----------------------------------------------------------------

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
            self.fermi = grep_fermi(file,filetype=self.filetype,silent=True)
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
    def grep_stress_tensor(self,kbar=False):
        """Returns the total stress tensor in (Ry/bohr**3) or (kbar) of scf.pwo file"""
        out=grep_stress_tensor(self.file,kbar=kbar)
        self.stress=out
        return out
    def grep_kpoints_energies(self):
        """ Greps the Kpoints, energies and weights...
        For more info check grep_kpoints_energies function"""
        out=grep_kpoints_energies(self.file,filetype=self.filetype,vectors=self.grep_lattice())
        self.kpoints_energies=out[0]
        self.kpoints_weights=out[1]
        return out
    def grep_DOS(self,fermi='auto',smearing=0.02,window=None,steps=500,precision=3):
        """
        Grep the density of states from a scf or nscf file. 
        For more info check grep_DOS function
        """
        if fermi == 'auto':
            fermi=grep_fermi(self.file,silent=True)
            if fermi==None:
                fermi=0
        out=grep_DOS(self.file,fermi=fermi,smearing=smearing,window=window,
                     steps=steps,precision=precision)
        self.DOS=out
        return out
    def grep_number_of_bands(self,window=None,fermi=None,filetype=None,silent=True):
        """
        Counts the number of bands in an energy window
        For more info check grep_number_of_bands function
        """
        if fermi==None:
            fermi=self.fermi
        out=grep_number_of_bands(self.file,window,fermi,self.filetype,silent)
        return out
    def grep_frequencies(self,return_star=True,filetype=None):
        """
        Greps the frequencies (in cm-1)  and q-points (QE alat units) from a qe.ph.out file.
        For more info check grep_frequencies function
        """
        out=grep_frequencies(self.file,return_star,self.filetype)
        self.frequencies=out[1]
        self.frequencies_points=out[0]
        return out
    def grep_electron_phonon_nesting(self,return_star=True,filetype=None):
        """
        Greps the nesting, frequencies (in cm-1),lamdas (e-ph coupling), gamma-linewidths (GHz) and q-points (QE alat units) from a qe.ph.out file
        For more info check grep_electron_phonon_nesting function
        """
        out=grep_electron_phonon_nesting(self.file,return_star,self.filetype)
        self.frequencies_points=out[0]
        self.nestings=out[1]
        self.frequencies=out[2]
        self.lambdas=out[3]
        self.gammas=out[4]
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
    """Greps the lattice vectors (in Angstroms) from a variety of outputs (it uses ase)

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

def grep_fermi(file,filetype=None,silent=False):
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
                    if silent==False:
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

def grep_stress_tensor(file,kbar=False):
    """Greps the total stress tensor in (Ry/bohr**3) or (kbar) of scf.pwo file
    returns either the stress tensor or a False boolean if the pressure was not found"""
    lines=open(file,'r')
    pressure=False
    stress=None
    for line in lines:
        if pressure==True:
            l=line.split()
            l=[float(item) for item in l]
            vec=np.array(l[:3])
            try:
                stress=np.vstack([stress,vec])
                if len(stress)==3:
                    pressure=False
            except NameError:
                stress=vec
        if re.search('total.*stress',line):
            pressure=True
            del stress
    if kbar==True:
        stress=stress*(const.Ry2jul/(const.bohr2metre**3))*const.pas2bar/1000
    return stress

def grep_kpoints_energies(file,filetype=None,vectors=np.array(None)):
    """Process the kpoints, energies and weights for different file kinds.
    returns energies, weights
    
    1- Energies and Kpoints are given as:
    The first three numbers are the K-point coordinate in 2/pi*a units
    The rest of the numbers are the energies for that K point.

    2- The weights are given in a separate numpy array.

    file = File with the bands
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo, relax.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    """
    if filetype == None:
        filetype = grep_filetype(file)

    weights = []
    read_weights=True
    RELAX_calc,RELAXED=False,False
    if filetype[:2]=="qe":
        file=open(file,'r')
        lines=file.readlines()
        for i,line in enumerate(lines):
            #Grep number of bands
            if re.search('number of Kohn-Sham',line):
                num_bands=int(line.split('=')[1])
            if re.search('number of k points',line):
                num_points=int(line.split()[4])
            if re.search('wk =',line) and read_weights==True:
                w=float(line.split()[-1])
                weights = weights + [w]
                if len(weights) == num_points:
                    read_weights=False
                    weights = np.array(weights)
            if re.search('End of .* calculation',line):
                if RELAX_calc == False:
                    results_line=i+1
                    break
                if RELAX_calc==True and RELAXED==True:
                    results_line=i+1
                    break
            if re.search('force convergence', line):
                RELAX_calc=True
            if re.search('Final scf calculation at the relaxed', line):
                RELAXED=True
        data=np.zeros([num_points,num_bands+3])
        data_lines=lines[results_line:]
        i,j=-1,1
        coord0=np.zeros(3)
        read_energies=False
        for line in data_lines:
            if re.search('Writing.*output',line):    #Reading is completed
                break
            elif re.search('bands \(ev\)',line):    #New k_point
                if '-' in line:
                    l=plot.__insert_space_before_minus(line)
                    l=l.split()
                else:
                    l=line.split()
                coord=np.array(l[2:5]).astype(float)  # Already in reciprocal cartesian coord (not like VASP)
                i=i+1
                j=3
                data[i,0:3]=coord
                read_energies=True
            elif re.search('occupation numbers',line):          #Stop reading energies when occupations
                read_energies=False
            elif read_energies==True:                           #Load energies
                line=plot.__insert_space_before_minus(line)
                l=line.split()
                energies=np.array(l).astype(float)
                data[i,j:j+len(energies)]=energies
                j=j+len(energies)
       
    elif filetype=='eigenval':
        file=open(file,'r')
        lines=file.readlines()
        num_points=int(lines[5].split()[1])
        num_bands=int(lines[5].split()[2])
        data_lines=lines[7:]

        data=np.zeros([num_points,num_bands+3])
        if vectors.all()!=None:                      #If there is no cell in the input
            K_vec=np.linalg.inv(vectors).transpose() #reciprocal vectors in columns
        else:
            K_vec=np.identity(3)   #If there is no cell it gives the out in crystaline units
        for i,num in enumerate(range(0,len(data_lines),num_bands+2)):    #load the x position
            line=data_lines[num]
            line=line.split()
            point=np.array(line).astype(float)[0:3]
            coord=np.matmul(point,K_vec)
            data[i,:3]=coord
            w=float(line[-1])
            weights = weights + [w]
        for band in range(1,num_bands+1):                             #load the bands
            i=0
            for num in range(band,len(data_lines),num_bands+2):
                line=data_lines[num]
                line=line.split()
                data[i,band+2]=line[1]
                i=i+1
        weights = np.array(weights)
    elif filetype=="outcar" or filetype=='vasp':
        read_weights=False
        read_energies=False
        file=open(file,'r')
        lines=file.readlines()
        for i,line in enumerate(lines):
            if re.search('k-points in reciprocal lattice',line):
                read_weights=False
            if read_weights==True:
                l=line.split()
                if l!=[]:
                    k=np.array([float(x) for x in l[0:3]])
                    try:
                        K=np.vstack((K,k))
                    except NameError:
                        K=k
                    w=float(l[-1])
                    weights=weights+[w]
            if re.search('2pi/SCALE',line):
                read_weights=True
            if read_energies==True:
                l=line.split()
                if l!=[]:
                    e=float(line.split()[1])
                    E=E+[e]
                else:
                    E=np.array(E)
                    try:
                        energies=np.vstack((energies,E))
                    except NameError:
                        energies=E
                    read_energies=False
            if re.search('band No.',line):
                read_energies=True
                E=[]
        weights=np.array(weights)
        data=np.hstack((K,energies))
    else:
        print("Not implemented for", filetype)
        data,weights = None, None
    return data, weights


def grep_DOS(file,fermi=0,smearing=0.02,window=None,steps=500,precision=3,filetype=None):
    """
    Grep the density of states from a scf or nscf file. 

    => returns energies, DOS
    
    file = File from which to extract the DOS
    fermi = Fermi level to shift accordingly
    smearing = Smearing of your normal distribution around each energy
    window = energy window in which to compute the DOS
    steps = Number of values for which to compute the DOS
    precision = Truncation of your normal distrib (truncated from precision*smearing)
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
    """
    if filetype==None:
        filetype = grep_filetype(file)
    data,weights=grep_kpoints_energies(file,filetype=filetype)
    data=data[:,3:]  #Remove the Kpoint info
    n_bands=len(data[0])
    for w in weights:
        try:
            W=np.vstack((W,np.ones(n_bands)*w))
        except NameError:
            W=np.ones(n_bands)*w
    s=np.shape(W)[0]*np.shape(W)[1]
    #SORT Energies and Weights
    D=data.reshape((s))
    W=W.reshape((s))
    sort=np.argsort(D)
    D=D[sort]-fermi
    W=W[sort]
    #Compute DOS
    if window==None:
        energies=np.linspace(D[0],D[-1],steps)
    elif type(window) is int or type(window) is float:
        energies=np.linspace(-window,window,steps)
    else:
        energies=np.linspace(window[0],window[1],steps)
    DOS=[]
    for E in energies:
        dos=0
        m=np.argmax(D>=E-precision*smearing)
        M=np.argmin(D<=E+precision*smearing)
        for i,e in enumerate(D[m:M]):
            dos=dos+normal_dist(e,E,smearing)*W[i+m]
        DOS=DOS+[dos]
    return energies,DOS

def grep_number_of_bands(file,window=None,fermi=None,filetype=None,silent=True):
    """Counts the number of bands in an energy window for all file types supported by grep_kpoints_energies and .gnu files from QE. (It counts them in the first k-point)

    file = File from in which you want to count the bands.
    window = Window of energies where you want to count the number of bands
    fermi = Fermi energy for applying such shift to energies
    silent = No text output
    filetype =Should be detected automatically, but it supports all file types supported by grep_kpoints_energies and .gnu files from QE
    
    return bands
    """
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype=filetype.lower()
    if filetype=='data':
        data=np.loadtxt(fname=file)
        data=data[:,:2]         #select the first two columns to process (for wannier tools)
        rows=1
        position=data[rows,0]
        while position!=0:     #counter will tell me how many points are in the x axis, number of rows
            rows=rows+1
            position=data[rows,0]
        columns=int(2*data.shape[0]/rows)
        data=np.reshape(data,(rows,columns),order='F')
        final_columns=int(columns/2-1)
        data=np.delete(data,np.s_[0:final_columns],1)
    else:
        data=grep_kpoints_energies(file,filetype=filetype)[0][:,2:]

    if fermi==None:
        fermi=grep_fermi(file)
        if fermi==None:
            fermi=0

    #The actual calculation
    if window==None:
        bands=data.shape[1]-1
        if silent==False:
            print("the total number of bands is",bands)
    else:
        bands=0
        first_q=data[0,1:]-fermi
        for item in first_q:
            if item>=window[0] and item<=window[1]:
                bands=bands+1
        if silent==False:
            print("The number of bands between",str(window[0])+"eV and",str(window[1])+"eV is",bands)
    return bands

def grep_frequencies(file,return_star=True,filetype=None):
    """
    Greps the frequencies (in cm-1)  and q-points (QE alat units) from a qe.ph.out file
    file = File to read from
    return_star = Boolean controling wether to return only que q point, or the whole star (if possible).
    The filetype should be detected automatically, but it supports:
    qe_ph_out

    return POINTS, FREQS
    """
    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype = filetype.lower()
    READ,READING,STAR=False,False,False
    POINTS=[]
    if filetype=='qe_ph_out':
        lines=open(file,'r')
        freqs=[]
        for line in lines:
            if re.search('Diagonalizing',line):
                READ=True
            if STAR==True and READ==False:
                point=np.array([float(x) for x in line.split()[1:]])
                try:
                    if len(point)==3:
                        star=np.vstack((star,point))
                except NameError:
                    star=point
            if re.search('List of q in the star',line) and return_star==True:
                STAR=True
            if READ==True and re.search('q = \(',line):
                if STAR==False:
                    star=np.array([float(x) for x in line.split()[3:6]])
                STAR=False
            if READ==True and re.search('freq',line):
                READING=True
                freqs=freqs+[float(line.split()[-2])]
            if READ==True and READING==True and not re.search('freq',line):
                READ,READING=False,False
                POINTS=POINTS+[star]
                try:
                    FREQS=np.vstack((FREQS,freqs))
                except NameError:
                    FREQS=np.array(freqs)
                del star
                freqs=[]
    else:
        print('FILE NOT SOPPORTED')
    return POINTS, FREQS


def grep_electron_phonon_nesting(file,return_star=True,filetype=None):
    """
    Greps the nesting, frequencies (in cm-1),lamdas (e-ph coupling), gamma-linewidths (GHz) and q-points (QE alat units) from a qe.ph.out file
    file = File to read from
    return_star = Boolean controling wether to return only que q point, or the whole star (if possible).
    The filetype should be detected automatically, but it supports:
    qe_ph_out

    return POINTS, NESTING, FREQS, LAMBDAS, GAMMAS
    """

    if filetype == None:
        filetype = grep_filetype(file)
    else:
        filetype = filetype.lower()
    READ,READING,STAR=False,False,False
    POINTS,NESTING=[],[]
    if filetype=='qe_ph_out':
        lines=open(file,'r')
        freqs,lambdas,gammas=[],[],[]
        for line in lines:
            if re.search('Diagonalizing',line):
                READ=True
            if STAR==True and READ==False:
                point=np.array([float(x) for x in line.split()[1:]])
                try:
                    if len(point)==3:
                        star=np.vstack((star,point))
                except NameError:
                    star=point
            if re.search('List of q in the star',line) and return_star==True:
                STAR=True
            if READ==True and re.search('q = \(',line):
                if STAR==False:
                    star=np.array([float(x) for x in line.split()[3:6]])
                STAR=False
            if READ==True and re.search('freq',line):
                freqs=freqs+[float(line.split()[-2])]
            if READ==True and re.search('double delta at Ef',line):
                nest=float(line.split()[-1])
            if READ==True and re.search('lambda',line):
                READING=True
                lam=float(line.split('=')[1].split()[0])
                lambdas=lambdas+[lam]
                gam=float(line.split('=')[2].split()[0])
                gammas=gammas+[gam]
            if READ==True and READING==True and not re.search('lambda',line):
                READ,READING=False,False
                POINTS=POINTS+[star]
                NESTING=NESTING+[nest]
                try:
                    FREQS=np.vstack((FREQS,freqs))
                    LAMBDAS=np.vstack((LAMBDAS,lambdas))
                    GAMMAS=np.vstack((GAMMAS,gammas))
                except NameError:
                    FREQS=np.array(freqs)
                    LAMBDAS=np.array(lambdas)
                    GAMMAS=np.array(gammas)
                del star
                freqs,lambdas,gammas=[],[],[]
    else:
        print('FILE NOT SOPPORTED')
    return POINTS, NESTING, FREQS, LAMBDAS, GAMMAS

#& Transformation tools----------------------------------------------------------------

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

def spherical2cartesian(coord,degrees=False):
    """From spherical to cartesian coord
    coord = mod, theta(z^x), phi(x^y)"""
    if degrees==True:
        coord[1:]=coord[1:]*2*np.pi/360
    r,t,p=coord
    x = r*np.sin(t)*np.cos(p)
    y = r*np.sin(t)*np.sin(p)
    z = r*np.cos(t)
    return np.array([x,y,z])

def cryst2spherical(cryst,cryst_basis,degrees=False):
    """From crystal coord to spherical (usefull for SKEAF)
    mod, theta(z^x), phi(x^y)"""
    xyz=cryst2cartesian(cryst,cryst_basis)
    spherical=cartesian2spherical(xyz,degrees)
    return spherical

def spherical2cryst(coord,cryst_basis,degrees=False):
    """From spherical coord to crystal (usefull for SKEAF)
    coord = mod, theta(z^x), phi(x^y)"""
    xyz=spherical2cartesian(coord,degrees=degrees)
    cryst=cartesian2cryst(xyz,cryst_basis)
    return cryst


#& Usefull functions----------------------------------------------------------------

def normal_dist(x , mean , sd,A=1):
    """Just a regular normal (gaussian) distribution generator. It integrates to one.
    A = An amplitud term, if A=1 it integrates to unity
    """
    prob_density = A/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mean)/sd)**2)
    return prob_density

def lorentzian_dist(x , center , hwhm, A=1):
    """Just a regular Lorentzian distribution generator.
    center = Point in which the lorentzian is centered
    hwhm = The half-widh half-maximum
    A = An amplitud term, if A=1 it integrates to unity

    """
    OUT=A*(1/np.pi)*hwhm/((x-center)**2+hwhm**2)
    return OUT
