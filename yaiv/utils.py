#PYTHON module with nice tu have utilitary tools, mostly focussed on text scraping

import numpy as np
import re
from ase import io

# GREPPING utilities***********************************************************************

class file:
    """A class for file scraping, depending on the filetype a different set of attributes will initialize.
    The filetype should be automatically detected, but can be manually introduced:
    QuantumEspresso: qe_scf_in, qe_scf_out, qe_bands_in
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
        if self.filetype in ['qe_scf_out','qe_scf_in','qe_bands_in','outcar','poscar']:
            self.lattice = grep_lattice(file)
        if self.filetype in ['qe_scf_out','outcar']:
            self.electrons = grep_electrons(file,filetype=self.filetype)
        if self.filetype in ['qe_scf_out','outcar']:
            self.fermi = grep_fermi(file,filetype=self.filetype)
        if self.filetype == 'kpath':
            self.path,self.labels = grep_ticks_labels_KPATH(file)
        if self.filetype == 'qe_bands_in':
            self.path = grep_ticks_QE(file)
    def __str__(self):
        return str(self.filetype) + ':\n' + self.file

def grep_filetype(file):
    """Returns the filetype, currently it supports:
    QuantumEspresso: qe_scf_in, qe_scf_out, qe_bands_in
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
        if re.search('calculation.*bands.*',line,re.IGNORECASE):
            filetype='qe_bands_in'
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
        else:
            filetype='data'
    return filetype

def grep_lattice(file):
    """Greps the lattice vectors from a variety of outputs (it uses ase)
    OUTPUT= np.array([vec1,vec2,vec3])
    """
    try:
        data=io.read(file)
        lattice=np.array(data.cell)
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

def grep_ticks_QE(file,silent=True):
    """Greps the K-path from a bands.pwi file.
    OUTPUT= np.array([K-point1, # of points to next],
                     [K-point2, # of points to next],
                     ...]
    """
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
        if re.search('K_POINTS.*crystal_b',line,flags=re.IGNORECASE):
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
