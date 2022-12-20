#PYTHON module with nice tu have utilitary tools, mostly focussed on text scraping

import numpy as np
import re

# GREPPING utilities***********************************************************************

def grep_vectors(file,filetype='qe'):
    """Greps the real vectors from a scf.pwo, bands.pwo (in the alat units) or file VASP OUTCAR 
    (it may work with other output files of QE)
    OUTPUT= np.array([vec1,vec2,vec3])
    """
    filetype=filetype.lower()
    count=0
    lattice_lines=False
    lines=open(file,'r')

    if filetype=='qe':
        for line in lines:
            if lattice_lines==True:
                X=float(line.split()[3])
                Y=float(line.split()[4])
                Z=float(line.split()[5])
                vec=np.array([X,Y,Z])
                if count==0:
                    vectors=vec
                else:
                    vectors=np.vstack((vectors,vec))            
                count=count+1
                if count>=3:
                    break
            if re.search('crystal axes',line,flags=re.IGNORECASE):
                lattice_lines=True
    elif filetype=='vasp':
        for line in lines:
            if re.search('direct lattice vectors',line):
                lattice_lines=True
            elif count>=3:
                break
            elif lattice_lines==True:
                count=count+1
                if count>0:
                    l=line.split()
                    X=float(l[0])
                    Y=float(l[1])
                    Z=float(l[2])
                    vec=np.array([X,Y,Z])
                    try:
                        vectors=np.vstack([vectors,vec])
                    except NameError:
                        vectors=vec
        norm=np.linalg.norm(vectors[0])
        vectors=vectors/norm
    else:
        print('FILETYPE NOT AVAILABLE')
        print('could not grep vectors')
    return vectors

def grep_ticks_labels_KPATH(file):
    """Greps ticks and labels of the ticks from a KPATH file of VASP.
    It expects the file to have the structure:
    0 0 0 !GM
    0 0.5 0 !X
    0 0.5 0 !X
    0.5 0.5 0 !

    It generates the correct ticks to plot with my scripts (even with splitted paths)

    return ticks, labels
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
#    print("you need to introduce",num_labels+1,"labels")
    while labels.count('000')>0:
        labels.remove('000')
    for i in range(len(labels)):
        if 'Gamma' in labels[i]:
            labels[i]=labels[i].split('Gamma')[0]+'\Gamma'+labels[i].split('Gamma')[1]
        labels[i]='$'+labels[i]+'$'
    return ticks, labels

def grep_ticks_QE(file):
    """Greps the path and generates ticks from a bands.pwi file.
    It takes into account when the distance between two points is 1 and therefore there is a splitted path for the bandstructure. It also informs for the number of labels needed.

    (it may work with other output files of QE, easy to addapt to matdyn)

    OUTPUT= np.array([tick1,tick2,tick3...])
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
    print("you need to introduce",num_labels+1,"labels")
    return ticks

def grep_fermi(file,filetype='qe'):
    """Greps the Fermi Energy from a scf.pwo, nscf.pwo ... or OUTCAR (VASP) file.
   returns the Fermi energy in eV
    """
    filetype=filetype.lower()
    if filetype=='qe':
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
    elif filetype=='vasp':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('E-fermi',line):
                E_f=float(line.split()[2])
    return E_f

def grep_electrons(file,filetype='qe'):
    """Greps the number of electrons from a scf.pwo or OUTCAR file.
   return num_elec
    """
    num_elec=None
    filetype=filetype.lower()
    if filetype=='qe':
        scf_out=open(file,'r')
        for line in scf_out:
            if re.search('number of electrons',line):
                num_elec=int(float(line.split()[4]))
    elif filetype=='vasp':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('NELECT',line):
                num_elec=int(float(line.split()[2]))
    return num_elec

def __grep_fermi_and_electrons(file,filetype='qe'):
    """Greps the Fermi Energy and number of electrons from a scf.pwo or OUTCAR file.
   returns the Fermi energy in eV
   (The advantage is we just read the file once)
    """
    num_elec=None
    filetype=filetype.lower()
    if filetype=='qe':
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
            if re.search('number of electrons',line):
                num_elec=int(float(line.split()[4]))
    elif filetype=='vasp':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('E-fermi',line):
                E_f=float(line.split()[2])
            if re.search('NELECT',line):
                num_elec=int(float(line.split()[2]))
    return E_f, num_elec
