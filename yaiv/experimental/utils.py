#PYTHON module for ploting bands, two main functions:
#p)lot_phonons(file,vectors,ticks,labels,title)
#plot_bands(file,vectors,ticks,labels,title,fermi,window)

import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import os
import subprocess
import yaiv.plot as plot
import yaiv.experimental.cell_analyzer as cell
import yaiv.constants as cons
import spglib as spg

#Utilities*****************************************************************************************

def fermi_surface(file):
    """Just launches fermisurfer for you from the python notebook"""
    cmd='fermisurfer '+file
    subprocess.call(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def adapt_path(vectors1,vectors2,KPATH):
    """Adapts the KPATH in TQC format to another set of coordinates in order to match paths with differente subsets
    vectors1 = original set of vectors (use grep_vectors)
    vectors2 = original set of vectors (use grep_vectors)
    KPATH = file with the KPATH in VASP format
    """
    #reciprocal vectors in columns
    rec1=np.linalg.inv(vectors1)
    rec2=np.linalg.inv(vectors2)
    M=np.matmul(vectors2,rec1)
    
    PATH=open(KPATH,'r')
    QE_path=np.zeros(0)
    q0=np.zeros(0)
    repeat=False
    path_section=False
    for line in PATH:
        if path_section==True:
            if len(line.split())!=0:
                X=float(line.split()[0])
                Y=float(line.split()[1])
                Z=float(line.split()[2])
                q1=np.array([X,Y,Z,100])
                q1[0:3]=np.matmul(M,q1[0:3])
                label=line.split('!')[1]
                if not np.array_equal(q0,q1):
                    if len(QE_path)==0:
                        QE_path=q1
                    else:
                        if repeat==False:
                            if len(QE_path.shape)>1:
                                if QE_path[len(QE_path)-2,3]!=1:
                                    QE_path[len(QE_path)-1,3]=1
                            QE_path=np.vstack((QE_path,q1))
                            q0=q1
                        else:
                            QE_path=np.vstack((QE_path,q1))
                            q0=q1
                            repeat=False
                else:
                    repeat=True
        if re.search('Reciprocal',line,flags=re.IGNORECASE):
            path_section=True
    QE_path[len(QE_path)-1,3]=1

    num_q=len(QE_path[:,0])

    np.savetxt('tmp.path',QE_path,fmt='%1.5f %1.5f %1.5f    %1.0f')

    f = open('QE_path','w')
    f.write(str(num_q)+"\n")
    tmp=open('tmp.path')
    for line in tmp:
        f.write(line)
    f.close()
    os.remove('tmp.path')

# GREPPING utilities***********************************************************************

def integrate_DOS(data,fermi,filetype=None,shift=None,doping=None,force_positive=False):
    """Integrates the density of states for different purposes
    data = file from which to read the DOS or DOS itself as given by grep_DOS.
    fermi = Energy up to which the integration is done.
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               outcar (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
               data file (as generated when running the plot.pdos QE routine)
    force_positive = The DOS may be negative with certain smearings, do you want to force it to be positive?

    This method serves a couple of porpuses and depending on the input gives different output. The integration is done with the trapezoidal method.
    JUST INTEGRATION
        returns the integrated DOS (aka number of electrons)
    Looking to shift the fermi level?
        shift = Float with the desired shift in the Fermi level
        returns the necessary doping per cell (negative being electron doping)
    Looking to achieve certain doping?
        doping = Doping per cell (in electron units) (negative being electron doping)
        returns = Stimated Fermi level for that doping
    """
#    if type(data)==str:
    data=np.loadtxt(data)
    #Force possitive DOS
    if force_positive==True:
        for i,num in enumerate(data[:,1]):
            if num<0:
                data[i,1]=0
                
    #Compute number of electrons up to charge neutrality        
    for i,num in enumerate(data[:,0]):
        if num>fermi:
            limit=i
            break
    num_elec=np.trapz(data[:limit,1],data[:limit,0])
    
    #If you are looking for doping in order to achieve certain shift in the Fermi level
    if shift!=None:
        for i,num in enumerate(data[:,0]):
            if num>fermi+shift:
                limit=i
                break
        doping=num_elec-np.trapz(data[:limit,1],data[:limit,0])
        return doping
    #If you want to predict the chemical potential shift from certain doping
    if doping!=None:
        tota_elec=num_elec-doping
        low=0
        for limit,dum in enumerate(data[:,0]):
            elec=np.trapz(data[:limit,1],data[:limit,0])
            high=data[limit,0]
            if elec>=tota_elec:
                break
            else:
                low=high
        return low+(high-low)/2
    return num_elec
