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
