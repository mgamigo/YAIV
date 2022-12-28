#PYTHON module for cutoff convergence analysis

import re
import glob
import numpy as np
import matplotlib.pyplot as plt

import yaiv.constants as const

def __sort_Kgrids(Kgrids):
    """Using the list of Kgrids as input ['4x4x1','3x3x1','10x10x1']
    this returns a sorted list for the grids"""
    num_k=[]
    for grid in Kgrids:
        x,y,z=[int(i) for i in grid.split('x')]
        num_k=num_k+[x*y*z]
    OUT=[Kgrids for _, Kgrids in sorted(zip(num_k, Kgrids))]
    return OUT

def __read_number_of_atoms(file):
    """Read the number of atoms from que Quantum Espresso output"""
    lines=open(file,'r')
    for line in lines:
        if re.search('number of atoms/cell',line):
            l=line.split()
            atoms_num=int(l[4])
    return atoms_num

def read_data(folder):
    """from the folder where data is stored it reads the scf.pwo files
    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    ...
    OUTPUT:
    A list where odd numbers are the Kdrid and even numbers are the corresponding folder info:
    [[   80.         -1416.5169454     -1.489      time          10GB    0.17]
     [   90.         -1416.51803251    -1.489      time          30Gb     0.22]
     [  cutoff(Ry)    energy(meV/atom)  fermi(eV)  time(hours)    RAM    forces(meV/au*atom)]
     [  110.         -1416.52358896    -1.4889     time    ...]
     [  120.         -1416.5240187     -1.4889     time    ...]]
     
    where first column is the cutoff, the secondo is the total energy(per atom in meV) and the third is the Fermi Energy
    """
    data=[]
    Kgrids=glob.glob(folder+"/*")
    Kgrids=[i.split("/")[-1] for i in Kgrids]
    Kgrids=__sort_Kgrids(Kgrids)
    for i in range(len(Kgrids)):
        Kgrid=Kgrids[i]
        files=glob.glob(folder+"/"+Kgrid+"/*/*pwo")
        grid_data=np.zeros([len(files),6])
        atoms_num=__read_number_of_atoms(files[0])
        for j in range(len(files)):
            cutoff=files[j].split("/")[-2]
            forces=0
            file = open (files[j],"r")
            for line in file:
                if re.search("!",line):
                    energy=line.split()[4]
                if  re.search("highest",line):
                    fermi=line.split()[4]
                if re.search("Fermi",line):
                    fermi=line.split()[4]
                if re.search("PWSCF.*WALL",line):
                    time=line.split('CPU')[1].split('WALL')[0]
                    h=0
                    m=0
                    s=0
                    if 'h' in time:
                        h=int(time.split('h')[0])
                        time=time.split('h')[1]
                    if 'm' in time:
                        m=int(time.split('m')[0])
                        time=time.split('m')[1]
                    if 's' in time:
                        s=float(time.split('s')[0])
                    time=(s+60*(m+60*h))/3600
                if re.search("total.*RAM",line):
                    if line.split()[6]=='MB':
                        RAM=str(float(line.split()[5])/1024)
                    else:
                        RAM=line.split()[5]
                if re.search("Total.*force",line):
                    forces=line.split()[3]
            grid_data[j,:]=np.array([cutoff,float(energy)*const.Ry2meV/atoms_num,fermi,time,RAM,float(forces)*const.Ry2meV/atoms_num])
        data=data+[Kgrid]+[grid_data]
    for i in range(0,len(data),2):
        data[i+1]=plot_data=data[i+1][data[i+1][:,0].argsort()] #sort acording to first column (x axis)(cutoff)
    return data

def energy_vs_cutoff(folder,grid=False,savefig=None,axis=None):
    """It plots the energy as a function of cutoff for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',label=data[i])
    ax.set_ylabel("total energy/atom (meV)")
    ax.set_xlabel("cutoff (Ry)")
    ax.legend(prop={'size': 7})
    if grid == True:
        ax.grid()
    if savefig!=None:
        plt.tight_layout()
        plt.savefig(savefig,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def energy_vs_Kgrid(folder,grid=False,savefig=None,axis=None):
    """It plots the total energy as a function of K_grid for different cutoffs
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)

    Kgrids=data[0::2]
    #fig=plt.figure()
    cutoffs=glob.glob(folder+"/*/*")
    cutoffs=[i.split('/')[-1] for i in cutoffs]
    cutoffs=list(set(cutoffs))
    cutoffs.sort(key=int)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for cutoff in cutoffs:
        cutoff_data=[]
        for i in range(len(Kgrids)):
            grid=Kgrids[i].split('x')[0]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+cutoff):
                cutoff_index=np.where(data[2*i+1][:,0]==int(cutoff))[0][0]
                cutoff_data=cutoff_data+[int(grid),data[2*i+1][cutoff_index,1]]
        ax.plot(cutoff_data[0::2],cutoff_data[1::2],'.-',label=cutoff)

    ax.set_ylabel("Total energy/atom (meV)")
    ax.set_xlabel("K grid first index")
    ax.set_xticks([int(i.split('x')[0]) for i in Kgrids])
    ax.legend(prop={'size': 7})
    ax.tight_layout()
    if grid == True:
        ax.grid()
    if savefig!=None:
        plt.tight_layout()
        plt.savefig(savefig,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()
