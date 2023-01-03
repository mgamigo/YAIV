#PYTHON module for cutoff convergence analysis

import re
import glob
import numpy as np
import matplotlib.pyplot as plt

import yaiv.constants as const
import yaiv.utils as utils

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
    """from the folder where data is stored it reads the scf.pwo files (it autodetects the file extension .pwo, .out, whatever...)
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
     
    where first column is the cutoff, the secondo is the total energy(per atom in meV) and the third is the Fermi Energy...
    """
    data=[]
    Kgrids=glob.glob(folder+"/*")
    Kgrids=[i.split("/")[-1] for i in Kgrids]
    Kgrids=__sort_Kgrids(Kgrids)

    #Grep output extension
    subfolders=glob.glob(folder+"/"+Kgrids[0]+"/*")
    files=glob.glob(subfolders[0]+"/*")
    for i,file in enumerate(files):
        if utils.grep_filetype(file) == 'qe_scf_out':
            extension=file.split('.')[-1]
            break

    for i in range(len(Kgrids)):
        Kgrid=Kgrids[i]
        files=glob.glob(folder+"/"+Kgrid+"/*/*"+extension)
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

def reverse_data(data):
    """From the data obtained with read_data it reverses so that the format is:
    ...
    OUTPUT:
    A list where odd numbers are the cutoffs and even numbers are the corresponding info:
    [[   K1xK2xK3         -1416.5169454     -1.489      time          10GB    0.17]
     [   K1xK2xK3         -1416.51803251    -1.489      time          30Gb     0.22]
     [  K1xK2xK3    energy(meV/atom)  fermi(eV)  time(hours)    RAM    forces(meV/au*atom)]
     [  K1xK2xK3         -1416.52358896    -1.4889     time    ...]
     [  K1xK2xK3         -1416.5240187     -1.4889     time    ...]]
     
    where first column is the total of K points, the secondo is the total energy(per atom in meV) and the third is the Fermi Energy...
    """

    Kgrids=data[0::2]
    cutoffs=[]
    for dataset in data[1::2]:
        for cutoff in dataset[:,0]:
            if cutoff not in cutoffs:
                cutoffs=cutoffs+[cutoff]
    new_data=[]
    for cutoff in cutoffs:
        for i,dataset in enumerate(data[1::2]):
            index=np.where(dataset[:,0]==cutoff)
            if len(index)==1:
                index=index[0][0]
                grid_K=np.prod([int(x) for x in Kgrids[i].split('x')])
                data_row=np.hstack((grid_K,dataset[index,1:]))
                try:
                    data_cutoff=np.vstack((data_cutoff,data_row))
                except NameError:
                    data_cutoff=data_row
            elif len(index)>1:
                print('Something really wrong happend...')
        new_data=new_data+[cutoff]+[data_cutoff]
        del data_cutoff
    return new_data

def energy_vs_cutoff(data,grid=False,savefig=None,axis=None):
    """It plots the energy as a function of cutoff for different k_grids
    data: Either the data, or folder where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)

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


def energy_vs_Kgrid(data,grid=False,savefig=None,axis=None,Kgrids=None):
    """It plots the total energy as a function of K_grid for different cutoffs
    data: Either the data, or folder where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',label=int(data[i]))
    ax.set_ylabel("total energy/atom (meV)")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
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

def fermi_vs_cutoff(data,grid=False,savefig=None,axis=None):
    """It plots the Fermi energy as a function of cutoff for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,2],'.-',label=data[i])
    ax.set_ylabel("Fermi energy (eV)")
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


def fermi_vs_Kgrid(data,grid=False,savefig=None,axis=None,Kgrids=None):
    """It plots the Fermi energy as a function of K_grid for different cutoffs
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,2],'.-',label=int(data[i]))
    ax.set_ylabel("Fermi energy (eV)")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
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

def time_vs_cutoff(data,grid=False,savefig=None,axis=None):
    """It plots the computation time as a function of cutoff for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,3],'.-',label=data[i])
    ax.set_ylabel("time in hours")
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


def time_vs_Kgrid(data,grid=False,savefig=None,axis=None,Kgrids=None):
    """It plots the computation time as a function of K_grid for different cutoffs
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,3],'.-',label=int(data[i]))
    ax.set_ylabel("time in hours")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
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


def RAM_vs_cutoff(data,grid=False,savefig=None,axis=None):
    """It plots the RAM as a function of cutoff for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,4],'.-',label=data[i])
    ax.set_ylabel("RAM (Gb)")
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


def RAM_vs_Kgrid(data,grid=False,savefig=None,axis=None,Kgrids=None):
    """It plots the RAM as a function of K_grid for different cutoffs
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,4],'.-',label=int(data[i]))
    ax.set_ylabel("RAM (Gb)")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
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

def forces_vs_cutoff(data,grid=False,savefig=None,axis=None):
    """It plots the (total force)/(num atoms) as a function of cutoff for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,5],'.-',label=data[i])
    ax.set_ylabel("Total force/atom (meV/au)")
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


def forces_vs_Kgrid(data,grid=False,savefig=None,axis=None,Kgrids=None):
    """It plots the num atomsRAM as a function of K_grid for different cutoffs
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,5],'.-',label=int(data[i]))
    ax.set_ylabel("Total force/atom (meV/au)")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
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


def energy(folder,savefig=None):
    """It plots the total energy as a function of cutoffs and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)
    data_K=reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Total energy')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    energy_vs_cutoff(data,grid=True,axis=ax1)
    energy_vs_Kgrid(data_K,grid=True,axis=ax2,Kgrids=Kgrids)
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()


def fermi(folder,savefig=None):
    """It plots the total fermi energy as a function of cutoffs and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)
    data_K=reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Fermi Energy')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fermi_vs_cutoff(data,grid=True,axis=ax1)
    fermi_vs_Kgrid(data_K,grid=True,axis=ax2,Kgrids=Kgrids)
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()


def time(folder,savefig=None):
    """It plots the total computation time as a function of cutoffs and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)
    data_K=reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Computing time')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    time_vs_cutoff(data,grid=True,axis=ax1)
    time_vs_Kgrid(data_K,grid=True,axis=ax2,Kgrids=Kgrids)
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()

def RAM(folder,savefig=None):
    """It plots the total needed RAM as a function of cutoffs and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)
    data_K=reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('RAM (GB)')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    RAM_vs_cutoff(data,grid=True,axis=ax1)
    RAM_vs_Kgrid(data_K,grid=True,axis=ax2,Kgrids=Kgrids)
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()

def forces(folder,savefig=None):
    """It plots the total (total force)/(num atoms) as a function of cutoffs and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)
    data_K=reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Total force')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    forces_vs_cutoff(data,grid=True,axis=ax1)
    forces_vs_Kgrid(data_K,grid=True,axis=ax2,Kgrids=Kgrids)
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()
