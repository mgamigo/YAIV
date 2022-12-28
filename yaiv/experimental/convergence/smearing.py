#PYTHON module for convergence analysis

import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
import getpass
import sys
user=getpass.getuser()
sys.path.append('/home/'+user+'/Software/Work_scripts/python_modules')
import constants as cons
Ry2cm=cons.Ry2cm
Ry2eV=cons.Ry2eV
Ry2meV=Ry2eV*1000


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
    And subfolders with the smearing number
    ...
    OUTPUT:
    A list where odd numbers are the Kdrid and even numbers are the corresponding folder info:
    [[  0.02         -1416.5169454     -1.489    time ]
     [  0.018         -1416.51803251    -1.489    time ]
     [  0.016         -1416.52138472    -1.4889   time ]
     [  0.014         -1416.52358896    -1.4889   time ]
     [  0.012         -1416.5240187     -1.4889    time]]
     
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
            smearing=files[j].split("/")[-2]
            forces=0
            file = open (files[j],"r")
            for line in file:
                if re.search("!",line):
                    energy=line.split()[4]
                if  re.search("highest",line):
                    fermi=line.split()[4]
                if re.search("the Fermi",line):
                    fermi=line.split()[4]
                if re.search("PWSCF.*WALL",line):
                    time=line.split('CPU')[1].split('WALL')[0]
                    d=0
                    h=0
                    m=0
                    s=0
                    if 'd' in time:
                        d=int(time.split('d')[0])
                        time=time.split('d')[1]
                    if 'h' in time:
                        h=int(time.split('h')[0])
                        time=time.split('h')[1]
                    if 'm' in time:
                        m=int(time.split('m')[0])
                        time=time.split('m')[1]
                    if 's' in time:
                        s=float(time.split('s')[0])
                    time=(s+60*(m+60*(h+24*d)))/3600
                if re.search("total.*RAM",line):
                    if line.split()[6]=='MB':
                        RAM=str(float(line.split()[5])/1024)
                    else:
                        RAM=line.split()[5]
                if re.search("Total.*force",line):
                    forces=line.split()[3]
            grid_data[j,:]=np.array([smearing,float(energy)*Ry2meV/atoms_num,fermi,time,RAM,float(forces)*Ry2meV/atoms_num])
        data=data+[Kgrid]+[grid_data]
    for i in range(0,len(data),2):
        data[i+1]=plot_data=data[i+1][data[i+1][:,0].argsort()] #sort acording to first column (x axis)(smearing)
    return data

def energy_vs_smearing(folder,savefig=None):
    """It plots the energy as a function of smearing for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    fig=plt.figure()
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        plt.plot(plot_data[:,0],plot_data[:,1],'.-',label=data[i])
    plt.ylabel("Total energy/atom (meV)")
    plt.xlabel("Smearing")
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def energy_vs_Kgrid(folder,savefig=None):
    """It plots the total energy as a function of K_grid for different smearings
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    #fig=plt.figure()
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    fig=plt.figure()
    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,1]]
        plt.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    plt.ylabel("Total energy/atom (meV)")
    plt.xlabel("K point number")
    plt.xticks(grids_K,labels=Kgrids,rotation='vertical')
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def fermi_vs_smearing(folder,savefig=None):
    """It plots the Fermi energy as a function of smearing for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    i=1
    plot_data=np.zeros([len(Kgrids),2])
    for j in range(0,len(Kgrids)):
        plot_data[j,0]=int(data[2*j].split('x')[0])
        plot_data[j,1]=data[(2*j)+1][i,2]
    label=str(data[1][i,0])

    fig=plt.figure()
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        plt.plot(plot_data[:,0],plot_data[:,2],'.-',label=data[i])
    plt.ylabel("Fermi Energy (eV)")
    plt.xlabel("Smearing")
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def fermi_vs_Kgrid(folder,savefig=None):
    """It plots the Fermi energy as a function of K_grid for different smearings
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    fig=plt.figure()
    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,2]]
        plt.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    plt.ylabel("Fermi Energy (eV)")
    plt.xlabel("K point number")
    plt.xticks(grids_K,labels=Kgrids,rotation='vertical')
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()


def time_vs_smearing(folder,savefig=None):
    """It plots the Fermi energy as a function of smearing for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    i=1
    plot_data=np.zeros([len(Kgrids),2])
    for j in range(0,len(Kgrids)):
        plot_data[j,0]=int(data[2*j].split('x')[0])
        plot_data[j,1]=data[(2*j)+1][i,2]
    label=str(data[1][i,0])

    fig=plt.figure()
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        plt.plot(plot_data[:,0],plot_data[:,3],'.-',label=data[i])
    plt.ylabel("time in hours")
    plt.xlabel("Smearing")
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def time_vs_Kgrid(folder,savefig=None):
    """It plots the Fermi energy as a function of K_grid for different smearings
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    fig=plt.figure()
    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,3]]
        plt.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    plt.ylabel("time in hours")
    plt.xlabel("K point number")
    plt.xticks(grids_K,labels=Kgrids,rotation='vertical')
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()



def RAM_vs_smearing(folder,savefig=None):
    """It plots the RAM as a function of smearing for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    i=1
    plot_data=np.zeros([len(Kgrids),2])
    for j in range(0,len(Kgrids)):
        plot_data[j,0]=int(data[2*j].split('x')[0])
        plot_data[j,1]=data[(2*j)+1][i,2]
    label=str(data[1][i,0])

    fig=plt.figure()
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        plt.plot(plot_data[:,0],plot_data[:,4],'.-',label=data[i])
    plt.ylabel("RAM (GB)")
    plt.xlabel("Smearing")
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def RAM_vs_Kgrid(folder,savefig=None):
    """It plots the RAM as a function of K_grid for different smearings
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    fig=plt.figure()
    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,4]]
        plt.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    plt.ylabel("RAM (GB)")
    plt.xlabel("K point number")
    plt.xticks(grids_K,labels=Kgrids,rotation='vertical')
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def forces_vs_smearing(folder,savefig=None):
    """It plots the Total force as a function of smearing for different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    i=1
    plot_data=np.zeros([len(Kgrids),2])
    for j in range(0,len(Kgrids)):
        plot_data[j,0]=int(data[2*j].split('x')[0])
        plot_data[j,1]=data[(2*j)+1][i,2]
    label=str(data[1][i,0])

    fig=plt.figure()
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        plt.plot(plot_data[:,0],plot_data[:,5],'.-',label=data[i])
    plt.ylabel("Total force/atom (meV/au)")
    plt.xlabel("Smearing")
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()

def forces_vs_Kgrid(folder,savefig=None):
    """It plots the Total force as a function of K_grid for different smearings
    folder: where data is stored it reads the scf.pwo files and plots
    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    Kgrids=data[0::2]
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    fig=plt.figure()
    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,5]]
        plt.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    plt.ylabel("Total force/atom (meV/au)")
    plt.xlabel("K point number")
    plt.xticks(grids_K,labels=Kgrids,rotation='vertical')
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()


def energy(folder,savefig=None):
    """It plots the total energy as a function of smearings and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Total energy')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax1.plot(plot_data[:,0],plot_data[:,1],'.-',label=data[i])
    ax1.set_ylabel("Total energy/atom (meV)")
    ax1.set_xlabel("Smearing")
    ax1.legend(prop={'size': 7})
    
    
    Kgrids=data[0::2]
    #fig=plt.figure()
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,1]]
        ax2.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    ax2.set_ylabel("Total energy/atom (meV)")
    ax2.set_xlabel("K grid")
#    ax2.set_xticks([int(i.split('x')[0]) for i in Kgrids])
    ax2.set_xticks(grids_K)
    ax2.set_xticklabels(Kgrids,rotation='vertical')
    ax2.legend(prop={'size': 7})
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()
    
def fermi(folder,savefig=None):
    """It plots the total fermi energy as a function of smearings and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Fermi Energy')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax1.plot(plot_data[:,0],plot_data[:,2],'.-',label=data[i])
    ax1.set_ylabel("Fermi Energy (eV)")
    ax1.set_xlabel("Smearing")
    ax1.legend(prop={'size': 7})
    
    
    Kgrids=data[0::2]
    #fig=plt.figure()
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,2]]
        ax2.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    ax2.set_ylabel("Fermi Energy (eV)")
    ax2.set_xlabel("K grid")
    ax2.set_xticks(grids_K)
    ax2.set_xticklabels(Kgrids,rotation='vertical')
    ax2.legend(prop={'size': 7})
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()
    
def time(folder,savefig=None):
    """It plots the total computation time as a function of smearings and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Computing time')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax1.plot(plot_data[:,0],plot_data[:,3],'.-',label=data[i])
    ax1.set_ylabel("time in hours")
    ax1.set_xlabel("Smearing")
    ax1.legend(prop={'size': 7})
    
    
    Kgrids=data[0::2]
    #fig=plt.figure()
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,3]]
        ax2.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    ax2.set_ylabel("time in hours")
    ax2.set_xlabel("K grid")
    ax2.set_xticks(grids_K)
    ax2.set_xticklabels(Kgrids,rotation='vertical')
    ax2.legend(prop={'size': 7})
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()
    
def RAM(folder,savefig=None):
    """It plots the total needed RAM as a function of smearings and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('RAM (GB)')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax1.plot(plot_data[:,0],plot_data[:,4],'.-',label=data[i])
    ax1.set_ylabel("RAM (GB)")
    ax1.set_xlabel("Smearing")
    ax1.legend(prop={'size': 7})
    
    
    Kgrids=data[0::2]
    #fig=plt.figure()
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,4]]
        ax2.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    ax2.set_ylabel("RAM (GB)")
    ax2.set_xlabel("K grid")
    ax2.set_xticks(grids_K)
    ax2.set_xticklabels(Kgrids,rotation='vertical')
    ax2.legend(prop={'size': 7})
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()


def forces(folder,savefig=None):
    """It plots the total Total force as a function of smearings and grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    data=read_data(folder)
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Total force')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for i in range(0,len(data),2):
        plot_data=data[i+1]
        ax1.plot(plot_data[:,0],plot_data[:,5],'.-',label=data[i])
    ax1.set_ylabel("Total force/atom (meV/au)")
    ax1.set_xlabel("Smearing")
    ax1.legend(prop={'size': 7})
    
    
    Kgrids=data[0::2]
    #fig=plt.figure()
    smearings=glob.glob(folder+"/*/*")
    smearings=[i.split('/')[-1] for i in smearings]
    smearings=list(set(smearings))
    smearings.sort(key=float)

    grids_K=[]
    for smearing in smearings:
        smearing_data=[]
        for i in range(len(Kgrids)):
            grid=[int(x) for x in Kgrids[i].split('x')]
            num_k=np.prod(grid)
            if num_k not in grids_K:
                grids_K=grids_K+[num_k]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+smearing):
                smearing_index=np.where(data[2*i+1][:,0]==float(smearing))[0][0]
                smearing_data=smearing_data+[num_k,data[2*i+1][smearing_index,5]]
        ax2.plot(smearing_data[0::2],smearing_data[1::2],'.-',label=smearing)

    ax2.set_ylabel("Total force/atom (meV/au)")
    ax2.set_xlabel("K grid")
    ax2.set_xticks(grids_K)
    ax2.set_xticklabels(Kgrids,rotation='vertical')
    ax2.legend(prop={'size': 7})
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()
