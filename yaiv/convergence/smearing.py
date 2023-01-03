#PYTHON module for Kgrid/smearing convergence analysis

import re
import glob
import numpy as np
import matplotlib.pyplot as plt

import yaiv.constants as const
import yaiv.utils as utils
import yaiv.convergence.cutoff as conv

def read_data(folder):
    """from the folder where data is stored it reads the scf.pwo files (it autodetects the file extension .pwo, .out, whatever...)
    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    ...
    OUTPUT:
    A list where odd numbers are the Kdrid and even numbers are the corresponding folder info:
     [[   0.02.         -1416.5169454     -1.489      time          10GB    0.17]
     [   0.018         -1416.51803251    -1.489      time          30Gb     0.22]
     [  smearing(Ry)    energy(meV/atom)  fermi(eV)  time(hours)    RAM    forces(meV/au*atom)]
     [  0.014         -1416.52358896    -1.4889     time    ...]
     [  0.016         -1416.5240187     -1.4889     time    ...]]
    
    where first column is the cutoff, the secondo is the total energy(per atom in meV) and the third is the Fermi Energy
    """
    data=[]
    Kgrids=glob.glob(folder+"/*")
    Kgrids=[i.split("/")[-1] for i in Kgrids]
    Kgrids=conv.__sort_Kgrids(Kgrids)

    #Grep output extension
    subfolders=glob.glob(folder+"/"+Kgrids[0]+"/*")
    files=glob.glob(subfolders[0]+"/*")
    for i,file in enumerate(files):
        if utils.grep_filetype(file) == 'qe_scf_out':
            extension=file.split('.')[-1]
            break

    for i in range(len(Kgrids)):
        Kgrid=Kgrids[i]
        files=glob.glob(folder+"/"+Kgrid+"/*/*pwo")
        grid_data=np.zeros([len(files),6])
        atoms_num=conv.__read_number_of_atoms(files[0])
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
            grid_data[j,:]=np.array([smearing,float(energy)*const.Ry2meV/atoms_num,fermi,time,RAM,float(forces)*const.Ry2meV/atoms_num])
        data=data+[Kgrid]+[grid_data]
    for i in range(0,len(data),2):
        data[i+1]=plot_data=data[i+1][data[i+1][:,0].argsort()] #sort acording to first column (x axis)(smearing)
    return data

def energy_vs_smearing(data,grid=False,temp=False,savefig=None,axis=None):
    """It plots the energy as a function of smearing for different k_grids
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
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})
    if temp==True:
        secax=ax.secondary_xaxis('top',functions=(const.smear2temp,const.temp2smear))
        secax.set_xlabel('T [K]')
    if grid == True:
        ax.grid()
    if savefig!=None:
        plt.tight_layout()
        plt.savefig(savefig,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def energy_vs_Kgrid(data,grid=False,temp=False,savefig=None,axis=None,Kgrids=None):
    """It plots the total energy as a function of K_grid for different smearings
    data: Either the data, or folder where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=conv.reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    labels=data[0::2]
    if temp==True:
        for i,label in enumerate(labels):
            labels[i]=str(label)+" ("+str(int(const.smear2temp(label)))+"K)"
    for x,i in enumerate(range(0,len(data),2)):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',label=labels[x])
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

def fermi_vs_smearing(data,grid=False,temp=False,savefig=None,axis=None):
    """It plots the Fermi level as a function of smearing for different k_grids
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
        ax.plot(plot_data[:,0],plot_data[:,2],'.-',label=data[i])
    ax.set_ylabel("Fermi level (eV)")
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})
    if temp==True:
        secax=ax.secondary_xaxis('top',functions=(const.smear2temp,const.temp2smear))
        secax.set_xlabel('T [K]')
    if grid == True:
        ax.grid()
    if savefig!=None:
        plt.tight_layout()
        plt.savefig(savefig,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def fermi_vs_Kgrid(data,grid=False,temp=False,savefig=None,axis=None,Kgrids=None):
    """It plots the total Fermi level as a function of K_grid for different smearings
    data: Either the data, or folder where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=conv.reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    labels=data[0::2]
    if temp==True:
        for i,label in enumerate(labels):
            labels[i]=str(label)+" ("+str(int(const.smear2temp(label)))+"K)"
    for x,i in enumerate(range(0,len(data),2)):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,2],'.-',label=labels[x])
    ax.set_ylabel("Fermi level (eV)")
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

def time_vs_smearing(data,grid=False,temp=False,savefig=None,axis=None):
    """It plots the computational time as a function of smearing for different k_grids
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
        ax.plot(plot_data[:,0],plot_data[:,3],'.-',label=data[i])
    ax.set_ylabel("time in hours")
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})
    if temp==True:
        secax=ax.secondary_xaxis('top',functions=(const.smear2temp,const.temp2smear))
        secax.set_xlabel('T [K]')
    if grid == True:
        ax.grid()
    if savefig!=None:
        plt.tight_layout()
        plt.savefig(savefig,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def time_vs_Kgrid(data,grid=False,temp=False,savefig=None,axis=None,Kgrids=None):
    """It plots the total computational time as a function of K_grid for different smearings
    data: Either the data, or folder where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=conv.reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    labels=data[0::2]
    if temp==True:
        for i,label in enumerate(labels):
            labels[i]=str(label)+" ("+str(int(const.smear2temp(label)))+"K)"
    for x,i in enumerate(range(0,len(data),2)):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,3],'.-',label=labels[x])
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


def RAM_vs_smearing(data,grid=False,temp=False,savefig=None,axis=None):
    """It plots the RAM a function of smearing for different k_grids
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
        ax.plot(plot_data[:,0],plot_data[:,4],'.-',label=data[i])
    ax.set_ylabel("RAM (Gb)")
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})
    if temp==True:
        secax=ax.secondary_xaxis('top',functions=(const.smear2temp,const.temp2smear))
        secax.set_xlabel('T [K]')
    if grid == True:
        ax.grid()
    if savefig!=None:
        plt.tight_layout()
        plt.savefig(savefig,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def RAM_vs_Kgrid(data,grid=False,temp=False,savefig=None,axis=None,Kgrids=None):
    """It plots the RAM as a function of K_grid for different smearings
    data: Either the data, or folder where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=conv.reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    labels=data[0::2]
    if temp==True:
        for i,label in enumerate(labels):
            labels[i]=str(label)+" ("+str(int(const.smear2temp(label)))+"K)"
    for x,i in enumerate(range(0,len(data),2)):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,3],'.-',label=labels[x])
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


def forces_vs_smearing(data,grid=False,temp=False,savefig=None,axis=None):
    """It plots the (total force)/(num atoms) a function of smearing for different k_grids
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
        ax.plot(plot_data[:,0],plot_data[:,5],'.-',label=data[i])
    ax.set_ylabel("Total force/atom (meV/au)")
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})
    if temp==True:
        secax=ax.secondary_xaxis('top',functions=(const.smear2temp,const.temp2smear))
        secax.set_xlabel('T [K]')
    if grid == True:
        ax.grid()
    if savefig!=None:
        plt.tight_layout()
        plt.savefig(savefig,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def forces_vs_Kgrid(data,grid=False,temp=False,savefig=None,axis=None,Kgrids=None):
    """It plots the (total force)/(num atoms) as a function of K_grid for different smearings
    data: Either the data, or folder where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=conv.reverse_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    labels=data[0::2]
    if temp==True:
        for i,label in enumerate(labels):
            labels[i]=str(label)+" ("+str(int(const.smear2temp(label)))+"K)"
    for x,i in enumerate(range(0,len(data),2)):
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,5],'.-',label=labels[x])
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


# NEEEEWWWWW


