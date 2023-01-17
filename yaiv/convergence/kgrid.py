#PYTHON module for Kgrid & smearing convergence analysis

import re
import glob
import numpy as np
import matplotlib.pyplot as plt

import yaiv.constants as const
import yaiv.utils as utils
import yaiv.convergence.cutoff as conv

def read_data(folder,shift=True):
    """from the folder where data is stored it reads the scf.pwo files (it autodetects the file extension .pwo, .out, whatever...)
    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    
    shift = Bolean regarding the option os shifting the total energies to zero.
    ...
    OUTPUT:
    A list where odd numbers are the Kdrid and even numbers are the corresponding folder info:
     [[   0.02.         -1416.5169454     -1.489      time          10GB    0.17]
     [   0.018         -1416.51803251    -1.489      time          30Gb     0.22]
     [  smearing(Ry)    energy(meV/atom)  fermi(eV)  time(hours)    RAM    forces(meV/au*atom)]
     [  0.014         -1416.52358896    -1.4889     time    ...]
     [  0.016         -1416.5240187     -1.4889     time    ...]]
    
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

    if shift==True:
        MIN=None
        for d in data[1::2]:        #Get the minimum total energy
            m=np.min(d[:,1])
            if MIN==None:
                MIN=m
            elif MIN>m:
                MIN=m
        for i,d in enumerate(data[1::2]):          #Shift all the data accordingly
            data[2*i+1][:,1]=d[:,1]-MIN

    return data

def energy_vs_smearing(data,grid=True,temp=False,save_as=None,axis=None,shift=True):
    """It plots the energy as a function of smearing for different k_grids

    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    shift = Bolean regarding the option os shifting the total energies to zero.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data,shift=shift)

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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def energy_vs_Kgrid(data,grid=True,temp=False,save_as=None,axis=None,shift=True,Kgrids=None):
    """It plots the total energy as a function of K_grid for different smearings

    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    shift = Bolean regarding the option os shifting the total energies to zero.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data,shift=shift)
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def fermi_vs_smearing(data,grid=True,temp=False,save_as=None,axis=None):
    """It plots the Fermi level as a function of smearing for different k_grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def fermi_vs_Kgrid(data,grid=True,temp=False,save_as=None,axis=None,Kgrids=None):
    """It plots the total Fermi level as a function of K_grid for different smearings
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def time_vs_smearing(data,grid=True,temp=False,save_as=None,axis=None):
    """It plots the computational time as a function of smearing for different k_grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def time_vs_Kgrid(data,grid=True,temp=False,save_as=None,axis=None,Kgrids=None):
    """It plots the total computational time as a function of K_grid for different smearings
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def RAM_vs_smearing(data,grid=True,temp=False,save_as=None,axis=None):
    """It plots the RAM a function of smearing for different k_grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def RAM_vs_Kgrid(data,grid=True,temp=False,save_as=None,axis=None,Kgrids=None):
    """It plots the RAM as a function of K_grid for different smearings
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
        ax.plot(plot_data[:,0],plot_data[:,4],'.-',label=labels[x])
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def forces_vs_smearing(data,grid=True,temp=False,save_as=None,axis=None):
    """It plots the (total force)/(num atoms) a function of smearing for different k_grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def forces_vs_Kgrid(data,grid=True,temp=False,save_as=None,axis=None,Kgrids=None):
    """It plots the (total force)/(num atoms) as a function of K_grid for different smearings
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
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
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def energy(data,grid=True,temp=False,save_as=None,shift=True):
    """It plots the total energy as a function of smearings and grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    shift = Bolean regarding the option os shifting the total energies to zero.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data,shift=shift)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Total energy')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    energy_vs_smearing(data,grid=grid,temp=temp,axis=ax1)
    energy_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax2,Kgrids=Kgrids)
    
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()

def fermi(data,grid=True,temp=False,save_as=None):
    """It plots the total Fermi level as a function of smearings and grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Fermi Energy')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    fermi_vs_smearing(data,grid=grid,temp=temp,axis=ax1)
    fermi_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax2,Kgrids=Kgrids)
    
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()

def time(data,grid=True,temp=False,save_as=None):
    """It plots the total computational time as a function of smearings and grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Computing time')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    time_vs_smearing(data,grid=grid,temp=temp,axis=ax1)
    time_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax2,Kgrids=Kgrids)
    
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()

def RAM(data,grid=True,temp=False,save_as=None):
    """It plots the total needed RAM as a function of smearings and grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('RAM (Gb)')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    RAM_vs_smearing(data,grid=grid,temp=temp,axis=ax1)
    RAM_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax2,Kgrids=Kgrids)
    
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()

def forces(data,grid=True,temp=False,save_as=None):
    """It plots the (total force)/(num atom) as a function of smearings and grids
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Total force')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    forces_vs_smearing(data,grid=grid,temp=temp,axis=ax1)
    forces_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax2,Kgrids=Kgrids)
    
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()


def analysis(data,grid=True,temp=False,save_as=None,shift=True,title=None):
    """It plots all the posible comparisons for a full convergence analysis 
    
    data = Either the data, or folder where data is stored it reads the scf.pwo files and plots.
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    shift = Bolean regarding the option os shifting the total energies to zero.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data,shift=shift)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]

    fig,ax=plt.subplots(5,2,figsize=(10.5,14.5))

    energy_vs_smearing(data,grid=grid,temp=temp,axis=ax[0,0])
    energy_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax[0,1],Kgrids=Kgrids)

    forces_vs_smearing(data,grid=grid,temp=temp,axis=ax[1,0])
    forces_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax[1,1],Kgrids=Kgrids)

    fermi_vs_smearing(data,grid=grid,temp=temp,axis=ax[2,0])
    fermi_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax[2,1],Kgrids=Kgrids)

    time_vs_smearing(data,grid=grid,temp=temp,axis=ax[3,0])
    time_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax[3,1],Kgrids=Kgrids)

    RAM_vs_smearing(data,grid=grid,temp=temp,axis=ax[4,0])
    RAM_vs_Kgrid(data_K,grid=grid,temp=temp,axis=ax[4,1],Kgrids=Kgrids)
    
    if title!=None:
        fig.suptitle(title,y=0.99,size=16)

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.show()
