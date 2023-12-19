#PYTHON module for convergence analysis

import re
import glob
import numpy as np
import matplotlib.pyplot as plt

import yaiv.constants as const
import yaiv.utils as utils
import yaiv.convergence.cutoff as conv

def read_data(folder):
    """from the folder where data is stored it reads the foo.dyn1 files
    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the (cutoff/smearing) number
    ...
    OUTPUT:
    A list where odd numbers are the Kdrid and even numbers are the corresponding folder info:
    [[  (cutoff/smearing)        freq1  freq1  freq3 ...]
     [   90.        freq1  freq1  freq3 ...]
     [  100.        freq1  freq1  freq3 ...]]

    where first column is the (cutoff/smearing).
    """
    data=[]
    Kgrids=glob.glob(folder+"/*")
    Kgrids=[i.split("/")[-1] for i in Kgrids]
    Kgrids=conv.__sort_Kgrids(Kgrids)
    for i in range(len(Kgrids)):
        grid_data=np.zeros(0)
        Kgrid=Kgrids[i]
        files=glob.glob(folder+"/"+Kgrid+"/**/*dyn*",recursive=True)
        for j in range(len(files)):
            cutoff=files[j].split(folder)[1].split('/')[2]
            cutoff_data=np.array([cutoff])
            file = open (files[j],"r")
            for line in file:
                if re.search("freq",line):
                    freq=line.split()[7]
                    cutoff_data=np.append(cutoff_data,freq)
            if grid_data.shape[0]==0:
                grid_data=cutoff_data
            else:
                grid_data=np.vstack((grid_data,cutoff_data))
        data=data+[Kgrid]+[grid_data]
    for i in range(0,len(data),2):
        order=data[i+1][:,0].astype(float).argsort()
        data[i+1]=data[i+1][order]   #sort acording to first column (x axis)(cutoff)
        data[i+1]=data[i+1].astype(float)
    return data

def __get_cmap(n, name='tab10'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def phonons_vs_cutoff(data,freqs=None,grid=True,save_as=None,axis=None,title=None):
    """It plots the frequencies as a function of cutoff for different k_grids
    
    data = Either the data, or folder where data is stored (it reads the foo.dyn1 files.)
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

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

    if freqs==None:
        freqs=len(data[1][0])
    freqs=freqs+1

    cmap=__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',color=color,label=data[i])
        ax.plot(plot_data[:,0],plot_data[:,2:freqs],'.-',color=color)

    ax.set_ylabel("Freqs (cm-1)")
    ax.set_xlabel("cutoff (Ry)")
    ax.legend(prop={'size': 7})

    if title!=None:
        fig.suptitle(title)
    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def phonons_vs_smearing(data,freqs=None,grid=True,temp=False,save_as=None,axis=None,title=None):
    """It plots the frequencies as a function of the smearing for different k_grids
    
    data = Either the data, or folder where data is stored (it reads the foo.dyn1 files.)
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

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

    if freqs==None:
        freqs=len(data[1][0])
    freqs=freqs+1

    cmap=__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',color=color,label=data[i])
        ax.plot(plot_data[:,0],plot_data[:,2:freqs],'.-',color=color)

    ax.set_ylabel("Freqs (cm-1)")
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})

    if title!=None:
        fig.suptitle(title)
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


def phonons_vs_Kgrid_cutoff(data,freqs=None,grid=True,save_as=None,axis=None,Kgrids=None,title=None):
    """It plots the frequencies as a function of Kgrid for different cutoffs
    
    data = Either the data, or folder where data is stored (it reads the foo.dyn1 files.)
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

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

    if freqs==None:
        freqs=len(data[1][0])
    freqs=freqs+1

    cmap=__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',color=color,label=data[i])
        ax.plot(plot_data[:,0],plot_data[:,2:freqs],'.-',color=color)

    ax.set_ylabel("Freqs (cm-1)")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
    ax.legend(prop={'size': 7})
    
    if title!=None:
        fig.suptitle(title)
    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def phonons_vs_Kgrid_smearing(data,freqs=None,grid=True,temp=False,save_as=None,axis=None,Kgrids=None,title=None):
    """It plots the frequencies as a function of Kgrid for different smearings
    
    data = Either the data, or folder where data is stored (it reads the foo.dyn1 files.)
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

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

    if freqs==None:
        freqs=len(data[1][0])
    freqs=freqs+1

    labels=data[0::2]
    if temp==True:
        for i,label in enumerate(labels):
            labels[i]=str(label)+" ("+str(int(const.smear2temp(label)))+"K)"

    cmap=__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',color=color,label=labels[int(i/2)])
        ax.plot(plot_data[:,0],plot_data[:,2:freqs],'.-',color=color)

    ax.set_ylabel("Freqs (cm-1)")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
    ax.legend(prop={'size': 7})

    if title!=None:
        fig.suptitle(title)
    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def phonons_cutoff(data,freqs=None,grid=True,save_as=None,title=None):
    """It plots the frequencies as a function of cutoffs and different k_grids
    
    data = Either the data, or folder where data is stored (it reads the foo.dyn1 files.)
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    if type(data)==str:
        data=read_data(data)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]

    if freqs==None:
        freqs=len(data[1][0])
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    phonons_vs_cutoff(data,freqs=freqs,grid=grid,axis=ax1)
    phonons_vs_Kgrid_cutoff(data_K,freqs=freqs,grid=grid,axis=ax2,Kgrids=Kgrids)
    
    if title!=None:
        fig.suptitle(title)
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()

def phonons_Kgrid(data,freqs=None,grid=True,temp=False,save_as=None,title=None):
    """It plots the frequencies as a function of smearings and different k_grids
    
    data = Either the data, or folder where data is stored (it reads the foo.dyn1 files.)
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    temp = Bolean if you want a secondary axis with the Temperature equiv of the smearing value (usefull for Fermi-Dirac smearing).
    save_as = name.format in which to save your figure.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the smearing number
    """
    if type(data)==str:
        data=read_data(data)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]

    if freqs==None:
        freqs=len(data[1][0])
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    phonons_vs_smearing(data,freqs=freqs,grid=grid,temp=temp,axis=ax1)
    phonons_vs_Kgrid_smearing(data_K,freqs=freqs,grid=grid,temp=temp,axis=ax2,Kgrids=Kgrids)
    
    if title!=None:
        fig.suptitle(title)
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()
