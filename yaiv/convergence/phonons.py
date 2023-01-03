#PYTHON module for convergence analysis

import re
import glob
import numpy as np
import matplotlib.pyplot as plt

import yaiv.constants as const
import yaiv.utils as utils
import yaiv.convergence.cutoff as conv

def read_data(folder):
    """from the folder where data is stored it reads the scf.pwo files
    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    ...
    OUTPUT:
    A list where odd numbers are the Kdrid and even numbers are the corresponding folder info:
    [[   80.        freq1  freq1  freq3 ]
     [   90.        freq1  freq1  freq3 ]
     [  100.        freq1  freq1  freq3 ]]

    where first column is the cutoff.
    """
    data=[]
    Kgrids=glob.glob(folder+"/*")
    Kgrids=[i.split("/")[-1] for i in Kgrids]
    Kgrids=conv.__sort_Kgrids(Kgrids)
    for i in range(len(Kgrids)):
        grid_data=np.zeros(0)
        Kgrid=Kgrids[i]
        files=glob.glob(folder+"/"+Kgrid+"/*/*dyn1")
        for j in range(len(files)):
            cutoff=files[j].split("/")[-2]
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
        order=data[i+1][:,0].astype(np.float).argsort()
        data[i+1]=data[i+1][order]   #sort acording to first column (x axis)(cutoff)
        data[i+1]=data[i+1].astype(np.float)
    return data

def __get_cmap(n, name='tab10'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def phonons_vs_cutoff(data,grid=True,savefig=None,axis=None):
    """It plots the frequencies as a function of cutoff for different k_grids
    folder: Either the data, or folder where data is stored it reads the scf.pwo files and plots

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

    cmap=__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',color=color,label=data[i])
        ax.plot(plot_data[:,0],plot_data[:,2:],'.-',color=color)

    ax.set_ylabel("Freqs (cm-1)")
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


def phonons_vs_Kgrid(folder,savefig=None):
    """It plots the total frequencies as a function of K_grid for different cutoffs
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
    cmap=__get_cmap(len(cutoffs))
    fig=plt.figure()
    color_num=0
    for cutoff in cutoffs:
        color=cmap(color_num)
        color_num=color_num+1
        cutoff_data=[]
        for i in range(len(Kgrids)):
            grid=Kgrids[i].split('x')[0]
            if os.path.isdir(folder+"/"+Kgrids[i]+"/"+cutoff):
                cutoff_index=np.where(data[2*i+1][:,0]==int(cutoff))[0][0]
                cutoff_data=cutoff_data+[int(grid),data[2*i+1][cutoff_index,1:]]
        #plt.plot(cutoff_data[0::2],cutoff_data[1::2],'.-',label=cutoff)
        plot_data=cutoff_data[1::2]
        
        plot1=[item[0] for item in plot_data]
        plot2=[item[1:] for item in plot_data]
        plt.plot(cutoff_data[0::2],plot1,'.-',color=color,label=cutoff)
        plt.plot(cutoff_data[0::2],plot2,'.-',color=color)


    plt.ylabel("Total energy (Ry)")
    plt.xlabel("K grid first index")
    plt.xticks([int(i.split('x')[0]) for i in Kgrids])
    plt.legend(prop={'size': 7})
    plt.tight_layout()
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.show()


def phonons_vs_Kgrid(data,grid=True,savefig=None,axis=None,Kgrids=None):
    """It plots the frequencies as a function of cutoff for different k_grids
    folder: Either the data, or folder where data is stored it reads the scf.pwo files and plots

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

    cmap=__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[:,0],plot_data[:,1],'.-',color=color,label=data[i])
        ax.plot(plot_data[:,0],plot_data[:,2:],'.-',color=color)

    ax.set_ylabel("Freqs (cm-1)")
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

def phonons(folder,grid=True,savefig=None):
    """It plots the frequencies as a function of cutoffs and different k_grids
    folder: where data is stored it reads the scf.pwo files and plots

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    And subfolders with the cutoff number
    """
    data=read_data(folder)
    data_K=conv.reverse_data(data)
    Kgrids=data[0::2]
    
    w, h = plt.figaspect(0.5)    
    fig=plt.figure(figsize=(w, h))
    fig.suptitle('Frequencies')
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    phonons_vs_cutoff(data,grid=grid,axis=ax1)
    phonons_vs_Kgrid(data_K,grid=grid,axis=ax2,Kgrids=Kgrids)
    
    if savefig!=None:
        plt.savefig(savefig,dpi=300)
    plt.tight_layout()
    plt.show()

