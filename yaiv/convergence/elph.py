#This should be added to the convergence module

import numpy as np
import glob
import re
import matplotlib.pyplot as plt

import yaiv.convergence as conv
import yaiv.constants as const

def read_eph_dyn(file):
    """
    Read a Quantum espresso dyn.elph file and collects smearings, lambdas and gammas. 
    Each row at the lamdas or gammas matrix corresponds to an smearing.
    return smearings, lambdas, gammas
    """
    lines=open(file,'r')
    for i,line in enumerate(lines):
        if i==0:
            num=int(line.split()[-1])
        if re.search('Gaussian Broadening',line):
            s=float(line.split()[2])
            try:
                smearings=np.hstack((smearings,s))
            except NameError:
                smearings=s                
        elif re.search('lambda',line):
            line=line.split('=')
            l=float(line[1].split()[0])
            g=float(line[2].split()[0])
            try:
                lam=np.hstack((lam,l))
                gam=np.hstack((gam,g))
            except NameError:
                lam=np.array([l])
                gam=g
            if len(lam)==num:
                try:
                    lambdas=np.vstack((lambdas,lam))
                    gammas=np.vstack((gammas,gam))
                except NameError:
                    lambdas=lam
                    gammas=gam
                del lam
                del gam
    lines.close()
    return smearings, lambdas, gammas

def read_data(folder):
    """
    From the folder where data is stored it reads the foo.dyn.elph.1 files
    Data must be organized with parent folders with the K grid as:
    K1xK2xK3

    OUTPUT:
    A list where odd numbers are the Kdrid and even numbers are the output from read_eph_dyn:
    [smearings,lambdas,gammas]
    """
    data=[]
    Kgrids=glob.glob(folder+"/*")
    Kgrids=[i.split("/")[-1] for i in Kgrids]
    Kgrids=conv.cutoff.__sort_Kgrids(Kgrids)
    data=[]
    for grid in Kgrids:
        files=glob.glob(folder+'/'+grid+'/**/*dyn.elph*',recursive=True)
        for file in files:
            n = list(read_eph_dyn(file))
            try:
                new[0]=np.hstack((new[0],n[0]))
                new[1]=np.vstack((new[1],n[1]))
                new[2]=np.vstack((new[2],n[2]))
            except NameError:
                new=n
        new[1]=new[1][new[0].argsort()]
        new[2]=new[2][new[0].argsort()]
        new[0]=new[0][new[0].argsort()]
        data=data+[grid]+[new]
        del new
    return data

def reverse_data(data_in):
    """From the data obtained with read_data it reverses so that the format is:
    Odd numbers are the smearings and even numbers are other sublist as:
    [K1xK2xK3,lambdas,gammas]
    """
    data=data_in.copy()
    for i in range(1,len(data),2):
        data[i]=np.hstack((data[i][0].reshape(np.shape(data[i][1])[0],1),data[i][1],data[i][2]))
    Kgrids=data[0::2]
    data=conv.cutoff.reverse_data(data)
    for i in range(1,len(data),2):
        grids=data[i][:,0]
        num=int((len(data[i][0])-1)/2)
        lambdas=data[i][:,1:num+1]
        gammas=data[i][:,num+1:]
        data[i]=[grids,lambdas,gammas]
    return data


def lambda_vs_smearing(data,freqs=None,grid=True,save_as=None,axis=None,title='E-ph coupling'):
    """
    Plots the Lambda (electron-phonon coupling strength) as a function of the smearing.

    data = Either the data, or folder where data is stored it reads the .elph.1 files and plots.
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    with subfiles as foo.dyn.elph.1 files
    """
    if type(data)==str:
        data=read_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    if freqs==None:
        freqs=len(data[1][1][0])
    freqs=freqs+1

    cmap=conv.phonons.__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[0],plot_data[1][:,0],'.-',color=color,label=data[i])
        ax.plot(plot_data[0],plot_data[1][:,2:freqs],'.-',color=color)

    ax.set_ylabel("Lambda")
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})

    if title!=None:
        ax.set_title(title)
    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def gamma_vs_smearing(data,freqs=None,grid=True,save_as=None,axis=None,title='Linewidths'):
    """
    Plots the gamma (phonon linewidths) as a function of the smearing.

    data = Either the data, or folder where data is stored it reads the .elph.1 files and plots.
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    with subfiles as foo.dyn.elph.1 files
    """
    if type(data)==str:
        data=read_data(data)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    if freqs==None:
        freqs=len(data[1][1][0])
    freqs=freqs+1

    cmap=conv.phonons.__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[0],plot_data[2][:,0]*const.GHz2meV,'.-',color=color,label=data[i])
        ax.plot(plot_data[0],plot_data[2][:,2:freqs]*const.GHz2meV,'.-',color=color)

    ax.set_ylabel("Gamma (meV)")
    ax.set_xlabel("smearing (Ry)")
    ax.legend(prop={'size': 7})

    if title!=None:
        ax.set_title(title)
    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def lambda_vs_Kgrid(data,freqs=None,grid=True,save_as=None,axis=None,Kgrids=None,title='E-ph coupling'):
    """
    Plots the Lambda (electron-phonon coupling strength) as a function of the Kgrid.

    data = Either the data, or folder where data is stored it reads the .elph.1 files and plots.
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    with subfiles as foo.dyn.elph.1 files
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

    if freqs==None:
        freqs=len(data[1][1][0])
    freqs=freqs+1

    cmap=conv.phonons.__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[0],plot_data[1][:,0],'.-',color=color,label=data[i])
        ax.plot(plot_data[0],plot_data[1][:,2:freqs],'.-',color=color)

    ax.set_ylabel("Lambda")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
    ax.legend(prop={'size': 7})
    
    if title!=None:
        ax.set_title(title)
    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def gamma_vs_Kgrid(data,freqs=None,grid=True,save_as=None,axis=None,Kgrids=None,title='Linewidths'):
    """
    Plots the gamma (phonon linewidths) as a function of the Kgrid.

    data = Either the data, or folder where data is stored it reads the .elph.1 files and plots.
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    with subfiles as foo.dyn.elph.1 files
    """

    if type(data)==str:
        data=read_data(data)
        Kgrids=data[0::2]
        data=reverse_data(data)

    if axis == None:
        ax.set_title(title)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    if freqs==None:
        freqs=len(data[1][1][0])
    freqs=freqs+1

    cmap=conv.phonons.__get_cmap(int(len(data)/2))
    for i in range(0,len(data),2):
        color=cmap(int(i/2))
        plot_data=data[i+1]
        ax.plot(plot_data[0],plot_data[2][:,0]*const.GHz2meV,'.-',color=color,label=data[i])
        ax.plot(plot_data[0],plot_data[2][:,2:freqs]*const.GHz2meV,'.-',color=color)

    ax.set_ylabel("Gamma (meV)")
    ax.set_xlabel("K point number")
    grids_K=[]
    for g in Kgrids:
        n=np.prod([int(x) for x in g.split('x')])
        grids_K=grids_K+[n]
    ax.set_xticks(grids_K,labels=Kgrids,rotation=60)
    ax.legend(prop={'size': 7})
    
    if title!=None:
        ax.set_title(title)
    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def elph_convergence(data,freqs=None,grid=True,save_as=None,title=None):
    """
    Plots the convergence of lambda (electron phonon strength) and gammas (linewidths)

    data = Either the data, or folder where data is stored it reads the .elph.1 files and plots.
    freqs = Select until which frequency you want to plot (integer).
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    title = A title of your liking

    Data must be organized with parent folders with the K grid as:
    K1xK2xK3
    with subfiles as foo.dyn.elph.1 files
    """

    if type(data)==str:
        data=read_data(data)
    data_K=reverse_data(data)
    Kgrids=data[0::2]
    if freqs==None:
        freqs=len(data[1][1][0])
    
    fig=plt.figure(figsize=(10.5,7))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    lambda_vs_smearing(data,freqs=freqs,grid=grid,axis=ax1)
    lambda_vs_Kgrid(data_K,freqs=freqs,grid=grid,axis=ax2,Kgrids=Kgrids)
    gamma_vs_smearing(data,freqs=freqs,grid=grid,axis=ax3)
    gamma_vs_Kgrid(data_K,freqs=freqs,grid=grid,axis=ax4,Kgrids=Kgrids)
    
    if title!=None:
        fig.suptitle(title)
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.tight_layout()
    plt.show()
