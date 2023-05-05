import re
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import yaiv.constants as cons

def __is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def minimization_data(logfile):
    lines=open(logfile)
    READ_MIN,READ_FREQ = True,False
    F,F_error,FC,FC_error,S,S_error,KL=[],[],[],[],[],[],[]
    for line in lines:
        if re.search('Running.=.*False',line):
            READ_MIN=False
        elif re.search('Frequencies',line):
            READ_FREQ=True
        if READ_MIN==True:
            if re.search('Free energy',line):
                l=line.split()
                F.append(float(l[3]));F_error.append(float(l[5]))
            elif re.search('FC gradient',line):
                l=line.split()
                FC.append(float(l[4]));FC_error.append(float(l[6]))
            elif re.search('Struct gradient',line):
                l=line.split()
                S.append(float(l[4]));S_error.append(float(l[6]))
            elif re.search('Kong-Liu',line):
                l=line.split()
                KL.append(float(l[5]))
        elif READ_FREQ==True:
            l=line.split()
            if len(l) > 0 and __is_float(l[0]):
                f=[float(x) for x in l]
                try:
                    freqs=np.vstack([freqs,f])
                except NameError:
                    freqs=f
    if 'freqs' not in locals():
        freqs=None
    return F,F_error,FC,FC_error,S,S_error,KL,freqs

def __natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def __get_log_files(folder):
    if folder[-4:]=='.log':
        folder=os.path.dirname(folder)
    logfiles=__natural_sort(glob.glob(folder+'/*.log'))
    return logfiles

def __concatenate_minimization_data(files):
    F,F_error,FC,FC_error,S,S_error,KL=[],[],[],[],[],[],[]
    POP=[]
    for file in files:
        data=minimization_data(file)
        F=F+data[0]
        F_error=F_error+data[1]
        FC=FC+data[2]
        FC_error=FC_error+data[3]
        S=S+data[4]
        S_error=S_error+data[5]
        KL=KL+data[6]
        POP=POP+[len(F)]
        try:
            freqs=np.vstack([freqs,data[7]])
        except NameError:
            freqs=data[7]
    return F,F_error,FC,FC_error,S,S_error,KL,freqs,POP

def track_free_energy(data,grid=True,save_as=None,axis=None,shift=True,full_minim=False):
    """It tracks the Free energy during the minimization
    data = Either the data as read by minimization_data or log file.
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    shift = Bolean regarding the option os shifting the total energies to zero.
    full_minim = Boolean with the option of plotting the full minimization (all populations).
    """
    if full_minim==False:
        if type(data)==str:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=minimization_data(data)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=data
    else:
        if type(data)==str:
            logfiles=__get_log_files(data)
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=__concatenate_minimization_data(logfiles)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=data

    steps=np.arange(len(F))
    if shift==True:
        F=np.array(F)
        F=F-F[-1]

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis
    
    ax.set_title("Free energy")
    ax.set_ylabel("F [meV]")
    ax.set_xlabel("steps")
    ax.errorbar(steps,F,F_error,elinewidth=0.5)

    if full_minim==True:
        for step in POP[:-1]:
            ax.axvline(step,linestyle='--',linewidth=1.2,color='grey')

    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def track_struc_gradient(data,grid=True,save_as=None,axis=None,full_minim=False):
    """It tracks the structure grandient during the minimization
    data = Either the data as read by minimization_data or log file.
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    """
    if full_minim==False:
        if type(data)==str:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=minimization_data(data)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=data
    else:
        if type(data)==str:
            logfiles=__get_log_files(data)
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=__concatenate_minimization_data(logfiles)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=data

    steps=np.arange(len(F))

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    ax.set_title("Structure gradient")
    ax.set_ylabel("meV/A")
    ax.set_xlabel("steps")
    ax.errorbar(steps,S,S_error,elinewidth=0.5)

    if full_minim==True:
        for step in POP[:-1]:
            ax.axvline(step,linestyle='--',linewidth=1.2,color='grey')

    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def track_kong_liu(data,grid=True,save_as=None,axis=None,full_minim=False):
    """It tracks the Kong-Liu ratio during the minimization
    data = Either the data as read by minimization_data or log file.
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    """
    if full_minim==False:
        if type(data)==str:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=minimization_data(data)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=data
    else:
        if type(data)==str:
            logfiles=__get_log_files(data)
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=__concatenate_minimization_data(logfiles)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=data

    steps=np.arange(len(F))

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    ax.set_title("Kong-Liu sample size")
    ax.set_xlabel("steps")
    ax.plot(steps,KL,'.-')

    if full_minim==True:
        for step in POP[:-1]:
            ax.axvline(step,linestyle='--',linewidth=1.2,color='grey')

    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def track_force_cte_gradient(data,grid=True,save_as=None,axis=None,full_minim=False):
    """It tracks the force constant grandient during the minimization
    data = Either the data as read by minimization_data or log file.
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    """
    if full_minim==False:
        if type(data)==str:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=minimization_data(data)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=data
    else:
        if type(data)==str:
            logfiles=__get_log_files(data)
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=__concatenate_minimization_data(logfiles)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=data

    steps=np.arange(len(F))

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis
    ax.set_title("FC gradient")
    ax.set_ylabel("bohr^2")
    ax.set_xlabel("steps")
    ax.errorbar(steps,FC,FC_error,elinewidth=0.5)

    if full_minim==True:
        for step in POP[:-1]:
            ax.axvline(step,linestyle='--',linewidth=1.2,color='grey')

    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()

def track_frequencies(data,grid=True,save_as=None,axis=None,full_minim=False):
    """It tracks the force constant grandient during the minimization
    data = Either the data as read by minimization_data or log file.
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    """
    if full_minim==False:
        if type(data)==str:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=minimization_data(data)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=data
    else:
        if type(data)==str:
            logfiles=__get_log_files(data)
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=__concatenate_minimization_data(logfiles)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=data

    steps=np.arange(len(F))

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis
    ax.set_title("Evolution of the frequencies")
    ax.set_ylabel("Frequencies [cm-1]")
    ax.set_xlabel("steps")
    ax.plot(steps,freqs*cons.Ry2cm)

    if full_minim==True:
        for step in POP[:-1]:
            ax.axvline(step,linestyle='--',linewidth=1.2,color='grey')

    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()


def track_minimization(data,grid=True,save_as=None,shift=True,title=None,full_minim=False):
    """It plots all the posible comparisons for a full convergence analysis 
    
    data = Either the data as read by minimization_data or log file.
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    shift = Bolean regarding the option os shifting the total energies to zero.
    title = A title of your liking
    """
    if full_minim==False:
        if type(data)==str:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=minimization_data(data)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs=data
    else:
        if type(data)==str:
            logfiles=__get_log_files(data)
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=__concatenate_minimization_data(logfiles)
        else:
            F,F_error,FC,FC_error,S,S_error,KL,freqs,POP=data

    fig,ax=plt.subplots(2,3,figsize=(13.5,8.5))
    gs = ax[0, 2].get_gridspec()
    ax[0,2].remove(),ax[1,2].remove()
    ax_freq= fig.add_subplot(gs[0:, 2])
    
    track_free_energy(data,axis=ax[0,0],shift=shift,grid=grid,full_minim=full_minim)
    track_force_cte_gradient(data,axis=ax[0,1],grid=grid,full_minim=full_minim)
    track_struc_gradient(data,axis=ax[1,0],grid=grid,full_minim=full_minim)
    track_kong_liu(data,axis=ax[1,1],grid=grid,full_minim=full_minim)
    if np.any(freqs!=None):
        track_frequencies(data,axis=ax_freq,grid=grid,full_minim=full_minim)
    
    if title!=None:
        fig.suptitle(title,y=0.99,size=16)

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.show()


def hessian_convergence(file,title=None,grid=True,save_as=None,axis=None):
    """It plots the result of the hessian as a function of the number of configs
    
    file = file where the results of the hessians is stored.
    title = A title of your liking
    grid = Bolean that allows for a grid in the plot.
    save_as = name.format in which to save your figure.
    axis = Matplotlib axis in which to plot.
    """
    results=np.loadtxt(file)

    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    ax.plot(results[:,0], results[:,1:], '.-', label='line 1', linewidth=1)
    ax.set_ylabel("frequencies [cm-1]")
    ax.set_xlabel("Number of configs")

    if grid == True:
        ax.grid()
    if save_as!=None:
        plt.tight_layout()
        plt.savefig(save_as,dpi=300)
        plt.show()
    if axis == None:
        plt.tight_layout()
        plt.show()
