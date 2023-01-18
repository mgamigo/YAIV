#PYTHON module for ploting 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import re
import glob
import os
import subprocess
from scipy.interpolate import griddata
import spglib as spg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import yaiv.experimental.cell_analyzer as cell
import yaiv.constants as cons
import yaiv.utils as ut
#import yaiv.transformations as trs

# Density Of States***********************************************************************

def __sum_pdos(folder):
    """Sums the pDOs for equal elements and orbitals to get the total contribution
    folder = String with the folder where the DOS files are.
    """
    #Generate the names for the desired pDOSes
    files=glob.glob(folder+'/*wfc*')
    elements=[]
    for file in files:
        elem=file.split('(')[1]
        elem=elem.split(')')[0]
        if elem not in elements:
            elements=elements+[elem]
    DOS_names=[]
    for elem in elements:
        orbitals=[]
        files=glob.glob(folder+'/*'+elem+'*wfc*')
        for file in files:
            orbital=file.split('(')[2]
            orbital=orbital.split('_')[0]
            if orbital not in orbitals:
                orbitals=orbitals+[orbital]
                DOS_names=DOS_names+[elem+'_'+orbital]

    #Summing the DOS for equal elements and orbitals to get total contribution
    PREFIX=os.getcwd()
    PATH='/home/martin/Software/qe-7.0/bin/'
    os.chdir(folder)
    for name in DOS_names:
        element=name.split('_')[0]
        orbital=name.split('_')[1]
        if not os.path.exists(name+'.dat'):
            cmd=PATH+'sumpdos.x *atm*\('+element+'\)*wfc*\('+orbital+'_* > '+name+'.dat'
            print(cmd)
            os.system(cmd)
    if not os.path.exists('total.dat'):
        print('Summing over the DOS files:')
        cmd=PATH+'sumpdos.x *atm*\(*\)*wfc*\(*_* > total.dat'
        print(cmd)
        os.system(cmd)
    os.chdir(str(PREFIX))
    return DOS_names

def __sum_pdos_WP(folder,scf_file):
    """Sums the pDOs for equal Wyckoff positions and orbitals to get the total contribution
    folder = String with the folder where the DOS files are.
    scf = Scf file in order to read the wyckoff positions
    """
    #Get equivalent WP
    atoms,wyckoff,pos,indices=cell.wyckoff_positions(scf_file)
    #Take account for doubled Wyckoff positions (same atom in another WP with the same name)
    atoms_wyckoff=[]
    for i in range(len(atoms)):
        atoms_wyckoff=atoms_wyckoff+[[atoms[i],wyckoff[i]]]
    for item in atoms_wyckoff:
        if atoms_wyckoff.count(item)>1:
            num=1
            duplicate=list(item)
            while atoms_wyckoff.count(duplicate)>0:
                i=atoms_wyckoff.index(duplicate)
                atoms_wyckoff[i][1]=atoms_wyckoff[i][1]+str(num)
                num=num+1
    atoms=list(np.array(atoms_wyckoff)[:,0])
    wyckoff=list(np.array(atoms_wyckoff)[:,1])

    #orbitals for each WP
    orbitals=[]
    for elem in atoms:
        orb=[]
        files=glob.glob(folder+'/*'+elem+'*wfc*')
        for file in files:
            orbital=file.split('(')[2]
            orbital=orbital.split('_')[0]
            if orbital not in orb:
                orb=orb+[orbital]
        orbitals=orbitals+[orb]
    #Summing the DOS for equal elements and orbitals to get total contribution
    PREFIX=os.getcwd()
    PATH='/home/martin/Software/qe-7.0/bin/'
    os.chdir(folder)
    DOS_names=[]
    for i,elem in enumerate(atoms):
        for orb in orbitals[i]:
            name=elem+'_'+orb+'_'+wyckoff[i]
            DOS_names=DOS_names+[name]
            if not os.path.exists(name+'.dat'):
                arguments = ''
                for num in indices[i]:
                    arguments=arguments+' pdos*atm#'+str(num+1)+'*wfc*\('+orb+'_*'            
                cmd=PATH+'sumpdos.x'+arguments+' > '+name+'.dat'
                print(name)
                print(cmd)
                os.system(cmd)
    if not os.path.exists('total.dat'):
        print('Summing over the DOS files:')
        cmd=PATH+'sumpdos.x *atm*\(*\)*wfc*\(*_* > total.dat'
        print(cmd)
        os.system(cmd)
    os.chdir(str(PREFIX))
    return DOS_names

def DOS(folder,nscf_out=None,title=None,only_tot=False,fermi=None,window=None,WP_list=None,save_as=None,
        figsize=None,axis=None):
    """Plots the total DOS and projected DOS (pDOS) over the different elements, orbitals and Wyckoff positions
    folder = String with the folder with the output of projwfc.x (Quantum Espresso)
    nscf_out = Output file of pw.x from which you want to read Fermi Energy and crystal structure

    However you can enter all manually:
    title = 'Your nice and original title for the plot'
    only_tot = Bolean (whether you want only the total DOS to be plotted)
    fermi = float with the Fermi energy for the DOS
    window = Window of energies to plot around Fermi, it can be a single number or 2
            either window=0.5 or window=[-0.5,0.5] => Same result
    WP_list = List containing the indices of DOS elements you want to plot
    save_as='wathever.format'
    figsize = (int,int) => Size and shape of the figure
    axis = ax in which to plot, if no axis is present new figure is created
    """
    if nscf_out!=None:
        if fermi==None:
            fermi=ut.grep_fermi(nscf_out)
        DOS_names=__sum_pdos_WP(folder,nscf_out)
    else:
        if fermi==None:
            fermi=0
        DOS_names=__sum_pdos(folder)
    DOS_names=['total']+DOS_names
    print(DOS_names)

    if WP_list!=None or only_tot==True:
        if only_tot==True:
            new_list=['total']
        else:
            new_list=[]
            for item in WP_list:
                new_list=new_list+[DOS_names[item]]
        DOS_names=new_list

    #Plotting the pDOS
    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for name in DOS_names:
        if name=='total':
            total=np.loadtxt(folder+'/total.dat')
            total[:,0]=total[:,0]-fermi
            for i in range(len(total[:,1])):
                if total[i,1]<0:
                    total[i,1]=0
            ax.plot(total[:,1],total[:,0],label='total')
        else:
            data=np.loadtxt(folder+'/'+name+'.dat')
            data[:,0]=data[:,0]-fermi
            for i in range(len(data[:,1])):
                if data[i,1]<0:
                    data[i,1]=0
            ax.plot(data[:,1],data[:,0],label=name)
    ax.legend()

    if window!=None:                   #Limits y axis
        if type(window) is int or type(window) is float:
            window=[-window,window]
        ax.set_ylim(window[0],window[1])
        low=0
        up=0
        for i,item in enumerate(total[:,0]):
            if item > window[0] and low==0:
                low=i
            elif item > window[1] and up==0:
                up=i
                break
        print(low,up)
        maxval=np.max(total[low:up,1])
        print(maxval)
        ax.set_xlim(-0.01,maxval*1.1)


    if title!=None:                             #Title option
        ax.set_title(title)
    ax.set_xlabel('DOS')
    if fermi!=0:
        ax.set_ylabel('Energy $(E-E_f)$ $eV$')
        ax.axhline(y=0,linestyle='--',color='black',linewidth=0.4)
    else:
        ax.set_ylabel('Energy $eV$')
    plt.tight_layout()
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis==None:                             #Saving option
        plt.show()

# WannierTools***********************************************************************

def __load_surfdos_data(file,only_surfdos=False):
    """Loads the surfdos data of WannierTools and interpolates the data to plot in a imshow way
    file = dat.dos file from WT
    only_surfdos = if you want only the contribution of the surface
    """
    if only_surfdos == False and os.path.exists(file+'_interp'):
        data=np.loadtxt(file)
        Z=np.loadtxt(file+'_interp')
    elif only_surfdos == True and os.path.exists(file+'_interp_surfonly'):
        data=np.loadtxt(file)
        Z=np.loadtxt(file+'_interp_surfonly')
    else:
        print('Interpolating...')
        data=np.loadtxt(file)
        #data must be a [M,N] grid
        M=1
        previous=data[0,1]
        for elem in data[1:,1]:
            if elem!=previous:
                M=M+1
            else:
                break
        N=int(data.shape[0]/M)
        if N*M != data.shape[0]:
            print("There is an error reading the data CHECK!!!")

        #INTERPOLATION
        x = np.linspace(data[0,0],data[-1,0], N)
        y = np.linspace(data[0,1],data[-1,1], M)
        X, Y = np.meshgrid(x, y)
        if only_surfdos == False:
            Z = griddata((data[:,0],data[:,1]),data[:,2],(X, Y), method='cubic')
            np.savetxt(file+'_interp',Z)
        elif only_surfdos == True:
            Z = griddata((data[:,0],data[:,1]),data[:,3],(X, Y), method='cubic')
            np.savetxt(file+'_interp_surfonly',Z)
        print('Done')
    return Z, data


def surfdos(file,title=None,save_as=None,only_surfdos=False,colormap='plasma'):
    """Plots and interpolates the surface DOS generated by WannierTools
    file = dos.dat file from WT
    title = string with the title for the plot
    save_as = string with the saving file and extensions
    only_surfdos = if you want only the contribution of the surface
    colormap = colormap (plasma, viridis, inferno, cividis...)

    It might get faster if we remove interpolation from plt.imshow
    """

    Z, data = __load_surfdos_data(file,only_surfdos)

    #GNUFILE for xticks
    gnu_file=''
    path=file.split('/')
    for part in path[:-1]:
        gnu_file=gnu_file+part+'/'
    gnu_file=gnu_file+'surfdos_bulk.gnu'

    gnu=open(gnu_file,'r')
    for line in gnu:
        if re.search('set xtics \(',line):
            line=line.split('(')[1]
            line=line.split(')')[0]
            line=line.split(',')
            labels=[]
            ticks=[]
            for i in line:
                i=i.split()
                labels=labels+[i[0].split('"')[1]]
                ticks=ticks+[float(i[1])]

    #PLOTING
    fig, ax = plt.subplots()
    pos=ax.imshow(Z,interpolation='bilinear',aspect='auto',cmap=colormap,
                 origin='lower',extent=(data[0,0],data[-1,0],data[1,1],data[-1,1]))
    cbar=fig.colorbar(pos, ax=ax)
    cbar.set_ticks([])

    # draw gridlines
    plt.axhline(y=0,color='gray',linestyle='-',linewidth=0.3)
    plt.xticks(ticks,labels)
    for tick in ticks:
        plt.axvline(tick,color='black',linestyle='-',linewidth=0.3)
    ax.set_ylabel('eV')

    if title!=None:
        plt.title(title)
    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()

def surfdos_all(folder,title=None,save_as=None,colormap='plasma'):
    """Plots and interpolates the surface DOS generated by WannierTools
    bulk_file = dos.bulk.dat file from WT
    surf = dos.surf.dat file from WT
    title = string with the title for the plot
    save_as = string with the saving file and extensions
    colormap = colormap (plasma, viridis, inferno, cividis...)

    It might get faster if we remove interpolation from plt.imshow
    """

    Z_bulk, data_b =__load_surfdos_data(folder+'/dos.dat_bulk')
    Z_sl, data_sl=__load_surfdos_data(folder+'/dos.dat_l')
    Z_sr, data_sr=__load_surfdos_data(folder+'/dos.dat_r')

    #GNUFILE for xticks
    gnu_file=folder+'/surfdos_bulk.gnu'

    gnu=open(gnu_file,'r')
    for line in gnu:
        if re.search('set xtics \(',line):
            line=line.split('(')[1]
            line=line.split(')')[0]
            line=line.split(',')
            labels=[]
            ticks=[]
            for i in line:
                i=i.split()
                labels=labels+[i[0].split('"')[1]]
                ticks=ticks+[float(i[1])]

    #PLOTING
    fig = plt.figure(figsize=(9,3.5))
    if title!=None:
        fig.suptitle(title)

    ax1 = fig.add_subplot(1,3,1)
    ax2 = fig.add_subplot(1,3,2)
    ax3 = fig.add_subplot(1,3,3)


    ax1.imshow(Z_bulk,interpolation='bilinear',aspect='auto',cmap=colormap,origin='lower',
                   extent=(data_b[0,0],data_b[-1,0],data_b[1,1],data_b[-1,1]))
    ax2.imshow(Z_sl,interpolation='bilinear',aspect='auto',cmap=colormap,origin='lower',
                   extent=(data_sl[0,0],data_sl[-1,0],data_sl[1,1],data_sl[-1,1]))
    ax3.imshow(Z_sr,interpolation='bilinear',aspect='auto',cmap=colormap,origin='lower',
                   extent=(data_sr[0,0],data_sr[-1,0],data_sr[1,1],data_sr[-1,1]))

    # draw gridlines
    ax1.axhline(y=0,color='gray',linestyle='-',linewidth=0.3)
    ax2.axhline(y=0,color='gray',linestyle='-',linewidth=0.3)
    ax3.axhline(y=0,color='gray',linestyle='-',linewidth=0.3)

    ax1.set_title("Bulk")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)
    ax2.set_title("Surface L")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels)
    ax3.set_title("Surface R")
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(labels)
    for tick in ticks:
        ax1.axvline(tick,color='black',linestyle='-',linewidth=0.3)
        ax2.axvline(tick,color='black',linestyle='-',linewidth=0.3)
        ax3.axvline(tick,color='black',linestyle='-',linewidth=0.3)

    ax1.set_ylabel('eV')
    ax2.set_yticks([])
    ax3.set_yticks([])

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()


def surfdos_only(folder,title=None,save_as=None,colormap='plasma'):
    """Plots and interpolates the surface DOS generated by WannierTools
    bulk_file = dos.bulk.dat file from WT
    surf = dos.surf.dat file from WT
    title = string with the title for the plot
    save_as = string with the saving file and extensions
    colormap = colormap (plasma, viridis, inferno, cividis...)

    It might get faster if we remove interpolation from plt.imshow
    """

    Z_sl, data_sl =__load_surfdos_data(folder+'/dos.dat_l',only_surfdos=True)
    Z_sr, data_sr=__load_surfdos_data(folder+'/dos.dat_r',only_surfdos=True)

    #GNUFILE for xticks
    gnu_file=folder+'/surfdos_bulk.gnu'
    
    gnu=open(gnu_file,'r')
    for line in gnu:
        if re.search('set xtics \(',line):
            line=line.split('(')[1]
            line=line.split(')')[0]
            line=line.split(',')
            labels=[]
            ticks=[]
            for i in line:
                i=i.split()
                labels=labels+[i[0].split('"')[1]]
                ticks=ticks+[float(i[1])]

    #PLOTING
    fig = plt.figure(figsize=(6.3,3.5))
    if title!=None:
        fig.suptitle(title)

    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)


    ax1.imshow(Z_sl,interpolation='bilinear',aspect='auto',cmap=colormap,origin='lower',
                   extent=(data_sl[0,0],data_sl[-1,0],data_sl[1,1],data_sl[-1,1]))
    ax2.imshow(Z_sr,interpolation='bilinear',aspect='auto',cmap=colormap,origin='lower',
                   extent=(data_sr[0,0],data_sr[-1,0],data_sr[1,1],data_sr[-1,1]))

    # draw gridlines
    ax1.axhline(y=0,color='gray',linestyle='-',linewidth=0.3)
    ax2.axhline(y=0,color='gray',linestyle='-',linewidth=0.3)

    ax1.set_title("Surface only L")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labels)
    ax2.set_title("Surface only R")
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(labels)
    for tick in ticks:
        ax1.axvline(tick,color='black',linestyle='-',linewidth=0.3)
        ax2.axvline(tick,color='black',linestyle='-',linewidth=0.3)

    ax1.set_ylabel('eV')
    ax2.set_yticks([])

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()


def __load_fermiarc_data(file):
    """Loads the surfdos data of WannierTools and interpolates the data to plot in a imshow way
    file = dat.dos file from WT
    only_surfdos = if you want only the contribution of the surface
    """
    if os.path.exists(file+'_interp'):
        data=np.loadtxt(file)
        Z=np.loadtxt(file+'_interp')
    else:
        print('Interpolating...')
        data=np.loadtxt(file)
        #data must be a [Kx,Ky] grid
        Kx=1
        previous=data[0,0]
        for elem in data[1:,0]:
            if elem!=previous:
                Kx=Kx+1
            else:
                break
        Ky=int(data.shape[0]/Kx)
        if Kx*Ky != data.shape[0]:
            print("There is an error reading the data CHECK!!!")

        #INTERPOLATION
        x = np.linspace(data[0,0],data[-1,0], Kx)
        y = np.linspace(data[0,1],data[-1,1], Ky)
        X, Y = np.meshgrid(x, y)
        Z = griddata((data[:,0],data[:,1]),data[:,2],(X, Y), method='cubic')
        np.savetxt(file+'_interp',Z)
        
        print('Done')
    return Z, data

def fermiarc(file,title=None,save_as=None,colormap='plasma'):
    """Plots and interpolates the surface DOS generated by WannierTools
    file = arc.dat file from WT
    title = string with the title for the plot
    save_as = string with the saving file and extensions
    colormap = colormap (plasma, viridis, inferno, cividis...)

    It might get faster if we remove interpolation from plt.imshow
    """

    Z, data = __load_fermiarc_data(file)

    #PLOTING
    fig, ax = plt.subplots()
    pos=ax.imshow(Z,interpolation='bilinear',aspect='auto',cmap=colormap,
                 origin='lower',extent=(data[0,0],data[-1,0],data[1,1],data[-1,1]))
    cbar=fig.colorbar(pos, ax=ax)
    cbar.set_ticks([])

    # draw gridlines
    plt.axhline(y=0,color='lightgray',linestyle='-',linewidth=0.3)
    plt.axvline(x=0,color='lightgray',linestyle='-',linewidth=0.3)

#    plt.xticks(ticks,labels)
#    for tick in ticks:
#        plt.axvline(tick,color='black',linestyle='-',linewidth=0.3)
    ax.set_ylabel('$k_y$')
    ax.set_xlabel('$k_x$')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.plot(0,0,'o', markersize=2,color='white') 
    ax.text(x=0.01, y=0.005, s='$\Gamma$', fontsize=12,color='white')

    plt.plot(data[-1,0]-0.005,data[-1,1]-0.0005,'o', markersize=2,color='white') 
    ax.text(x=data[-1,0]-0.05, y=data[-1,1]-0.015, s='$M$', fontsize=12,color='white')
    
    plt.plot(data[-1,0]-0.005,0,'o', markersize=2,color='white') 
    ax.text(x=data[-1,0]-0.05, y=0.005, s='$X$', fontsize=12,color='white')
    
    plt.plot(0,data[-1,1]-0.0005,'o', markersize=2,color='white') 
    ax.text(x=0.01, y=data[-1,1]-0.015, s='$Y$', fontsize=12,color='white')    


    if title!=None:
        plt.title(title)
    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()
    
def fermiarc_LR(folder,title=None,save_as=None,colormap='plasma'):
    """Plots and interpolates the surface DOS generated by WannierTools
    file = arc.dat file from WT
    title = string with the title for the plot
    save_as = string with the saving file and extensions
    colormap = colormap (plasma, viridis, inferno, cividis...)

    It might get faster if we remove interpolation from plt.imshow
    """
    L_file=folder+'/arc.dat_l'
    R_file=folder+'/arc.dat_r'
    Z_L, data_L = __load_fermiarc_data(L_file)
    Z_R, data_R = __load_fermiarc_data(R_file)
    
    #PLOTTING
    fig = plt.figure(figsize=(8.5,4))
    
    if title!=None:
        if title==True:
            if folder[-1]=='/':
                title=folder.split('/')[-2]
            else:
                title=folder.split('/')[-1]
        fig.suptitle(title)
        
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)
    ax1.set_title("Left")
    ax2.set_title("Right")
    
    #PLOT_LEFT
    pos=ax1.imshow(Z_L,interpolation='bilinear',aspect='auto',cmap=colormap,
                 origin='lower',extent=(data_L[0,0],data_L[-1,0],data_L[1,1],data_L[-1,1]))
    cbar=fig.colorbar(pos, ax=ax1)
    cbar.set_ticks([])
    
    #PLOT_RIGHT
    pos=ax2.imshow(Z_R,interpolation='bilinear',aspect='auto',cmap=colormap,
                 origin='lower',extent=(data_R[0,0],data_R[-1,0],data_R[1,1],data_R[-1,1]))
    cbar=fig.colorbar(pos, ax=ax2)
    cbar.set_ticks([])
    
    #BEAUTIFY LEFT
    # draw gridlines
    ax1.axhline(y=0,color='lightgray',linestyle='-',linewidth=0.3)
    ax1.axvline(x=0,color='lightgray',linestyle='-',linewidth=0.3)
    
    ax1.set_ylabel('$k_y$')
    ax1.set_xlabel('$k_x$')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.plot(0,0,'o', markersize=2,color='white') 
    ax1.text(x=0.01, y=0.005, s='$\Gamma$', fontsize=12,color='white')

    ax1.plot(data_L[-1,0]-0.005,data_L[-1,1]-0.0005,'o', markersize=2,color='white') 
    ax1.text(x=data_L[-1,0]-0.1, y=data_L[-1,1]-0.02, s='$M$', fontsize=12,color='white')
    
    ax1.plot(data_L[-1,0]-0.005,0,'o', markersize=2,color='white') 
    ax1.text(x=data_L[-1,0]-0.1, y=0.005, s='$X$', fontsize=12,color='white')
    
    ax1.plot(0,data_L[-1,1]-0.0005,'o', markersize=2,color='white') 
    ax1.text(x=0.01, y=data_L[-1,1]-0.02, s='$Y$', fontsize=12,color='white')   
    
    #BEAUTIFY RIGHT
    # draw gridlines
    ax2.axhline(y=0,color='lightgray',linestyle='-',linewidth=0.3)
    ax2.axvline(x=0,color='lightgray',linestyle='-',linewidth=0.3)
    
    ax2.set_ylabel('$k_y$')
    ax2.set_xlabel('$k_x$')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.plot(0,0,'o', markersize=2,color='white') 
    ax2.text(x=0.01, y=0.005, s='$\Gamma$', fontsize=12,color='white')

    ax2.plot(data_R[-1,0]-0.005,data_R[-1,1]-0.0005,'o', markersize=2,color='white') 
    ax2.text(x=data_R[-1,0]-0.1, y=data_R[-1,1]-0.02, s='$M$', fontsize=12,color='white')
    
    ax2.plot(data_R[-1,0]-0.005,0,'o', markersize=2,color='white') 
    ax2.text(x=data_R[-1,0]-0.1, y=0.005, s='$X$', fontsize=12,color='white')
    
    ax2.plot(0,data_R[-1,1]-0.0005,'o', markersize=2,color='white') 
    ax2.text(x=0.01, y=data_R[-1,1]-0.02, s='$Y$', fontsize=12,color='white')   
    
    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()


# SKEAF (Quantum Oscillations)***********************************************************************

def __read_orbits_file(file):
    """Reads the orbits file generated by SKEAF and returns frequencies and orbits coordinates
    return freqs, orbits"""
    freqs=[]
    orbits=[]
    read_points=False
    diver=False
    lines=open(file,'r')
    for line in lines:
        if re.search('Slice',line):
            l=line.split()
            if '*' in l[10]:
                print('ORBIT FREQ DIVERGING')
                if diver==False:
                    div_freq=freqs[-1]*2
                    diver=True
                freq=div_freq
            else:
                freq=float(l[10])
            freqs=freqs+[freq]
            read_points=False
        elif read_points==True:
            l=line.split()
            p=[float(n) for n in l]
            if first==True:
                orbit=np.array(p)
                first=False
            else:
                orbit=np.vstack((orbit,p))
            if len(orbit)==points:
                orbits=orbits+[orbit]
        elif re.search('Points',line):
            l=line.split()
            points=int(l[2])
        elif re.search('kx.*ky.*kz',line):
            read_points=True
            first=True
    return freqs,orbits

def __read_orbits(files):
    """Reads the orbits file generated by SKEAF and returns frequencies and orbits coordinates
    return freqs, orbits"""
    if type(files)==str:
        freqs,orbits=__read_orbits_file(files)
    else:
        freqs=[]
        orbits=[]
        for file in files:
            f,o=__read_orbits_file(file)
            freqs=freqs+f
            orbits=orbits+o
    return freqs,orbits

def __set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def quantum_osc_orbits(files,scf_file=None,log_scale=False,figsize=None):
    """Plots the maximal orbits detected by SKEAF in order to replicate quantum oscillations measurements of
    the frequencies of Haas–van Alphen oscillations
    files = either a file or a list of files you want to plot.
    scf_file = scf.pwo to read alat parameter
    figsize = (int,int) => Size and shape of the figure"""
    freqs,orbits=__read_orbits(files)
    if scf_file != None:
        alat=__read_alat(scf_file)
        freqs=np.array(freqs)
        freqs=freqs*(1/(alat**2))
    
    fig=plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    cmap = mpl.cm.viridis
    if log_scale==True:
        norm = mpl.colors.LogNorm(vmin=min(freqs), vmax=max(freqs))
    else:
        norm = mpl.colors.Normalize(vmin=min(freqs), vmax=max(freqs))

    for i,orbit in enumerate(orbits):
        color=cmap(norm(freqs[i]))
        ax.plot(orbit[:,0],orbit[:,1],orbit[:,2],linewidth=1,color=color)

    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,shrink=0.5,location='left')
    __set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

def __read_alat(file):
    """Reads alat parameter of an scf.out file"""
    lines=open(file,'r')
    for line in lines:
        if re.search('celldm\(1\)',line):
            l=line.split()
            alat=float(l[1])
    return alat

def __read_skeaf_config(file):
    """reads skeaf's config and gives starting and finishing angles
    return s_phi, s_theta, e_phi, e_theta"""
    lines=open(file,'r')
    for line in lines:
        if re.search('Starting phi',line):
            s_phi=float(line.split()[0])
        elif re.search('Starting theta',line):
            s_theta=float(line.split()[0])
        elif re.search('Ending phi', line):
            e_phi=float(line.split()[0])
        elif re.search('Ending theta',line):
            e_theta=float(line.split()[0])
    return s_phi, s_theta, e_phi, e_theta


def __plot_quantum_osc_freqs(freqs_file,alat,angle_column,s_angle,e_angle,start,end,color=None,legend=None,plot=True):
    """Plots the Haas–van Alphen frequencies agains the angle of the output generated by SKEAF. Neads an scf output in order to read the alat parameter for area conversion. SKEAF expects 1/a.u, while QE bxsf provides 1/alat
    freqs_file = The results_freqvsangle.out file of skeaf output
    alat = alat parameter red from scf.pwo
    angle_column = column to read, either 0 or 1
    s_angle= Start angle red at the config.in file of skeaf
    e_angle= Finishing angle red at the config.in file of skeaf
    start = Angle you want to have as your initial (just for plotting)
    end = Finishing angle you want to have (just for plotting)
    plot = If false it will return the raw quantum oscillation data in case you want to plot (when one single file is plotted)
    """
    data=np.loadtxt(freqs_file,skiprows=1,delimiter=',')
    q_osc=np.zeros([len(data),2])
    q_osc[:,1]=data[:,2]*(1/(alat**2))
    q_osc[:,0]=data[:,angle_column]
    #change x axis so it matches start and end
    q_osc[:,0]=q_osc[:,0]-s_angle
    q_osc[:,0]=q_osc[:,0]*end/(e_angle-s_angle)
    #the plotting of the data
    if plot==True:
        plt.plot(q_osc[:,0],q_osc[:,1],'.',markersize=2,color=color,label=legend)
    return q_osc

def quantum_osc_freqs(files_path,angle,scf_file=None,start=0,end=90,save_as=None,dpi=500,plot=True,figsize=None):
    """Plots the Haas–van Alphen frequencies agains the angle of the output generated by SKEAF. Neads an scf output in order to read the alat parameter for area conversion. SKEAF expects 1/a.u, while QE bxsf provides 1/alat

    files_path = either the path of the dir containing various bands or just the file (freqvsangle.out) to plot
    scf_file = scf.pwo to read alat parameter
    angle = column to read, either 0 or 1
    start = Angle you want to have as your initial (just for plotting)
    end = Finishing angle you want to have (just for plotting)
    plot = If false it will return the raw quantum oscillation data in case you want to plot (when one single file is plotted)
    figsize = (int,int) => Size and shape of the figure

    Returns 
    """
    if scf_file!=None:
        alat=__read_alat(scf_file)
    else:
        alat=1

    #READING CONFIG.IN
    #if just one file
    if os.path.isfile(files_path):
        config=files_path.split('/')
        config[-1]='config.in'
        config_file=''
        for word in config:
            config_file=config_file+word+'/'
        config_file=config_file[:-1]
    #if a directory with various bands
    elif type(files_path) is str:
        files=glob.glob(files_path+'/*')
        config_file=files[0]+'/config.in'
    else:
        print('ERROR I don not understand the files_path')

    s_phi,s_theta,e_phi,e_theta=__read_skeaf_config(config_file)
    print('config in angles:',s_theta,s_phi,e_theta,e_phi)
    if angle==0:
        s_angle=s_theta
        e_angle=e_theta
    elif angle==1:
        s_angle=s_phi
        e_angle=e_phi
    else:
        print('ERROR in angle input')

    #THE PLOTTING
    #if just one file
    if plot==True:
        plt.figure(figsize=figsize)
    if os.path.isfile(files_path):
        q_osc=__plot_quantum_osc_freqs(files_path,alat,angle,s_angle,e_angle,start,end,plot=plot)
    #if a directory with various bands
    else:
        files=glob.glob(files_path+'/*')
        for f in files:
            file=f+'/results_freqvsangle.out'
            q_osc=__plot_quantum_osc_freqs(file,alat,angle,s_angle,e_angle,start,end,plot=plot)
    if plot==True:
        plt.xlim(start,end)
        plt.xlabel('$\phi$ deg')
        if scf_file!=None:
            plt.ylabel('dHvA frequency (kT)')
        else:
            plt.ylabel('dHvA frequency (a.u.)')
        plt.tight_layout()
        if save_as!=None:                             #Saving option
            plt.savefig(save_as, dpi=dpi)
        plt.show()
    if plot==False:
        return q_osc

def quantum_osc_freqs_compare(files_paths,scf_files,legends,angles,start=0,end=90,save_as=None,dpi=500,figsize=None):
    """Plots the Haas–van Alphen frequencies agains the angle of the output generated by SKEAF. Neads an scf output in order to read the alat parameter for area conversion. SKEAF expects 1/a.u, while QE bxsf provides 1/alat

    files_paths = List of paths where (in each of which you have different bands)
    scf_file = list of scf.pwo to read alat parameter
    angles = list of columns to read, either 0 or 1
    start = Angle you want to have as your initial (just for plotting)
    end = Finishing angle you want to have (just for plotting)
    figsize = (int,int) => Size and shape of the figure
    """
    plt.figure(figsize=figsize)
    colors=['C0','C1','C2','C3','C4','C5','C6','C7','C8']
    for i,path in enumerate(files_paths):
        alat=__read_alat(scf_files[i])
        #READING CONFIG.IN
        files=glob.glob(path+'/*')
        config_file=files[0]+'/config.in'
        s_phi,s_theta,e_phi,e_theta=__read_skeaf_config(config_file)
        print('config in angles:',s_theta,s_phi,e_theta,e_phi)
        if angles[i]==0:
            s_angle=s_theta
            e_angle=e_theta
        elif angles[i]==1:
            s_angle=s_phi
            e_angle=e_phi
        else:
            print('ERROR in angle input')

        #THE PLOTTING
        files=glob.glob(path+'/*')
        for j,f in enumerate(files):
            file=f+'/results_freqvsangle.out'
            if j==0:
                __plot_quantum_osc_freqs(file,alat,angles[i],s_angle,e_angle,start,end,color=colors[i],legend=legends[i])
            else:
                __plot_quantum_osc_freqs(file,alat,angles[i],s_angle,e_angle,start,end,color=colors[i])

    plt.xlim(start,end)
    plt.xlabel('$\phi$ deg')
    plt.ylabel('dHvA frequency (kT)')
    plt.legend(markerscale=3)
    plt.tight_layout()

    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=dpi)
    plt.show()
