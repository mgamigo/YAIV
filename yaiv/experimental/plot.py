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

# PLOTTING BANDS**************************************************************************************

def __insert_space_before_minus(string):
    new_string=''
    for i in string:
        if i=='-':
            new_string=new_string+' -'
        else:
            new_string=new_string+i
    return new_string

def __ticks_generator(vectors,ticks,grid=None):
    """From the real vectors, the High Sym points for the path in crystal reciprocal space units and the 
    grid, it generates the positions for the hight sym points (tick_pos) and grid (grid_pos):
    
    vectors=[[a1,a2,a3],...,[c1,c2,c3]]
    ticks=[[tick1x,tick1y,tick1z,100],...,[ticknx,tickny,ticknz,1]]
    grid=[[grid1x,grid1y,grid1z],...,[gridnx,gridny,gridnz]]
    
    returns either tick_pos or tick_pos, grid_pos
    """
    K_vec=np.linalg.inv(vectors).transpose() #reciprocal vectors in columns
    path=0
    tick0=ticks[0][:3]
    ticks_pos=np.array(0)
    for i in range(1,ticks.shape[0]):
        tick1=ticks[i][:3]
        if  ticks[i-1][3]==1:
            tick0=tick1
        else:
            if np.any(grid!= None):
                #print(tick0,tick1)
                for point in grid:
                    dist=__lineseg_dist(point,tick0,tick1)
                    if np.around(dist,decimals=3)==0:
                       # print(point)
                        if np.around(np.linalg.norm(point-tick0),decimals=3)==0:
                            delta=0
                        elif np.around(np.linalg.norm(point-tick1),decimals=3)==0:
                            vector=(tick1-tick0)
                            delta=np.linalg.norm(np.matmul(vector,K_vec))
                        else:
                            vector=(point-tick0)
                            delta=np.linalg.norm(np.matmul(vector,K_vec))
                        try:
                            if np.all(grid_ticks!=(path+delta)):  #grid_ticks are not degenerate
                                grid_ticks=np.append(grid_ticks,path+delta)
                        except NameError:
                            grid_ticks=np.array(path+delta)
                #print()
            vector=(tick1-tick0)
            delta=np.matmul(vector,K_vec)
            path=path+np.linalg.norm(delta)
            ticks_pos=np.append(ticks_pos,path)
            tick0=tick1
    if np.any(grid!= None):
        return ticks_pos, grid_ticks
    else:
        return ticks_pos


def __process_electron_bands(filename,filetype=None,vectors=np.array(None)):
    """Process the bands from various file types with each band separately separated by blank lines
    to a matrix where each column is a band and first column is x axis
    
    filename = File with the bands
    filetype = qe (quantum espresso bands.pwo)
               vaps (VASP EIGENVAL file)
               gnu (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    """
    filetype=filetype.lower()

    if filetype=="gnu" or filetype==None:
        data=np.loadtxt(fname=filename)
        data=data[:,:2]         #select the first two columns to process (for wannier tools)
        rows=1
        position=data[rows,0]
        while position!=0:     #counter will tell me how many points are in the x axis, number of rows
            rows=rows+1
            position=data[rows,0]
        columns=np.int(2*data.shape[0]/rows)
        data=np.reshape(data,(rows,columns),order='F')
        final_columns=np.int(columns/2-1)
        data=np.delete(data,np.s_[0:final_columns],1)

    if filetype=="qe" or filetype==None:
        file=open(filename,'r')
        lines=file.readlines()
        for i,line in enumerate(lines):
            #Grep number of bands
            if re.search('number of Kohn-Sham',line):
                num_bands=int(line.split('=')[1])
            if re.search('number of k points',line):
                num_points=int(line.split()[4])
            if re.search('End of band structure calculation',line):
                results_line=i+1
                break
        data=np.zeros([num_points,num_bands+1])
        data_lines=lines[results_line:]
        dist=0
        i=-1
        j=1
        coord0=np.zeros(3)
        for line in data_lines:
            if re.search('Writing output',line):    #Reading is completed
                break
            elif re.search('bands \(ev\)',line):    #New k_point
                if '-' in line:
                    l=__insert_space_before_minus(line)
                    l=l.split()
                else:
                    l=line.split()
                point1=np.array(l[2:5]).astype(np.float)
                coord1=point1              # Already in reciprocal cartesian coord (not like VASP)
                delta=np.linalg.norm(coord1-coord0)
                if delta>=0.25:                                  #fast fix for broken paths
                    delta=0
                dist=dist+delta
                coord0=coord1
                i=i+1
                j=1
                data[i,0]=dist
            else:                           #Load energies
                line=__insert_space_before_minus(line)
                l=line.split()
                energies=np.array(l).astype(np.float)
                data[i,j:j+len(energies)]=energies
                j=j+len(energies)
       
    elif filetype=="vasp":
        file=open(filename,'r')
        lines=file.readlines()
        num_points=int(lines[5].split()[1])
        num_bands=int(lines[5].split()[2])
        data_lines=lines[7:]

        data=np.zeros([num_points,num_bands+1])
        if vectors.all()!=None:                      #If there is no cell in the input
            K_vec=np.linalg.inv(vectors).transpose() #reciprocal vectors in columns
        else:
            K_vec=np.identity(3)
            print('CAUTIION: There was no cell introduced, threfore distances are wrong')
        dist=0
        i=0
        if K_vec[0].size == 2:
            coord0=np.zeros(2)
            for num in range(0,len(data_lines),num_bands+2):    #load the x position
                line=data_lines[num]
                line=line.split()
                point1=np.array(line).astype(np.float)[0:2]
                coord1=np.matmul(point1,K_vec)
                delta=np.linalg.norm(coord1-coord0)
                if delta>=0.25:                                  #fast fix for broken paths
                    delta=0
                dist=dist+delta
                data[i,0]=dist
                coord0=coord1
                i=i+1
        else:
            coord0  =np.zeros(3)
            for num in range(0,len(data_lines),num_bands+2):    #load the x position
                line=data_lines[num]
                line=line.split()
                point1=np.array(line).astype(np.float)[0:3]
                coord1=np.matmul(point1,K_vec)
                delta=np.linalg.norm(coord1-coord0)
                if delta>=0.25:                                  #fast fix for broken paths
                    delta=0
                dist=dist+delta
                data[i,0]=dist
                coord0=coord1
                i=i+1
        for band in range(1,num_bands+1):                             #load the bands
            i=0
            for num in range(band,len(data_lines),num_bands+2):
                line=data_lines[num]
                line=line.split()
                data[i,band]=line[1]
                i=i+1
        data[:,0]=data[:,0]-data[0,0]
    return data

def __plot_electrons_gnu(file,filetype=None,vectors=np.array(None),ticks=np.array(None),fermi=None,color=None,style=None,marker=None,legend=None,num_elec=None,save_raw_data=None,ax=None):
    """Print the bands given by __process_electron_bands 
    BUT DOES NOT SHOW THE OUTPUT (not plt.show())
    file = Path to the file
    filetype = qe (quantum espresso bands.pwo)
               vaps (VASP EIGENVAL file)
               gnu (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks = np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
    fermi = Fermi energy in order to shift the band structure and put it at zero
    color = string with the color for the bands or 'VC' and num_electrons to use blue/red for valence/conduction
    num_elec= Number of electrons
    style = string with the linestyle (solid, dashed, dotted)
    save_raw_data = 'File to save the plotable data'
    ax = ax in which to plot
    """
    data=__process_electron_bands(file,filetype,vectors)
    if save_raw_data != None:
        np.savetxt(save_raw_data,data)

    if fermi!=None:                       #Fermi energy
        data[:,1:]=data[:,1:]-fermi

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        x_lim=ticks[ticks.shape[0]-1]                #resize x_data for wannier and other codes output
        data[:,0]=data[:,0]*(x_lim/data[:,0].max())

    if color=='VC':
        ax.plot(data[:,0],data[:,1],linestyle=style,marker=marker,linewidth=0.7,color='tab:blue',label=legend)
        ax.plot(data[:,0],data[:,2:num_elec+1],linestyle=style,marker=marker,linewidth=0.7,color='tab:blue')
        ax.plot(data[:,0],data[:,num_elec+1:],linestyle=style,marker=marker,linewidth=0.7,color='tab:red')
    else:
        ax.plot(data[:,0],data[:,1],linestyle=style,marker=marker,linewidth=0.7,color=color,label=legend)
        ax.plot(data[:,0],data[:,2:],linestyle=style,marker=marker,linewidth=0.7,color=color)


    delta_y=data[:,1:].max()-data[:,1:].min()
    #returns max and min values for y and x of the data sample (usefull for plotting)
    #x_min,x_max,y_min,y_max
    #plt.show()
    return [data[:,0].min(),data[:,0].max(),data[:,1:].min(),data[:,1:].max()]

def bands(file,KPATH=None,aux_file=None,title=None,vectors=np.array(None),ticks=np.array(None),labels=None,
               fermi=None,window=None,num_elec=None,color=None,filetype='qe',figsize=None,save_as=None,save_raw_data=None,axis=None):
    """Plots the:
        bands.pwo file of a band calculation in Quantum Espresso
        EIGENVALUES file of a VASP calculation
        bands.dat.gnu file of bands postprocessing (Quantum Espresso)
        band.dat file in Wannier90
        bulkek.dat in Wanniertools

    Minimal plots can be done with just:
        file = Path to the file with bandstructure
        filetype = qe (quantum espresso bands.pwo)
                   vasp (VASP EIGENVAL file)
                   gnu (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)

    Two aditional files can be provide to autocomplete almost everything:
        KPATH = File with PATH and legends for the HSP in the VASP format as provided by topologicalquantumchemistry.fr
        aux_file = A file from which read the Fermi level, number of electrons and structure.
                   In the case of QE this would be an scf.pwo of nscf.pwo
                   In the case of VASP this is the OUTCAR

    However everything may be introduced manually:

    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks = np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
            Ticks in your bandstructure (the usual HSP)
    labels = ["$\Gamma$","$X$","$M$","$\Gamma$"]
    title = 'Your nice and original title for the plot'
    fermi = Fermi energy in order to shift the band structure accordingly
    window = Window of energies to plot around Fermi, it can be a single number or 2
            either window=0.5 or window=[-0.5,0.5] => Same result
    color = Either a color or "VC" to use Valence and Conduction bands with different color
    figsize = (int,int) => Size and shape of the figure
    save_as = 'wathever.format'
    save_raw_data = 'file.dat' The processed data ready to plot
    axis = ax in which to plot, if no axis is present new figure is created
    """
    if KPATH!=None:
        ticks,labels=ut.grep_ticks_labels_KPATH(KPATH)
    if filetype=='qe' and aux_file!=None:
        vectors=ut.grep_vectors(file)

    if aux_file!=None:
        lines=open(aux_file,'r')            #Automatically detected the kind of file, qe or VASP
        for line in lines:
            if re.search('VASP',line):
                kind='vasp'
                break
            if re.search('ESPRESSO',line):
                kind='qe'
                break
        v=ut.grep_vectors(aux_file,filetype=kind)
        f,n=ut.__grep_fermi_and_electrons(aux_file,filetype=kind)
        if fermi==None:
            fermi=f
        if num_elec==None:
            num_elec=n
        if vectors.any()==None:
            vectors=v

    if fermi!=None:
        if window==None:
            window=1
        if num_elec!=None:
            color='VC'


    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis
    limits=__plot_electrons_gnu(file,filetype,vectors,ticks,fermi,color=color,num_elec=num_elec,save_raw_data=save_raw_data,ax=ax)

    ax.set_ylabel('energy (eV)',labelpad=-1)

    if fermi!=None:                       #Fermi energy
        ax.axhline(y=0,color='black',linewidth=0.4)
        if window!=None:                   #Limits y axis
            if type(window) is int or type(window) is float:
                ax.set_ylim(-window,window)
            elif type(window) is list:
                ax.set_ylim(window[0],window[1])
        else:
            delta_y=limits[3]-limits[2]
            ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1)

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        if labels != None :
            ax.set_xticks(ticks,labels)
        else:
            ax.set_xticks(ticks)
        for i in range(1,ticks.shape[0]-1):
            ax.axvline(ticks[i],color='gray',linestyle='--',linewidth=0.4)
    else:
        ax.set_xticks([])

    if title!=None:                             #Title option
        ax.set_title(title)

    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    plt.tight_layout()

    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

def bands_compare(files,KPATH=None,filetypes=None,fermi=None,legends=None,title=None,aux_file=None,vectors=np.array(None),
                  ticks=np.array(None),labels=None,window=1,figsize=None,save_as=None,
                  styles=['-','--','-.',':'],
                  markers=['','','',''],
                  colors=['tab:blue','tab:red','tab:green','tab:orange'],
                  axis=None):
    """Plots the:
        bands.pwo file of a band calculation in Quantum Espresso
        EIGENVALUES file of a VASP calculation
        bands.dat.gnu file of bands postprocessing (Quantum Espresso)
        band.dat file in Wannier90
        bulkek.dat in Wanniertools

    Minimal plots can be done with just:
        files = list of files with bandstructures 
        filetypes = list with the filetype for each file
                    qe (quantum espresso bands.pwo)
                   vasp (VASP EIGENVAL file)
                   gnu (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)

    Four aditional parameters can be provide to autocomplete almost everything:
        KPATH = File with PATH and legends for the HSP in the VASP format as provided by topologicalquantumchemistry.fr
        fermi = list with the different Fermi energies
        legends = List with the legend for each dataset
        aux_file = File from which to read the structure to correctly represent distances in reciprocal space

    However everything may be introduced manually:

    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks = np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
            Ticks in your bandstructure (the usual HSP)
    labels = ["$\Gamma$","$X$","$M$","$\Gamma$"]
    title = 'Your nice and original title for the plot'
    window = Window of energies to plot around Fermi, it can be a single number or 2
            either window=0.5 or window=[-0.5,0.5] => Same result
    colors = list with colors
    styles = list with style lines (solid, dashed, dotted...)
    figsize = (int,int) => Size and shape of the figure
    save_as = 'wathever.format'
    axis = ax in which to plot, if no axis is present new figure is created
    """
    styles=styles*4
    colors=colors*4

    if fermi==None:
        fermi=np.zeros(len(files))
    if legends==None:
        legends=['Data ' + str(n+1) for n in range(len(files))]
    if filetypes==None:
        filetypes=['qe' for n in range(len(files))]
    if KPATH!=None:
        ticks,labels=ut.grep_ticks_labels_KPATH(KPATH)
    for i,kind in enumerate(filetypes):            #Automatically read vectors if possible
        if kind=='qe' and aux_file==None and vectors.any()==None:
            vectors=ut.grep_vectors(files[i],filetype='qe')
            break
    if aux_file!=None:
        lines=open(aux_file,'r')            #Automatically detected the kind of file, qe or VASP
        for line in lines:
            if re.search('VASP',line):
                kind='vasp'
                break
            if re.search('ESPRESSO',line):
                kind='qe'
                break
        v=ut.grep_vectors(aux_file,filetype=kind)
        if vectors.any()==None:
            vectors=v
    
    if axis == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = axis
    for num in range(len(files)):
        data_limits=__plot_electrons_gnu(files[num],filetypes[num],vectors,ticks,fermi=fermi[num],color=colors[num]
                                       ,style=styles[num],marker=markers[num],legend=legends[num],ax=ax)
        if num==0:
            limits=data_limits
        else:
            limits[0]=min(data_limits[0],limits[0])
            limits[1]=max(data_limits[1],limits[1])
            limits[2]=min(data_limits[2],limits[2])
            limits[3]=max(data_limits[3],limits[3])
        
    ax.set_ylabel('energy (eV)',labelpad=-1)
    ax.axhline(y=0,color='black',linewidth=0.4) #Fermi energy (mandatory, if not comparison has no sense)
    
    if type(window) is int or type(window) is float: #window of energyes to plot
        ax.set_ylim(-window,window)
    elif type(window) is list:
        ax.set_ylim(window[0],window[1])
    else:
        delta_y=limits[3]-limits[2]
        ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1)
    
    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        if labels != None :
            ax.set_xticks(ticks,labels)
        else:
            ax.set_xticks(ticks)
        for i in range(1,ticks.shape[0]-1):
            ax.axvline(ticks[i],color='gray',linestyle='--',linewidth=0.4)
    else:
        ax.set_xticks([])
    
    if title!=None:                             #Title option
        ax.set_title(title)
    
    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    
    ax.legend()
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

def __process_phonon_bands(gnu_file,matdyn_in=None):
    if matdyn_in==None:
        data=np.loadtxt(fname=gnu_file)
    else:
        #LOAD THE PATH FROM MATDYN FILE
        KPATH=open(matdyn_in)
        QE_path=np.zeros(0)
        num_q=0
        path_section=False
        for line in KPATH:
            if path_section==True:
                if num_q==0:
                    num_q=int(line)
                    i=0
                elif i<num_q:
                    if len(line.split())!=0:
                        X=float(line.split()[0])
                        Y=float(line.split()[1])
                        Z=float(line.split()[2])
                        points=int(line.split()[3])
                        q1=np.array([X,Y,Z,points])
                        if len(QE_path)==0:
                            QE_path=q1
                        else:
                            QE_path=np.vstack((QE_path,q1))
                    i=i+1
            if re.search('/',line) and 'matdyn' in matdyn_in:
                path_section=True

        #SELECT THE LINES WHERE THE PATH IS SPLITED
        lines=[]
        line=0
        for i in range(num_q):
            if QE_path[i,3]==1:
                lines=lines+[line]
            line=line+int(QE_path[i,3])

        #LOAD THE DATA AND CORRECT THE SPLITS TO MAKE IT "CONTINOUS"
        data=np.loadtxt(gnu_file)
        for line in lines:
            if line < data.shape[0]-1:
                data[line+1:,0]=data[line+1:,0]-(data[line+1,0]-data[line,0])
    return data

# PLOTTING PHONONS**************************************************************************************

def __lineseg_dist(p, a, b):
    """Function lineseg_dist returns the distance the distance from point p to line segment [a,b]. p, a and b are np.arrays."""
    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))
    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)
    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])
    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))


def __plot_phonons_output(file,linewidth,vectors=np.array(None),ticks=np.array(None),
                        color=None,style=None,legend=None,matdyn_in=None,ax=None):
    """Print the phonons.freq.gp file of matdyn output (Quantum Espresso)
    BUT DOES NOT SHOW THE OUTPUT (not plt.show())
    plot_phonons(file,real_vecs,ticks)
    file=Path to the file
    vectors=np.array([[a1,a2,a3],...,[c1,c2,c3]])
    ticks=np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
    color = string with the color for the bands
    style = string with the linestyle (solid, dashed, dotted)
    legend = legend to add for the data set
    matdyn_in = matdyn file for splitted paths, it will correct the path to make it "continous"
    ax = ax in which to plot
    """
    data=__process_phonon_bands(file,matdyn_in)

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        x_lim=ticks[ticks.shape[0]-1]                #resize x_data for wannier and other codes output
        data[:,0]=data[:,0]*(x_lim/data[:,0].max())

    ax.plot(data[:,0],data[:,1],linestyle=style,linewidth=linewidth,color=color,label=legend)
    ax.plot(data[:,0],data[:,2:],linestyle=style,linewidth=linewidth,color=color)

    return [data[:,0].min(),data[:,0].max(),data[:,1:].min(),data[:,1:].max()]

def phonons(file,KPATH=None,ph_out=None,matdyn_in=None,title=None,grid=True,vectors=np.array(None),
                ticks=np.array(None),labels=None,save_as=None,figsize=None,color=None,linewidth=0.7,axis=None):
    """Plots phonon spectra provided by Quantum Espresso output 
    (it supports discontinous paths and highlights the computed points)
    Minimal plots can be done with just:
        file = Path to the freq.gp file provided by matdyn

    Three aditional files can be provide to autocomplete almost everything:
        KPATH = File with PATH and legends for the HSP in the VASP format as provided by topologicalquantumchemistry.fr
        ph_out = ph.x output (Needed to read lattice parameters and grid where the phonons are computed)
        matdyn_in = matdyn subroutine input (in order to read the KPATH and correct for discontinous paths)

    However everything may be introduced manually:
    title = 'Your nice and original title for the plot'
    grid = Bolean (Whether you want the grid to be highlighted in pink)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks=np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
    labels=["$\Gamma$","$X$","$M$","$\Gamma$"]
    save_as='wathever.format'
    figsize = (int,int) => Size and shape of the figure
    color = color for your plot
    linewidth = linewidth of your plot
    axis = ax in which to plot, if no axis is present new figure is created
    """

    if KPATH!=None:
        ticks,labels=ut.grep_ticks_labels_KPATH(KPATH)
    if ph_out!=None:
        v=ut.grep_vectors(ph_out,filetype='qe')
        if vectors.any()==None:
            vectors=v
    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    limits=__plot_phonons_output(file,linewidth,vectors,ticks,color=color,matdyn_in=matdyn_in,ax=ax)

    ax.set_ylabel('Frequency $(cm^{-1})$')
    ax.axhline(y=0,color='gray',linestyle='--',linewidth=0.4)

    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    delta_y=limits[3]-limits[2]
    ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1) #Limits in the y axis

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        if ph_out!=None and grid==True:
            grid_points=ut.grep_grid_points(ph_out,expanded=True,decimals=15) #to correctly find distances
            ticks,grid=__ticks_generator(vectors,ticks,grid_points)
            ax.set_xticks(ticks,labels)
            for point in grid:
                for i in range(len(ticks)):
                    if point==ticks[i]:
                        ticks=np.delete(ticks,i)
                        break
            for i in range(len(grid)):
                ax.axvline(grid[i],color='deeppink',linestyle='-.',linewidth=0.5)
        else:
            ticks=__ticks_generator(vectors,ticks)
            if labels != None:
                ax.set_xticks(ticks,labels)
            else:
                ax.set_xticks(ticks)
        for i in range(len(ticks)):
            ax.axvline(ticks[i],color='gray',linestyle='--',linewidth=0.4)
    else:
        ax.set_xticks([])

    if title!=None:                             #Title option
        ax.set_title(title)

    plt.tight_layout()
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

def phonons_compare(files,KPATH=None,ph_outs=None,matdyn_in=None,legends=None,title=None,grid=True,
                         vectors=np.array(None),ticks=np.array(None),labels=None,save_as=None,
                         colors=['tab:blue','tab:red','tab:green','tab:orange'],
                         styles=['solid','dashed','dashdot','dotted'],linewidth=0.7,figsize=None,axis=None):
    """Plots phonon spectra provided by Quantum Espresso output 
    (it supports discontinous paths and highlights the computed points)
    Minimal plots can be done with just:
        files = List of paths to the freq.gp file provided by matdyn

    Four aditional parameters can be provide to autocomplete almost everything:
        KPATH = File with PATH and legends for the HSP in the VASP format as provided by topologicalquantumchemistry.fr
        ph_out = List of ph.x output files (Needed to read lattice parameters and grid where the phonons are computed)
        matdyn_in = matdyn subroutine input (in order to read the KPATH and correct for discontinous paths)
        legends = List with the legend for each dataset

    However everything may be introduced manually:
    title = 'Your nice and original title for the plot'
    grid = Bolean (Whether you want the grid to be highlighted in pink)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks=np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
    labels=["$\Gamma$","$X$","$M$","$\Gamma$"]
    save_as='wathever.format'
    figsize = (int,int) => Size and shape of the figure
    linewidth = linewidth of your plot
    colors = list with colors
    styles = list with style lines (solid, dashed, dotted...)
    axis = ax in which to plot, if no axis is present new figure is created
    """
    styles=styles*4
    colors=colors*4

    if legends==None:
        legends=['Data ' + str(n+1) for n in range(len(files))]
    if KPATH!=None:
        ticks,labels=ut.grep_ticks_labels_KPATH(KPATH)
    if ph_outs!=None:
        v=ut.grep_vectors(ph_outs[0],filetype='qe')
        if vectors.any()==None:
            vectors=v


    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for num in range(len(files)):
        data_limits=__plot_phonons_output(files[num],linewidth,vectors,ticks,color=colors[num]
                                       ,style=styles[num],legend=legends[num],matdyn_in=matdyn_in,ax=ax)
        if num==0:
            limits=data_limits
        else:
            limits[0]=min(data_limits[0],limits[0])
            limits[1]=max(data_limits[1],limits[1])
            limits[2]=min(data_limits[2],limits[2])
            limits[3]=max(data_limits[3],limits[3])

    ax.set_ylabel('Frequency $(cm^{-1})$')
    ax.axhline(y=0,color='gray',linestyle='--',linewidth=0.4)

    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    delta_y=limits[3]-limits[2]
    ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1) #Limits in the y axis

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        if ph_outs!=None and grid==True:
            grid_points=ut.grep_grid_points(ph_outs[0],expanded=True,decimals=15)
            path=ticks
            ticks,grid=__ticks_generator(vectors,ticks,grid_points)
            ax.set_xticks(ticks,labels)
            for num in range(len(ph_outs)):
                grid_points=ut.grep_grid_points(ph_outs[num],expanded=True,decimals=15)
                grid=__ticks_generator(vectors,path,grid_points)[1]
                for point in grid:
                    for i in range(len(ticks)):
                        if point==ticks[i]:
                            ticks=np.delete(ticks,i)
                            break
                for i in range(len(grid)):
                    if len(ph_outs)==1:
                        ax.axvline(grid[i],color='deeppink',linestyle='-.',linewidth=0.5)
                    else:
                        ax.axvline(grid[i],color=colors[num],linestyle=styles[num],linewidth=0.5)
        else:
            ticks=__ticks_generator(vectors,ticks)
            if labels != None:
                ax.set_xticks(ticks,labels)
            else:
                ax.set_xticks(ticks)
        for tick in ticks:
            ax.axvline(tick,color='gray',linestyle='--',linewidth=0.4)
    else:
        ax.set_xticks([])

    if title!=None:                             #Title option
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

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

# WANNIER90****************************************************************************************

def __read_w90out(file):
    """reads w90out file in order to analyze the minimization
    outputs:
    return total_spread,spread_delta,wan_spreads,wan_centers,disentang,dis_delta
    """
    lines=open(file,'r')
    read_dis=False
    read_wan=False
    read_step=False
    iteration=-1
    disentang=0
    dis_delta=0
    for i,line in enumerate(lines):

        if line == '\n':
            if read_dis==True:
                dis_delta=float(l[3])
            read_dis=False
            read_step=False

        #READ disentanglement
        if read_dis==True:
            l=line.split()
            if l[0]=='1':
                i=int(l[0])
                O_i=float(l[1])
                disentang=np.array([i,O_i])
                read_step=True
            elif read_step==True:
                i=int(l[0])
                O_i=float(l[1])
                disentang=np.vstack([disentang,np.array([i,O_i])])

        #READ wannierization
        if read_wan==True:
            l=line.split()

            #Counts in which step you are
            if len(l)>5 and l[4]=='1':
                if iteration==0:
                    total_spread=new_t_spread
                    deltas=new_delta
                    wan_centers=new_centers
                    wan_spreads=new_spreads
                    wan_num=len(new_spreads)
                elif iteration>0:
                    total_spread=np.vstack([total_spread,new_t_spread])
                    deltas=np.vstack([deltas,new_delta])
                    if len(new_spreads) == wan_num:     #Check that the step is complete
                        wan_centers=np.vstack([wan_centers,new_centers])
                        wan_spreads=np.vstack([wan_spreads,new_spreads])
                    else:
                        iteration=iteration-1
                iteration=iteration+1
                new_centers=iteration
                new_spreads=iteration
            #Reads individual WF
            if re.search('WF centre and spread',line):
                spread=float(l[-1])
                if l[-5][0]=='(':
                    l[-5]=l[-5][1:]
                centre=np.array([float(l[-5][:-1]),float(l[-4][:-1]),float(l[-3])])
                new_spreads=np.hstack([new_spreads,spread])
                new_centers=np.hstack([new_centers,centre])
            if re.search('<-- CONV',line):
                new_t_spread=np.array([iteration,float(l[3])])
                new_delta=np.array([iteration,float(l[1])])

        if re.search('Extraction of optimally',line):
            read_dis=True
        if re.search('Initial State',line):
            read_wan=True
            Initial=True
        else:
            Initial=False
        if re.search('WANNIER90',line):
            read_wan=False

        if re.search('Number of Wannier',line):
            l=line.split()
            num_wan=int(l[6])

    spread_delta=deltas[-1,1]

    return total_spread,spread_delta,wan_spreads,wan_centers,disentang,dis_delta

def w90_wannierization(file,save_as=None):
    """ Plots the evolution of the wannierization of the w90 routine (total spread).
    Uses the .wout file as input
    """
    total_spread,spread_delta,wan_spreads,wan_centers,disentang,dis_delta=__read_w90out(file)    
    plt.figure()
    data=total_spread
    delta=np.hstack([0,np.diff(data[:,1])])
    data=np.insert(data,2,delta,axis=1)
    plt.plot(data[:,0],data[:,1])
    #plt.errorbar(data[:,0],data[:,1],yerr=data[:,2],elinewidth=0.4)

    plt.text(data[-1,0]*0.8,data[0,1]-(data[0,1]-data[-1,1])*0.07
             , 'Final $\Delta\Omega$ ='+str(spread_delta)
             , size=12, ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       )
             )
    plt.grid(axis='both',color='0.95')   
    plt.title('Wannerization')
    plt.ylabel('$\Omega \ (\AA^2)$')
    plt.xlabel('Iteration')

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()

def w90_disentanglement(file,save_as=None):
    """ Plots the disentanglement procedure of the w90 routine (the gauge invariant spread).
    Uses the .wout file as input
    """
    total_spread,spread_delta,wan_spreads,wan_centers,disentang,dis_delta=__read_w90out(file)    
    plt.figure()
    data=disentang
    plt.plot(data[:,0],data[:,1])

    plt.text(data[-1,0]*0.8,data[0,1]-(data[0,1]-data[-1,1])*0.07
             , 'Final $\Delta\Omega_I$ ='+str(dis_delta)
             , size=12, ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       )
             )
    
    plt.grid(axis='both',color='0.95')   
    plt.title('Disentanglement')
    plt.ylabel('$\Omega_I \ (\AA^2)$')
    plt.xlabel('Iteration')

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()
    
    
def w90_centers(file,save_as=None):
    """ Plots the evolution of the wannier centers of the w90 routine.
    Uses the .wout file as input
    """
    total_spread,spread_delta,wan_spreads,wan_centers,disentang,dis_delta=__read_w90out(file)    
    plt.figure()
    data=wan_centers
    center_num=len(wan_centers[0,:])-1
    wan_num=len(wan_spreads[0,:])-1

    plt.plot(data[:,0],data[:,1],color='C0',linewidth=0.7,label='x')
    plt.plot(data[:,0],data[:,2],color='C1',linewidth=0.7,label='y')
    plt.plot(data[:,0],data[:,3],color='C2',linewidth=0.7,label='z')


    for n in range(4,center_num+1,3):
        plt.plot(data[:,0],data[:,n],color='C0',linewidth=0.7)
    for n in range(5,center_num+1,3):
        plt.plot(data[:,0],data[:,n],color='C1',linewidth=0.7)
    for n in range(6,center_num+1,3):
        plt.plot(data[:,0],data[:,n],color='C2',linewidth=0.7)        

    plt.grid(axis='both',color='0.95')   
    plt.title('WFs centers ('+str(wan_num)+')')
    plt.ylabel('Center coord. ($\AA$)')
    plt.xlabel('Iteration')
    plt.legend()

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()
    
def w90_spreads(file,save_as=None):
    """ Plots the evolution of the individual spreads of the w90 routine.
    Uses the .wout file as input
    """
    total_spread,spread_delta,wan_spreads,wan_centers,disentang,dis_delta=__read_w90out(file)    
    plt.figure()
    data=wan_spreads
    wan_num=len(wan_spreads[0,:])-1
    for n in range(1,wan_num+1):
        plt.plot(data[:,0],data[:,n],color='C0',linewidth=0.7)
    
    plt.grid(axis='both',color='0.95')   
    plt.title('WFs spreads ('+str(wan_num)+')')
    plt.ylabel('$\Omega \ (\AA^2)$')
    plt.xlabel('Iteration')

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()

def w90(file,save_as=None):
    """ Plots the evolution of Wannierization routine.
    Uses the .wout file as input
    ax1: Wannierization of the w90 routine (total spread)
    ax2: Disentanglement procedure of the w90 routine (the gauge invariant spread)
    ax3: Evolution of the individual spreads of the w90 routine
    ax4: Evolution of the wannier centers of the w90 routine
    """
    fig = plt.figure(figsize=(9.0, 7.0)) #size of our plots (in inches)
    axes1=fig.add_subplot(2,2,1) #rows of sublots, colums of subplots, place of this figure
    axes2=fig.add_subplot(2,2,2)
    axes3=fig.add_subplot(2,2,3)
    axes4=fig.add_subplot(2,2,4)
    
    total_spread,spread_delta,wan_spreads,wan_centers,disentang,dis_delta=__read_w90out(file)

    data=total_spread
    axes1.plot(data[:,0],data[:,1])
    axes1.text(data[-1,0]*0.75,data[0,1]-(data[0,1]-data[-1,1])*0.07
             , 'Final $\Delta\Omega$ ='+str(spread_delta)
             , size=10, ha="center", va="center",
             bbox=dict(boxstyle="round",
                       ec=(1., 0.5, 0.5),
                       fc=(1., 0.8, 0.8),
                       )
             )
    axes1.grid(axis='both',color='0.95')   
    axes1.set_title('Total Spread')
    axes1.set_ylabel('$\Omega \ (\AA^2)$')
    axes1.set_xlabel('Iteration')
    
    if dis_delta!=0:
        data=disentang
        axes2.plot(data[:,0],data[:,1])
        axes2.text(data[-1,0]*0.75,data[0,1]-(data[0,1]-data[-1,1])*0.07
                 , 'Final $\Delta\Omega_I$ ='+str(dis_delta)
                 , size=10, ha="center", va="center",
                 bbox=dict(boxstyle="round",
                           ec=(1., 0.5, 0.5),
                           fc=(1., 0.8, 0.8),
                           )
                 )
        axes2.grid(axis='both',color='0.95')   
        axes2.set_title('Disentanglement')
        axes2.set_ylabel('$\Omega_I \ (\AA^2)$')
        axes2.set_xlabel('Iteration')
    else:
        axes2.set_title('No Disentanglement')
    
    data=wan_spreads
    wan_num=len(wan_spreads[0,:])-1
    for n in range(1,wan_num+1):
        axes3.plot(data[:,0],data[:,n],color='C0',linewidth=0.7)
    axes3.grid(axis='both',color='0.95')   
    axes3.set_title('WFs spreads')
    axes3.set_ylabel('$\Omega \ (\AA^2)$')
    axes3.set_xlabel('Iteration')
    
    
    data=wan_centers
    center_num=len(wan_centers[0,:])-1
    axes4.plot(data[:,0],data[:,1],color='C0',linewidth=0.7,label='x')
    axes4.plot(data[:,0],data[:,2],color='C1',linewidth=0.7,label='y')
    axes4.plot(data[:,0],data[:,3],color='C2',linewidth=0.7,label='z')
    for n in range(4,center_num+1,3):
        axes4.plot(data[:,0],data[:,n],color='C0',linewidth=0.7)
    for n in range(5,center_num+1,3):
        axes4.plot(data[:,0],data[:,n],color='C1',linewidth=0.7)
    for n in range(6,center_num+1,3):
        axes4.plot(data[:,0],data[:,n],color='C2',linewidth=0.7)        
    axes4.grid(axis='both',color='0.95')   
    axes4.set_title('WFs centers')
    axes4.set_ylabel('Center coord. ($\AA$)')
    axes4.set_xlabel('Iteration')
    axes4.legend()
    
    fig.suptitle('Wannierization ('+str(wan_num)+' WF)', fontsize=16)
    plt.tight_layout()
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


def __get_brillouin_zone_3d(cell):
    """
    Generate the Brillouin Zone of a given cell. The BZ is the Wigner-Seitz cell
    of the reciprocal lattice, which can be constructed by Voronoi decomposition
    to the reciprocal lattice.  A Voronoi diagram is a subdivision of the space
    into the nearest neighborhoods of a given set of points. 

    https://en.wikipedia.org/wiki/Wigner%E2%80%93Seitz_cell
    https://docs.scipy.org/doc/scipy/reference/tutorial/spatial.html#voronoi-diagrams
    """

    cell = np.asarray(cell, dtype=float)
    assert cell.shape == (3, 3)

    px, py, pz = np.tensordot(cell, np.mgrid[-1:2, -1:2, -1:2], axes=[0, 0])
    points = np.c_[px.ravel(), py.ravel(), pz.ravel()]

    from scipy.spatial import Voronoi
    vor = Voronoi(points)

    bz_facets = []
    bz_ridges = []
    bz_vertices = []

    # for rid in vor.ridge_vertices:
    #     if( np.all(np.array(rid) >= 0) ):
    #         bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
    #         bz_facets.append(vor.vertices[rid])

    for pid, rid in zip(vor.ridge_points, vor.ridge_vertices):
        # WHY 13 ????
        # The Voronoi ridges/facets are perpendicular to the lines drawn between the
        # input points. The 14th input point is [0, 0, 0].
        if(pid[0] == 13 or pid[1] == 13):
            bz_ridges.append(vor.vertices[np.r_[rid, [rid[0]]]])
            bz_facets.append(vor.vertices[rid])
            bz_vertices += rid

    bz_vertices = list(set(bz_vertices))

    return vor.vertices[bz_vertices], bz_ridges, bz_facets


def brillouin_zone_3d(cell,ax,basis=True,sides=True,line_width=1,reciprocal=True):
    """
    Plot Brillouin zone in 3d ax, it uses as input the real space cell or a QE output file containing it
    ax = ax over with to plot (ax = fig.add_subplot...)
    basis = Whether to plot the basis
    sides = Whether to plot or not the sides
    line_width = Line width for the edges
    reciprocal = Whether or not transform to reciprocal coordinates
    """
    if type(cell)==str:
        cell=ut.grep_vectors(cell)
    if reciprocal==True:
        K_vec=ut.K_basis(cell)
    else:
        K_vec=cell
    #Plot K_vec
    if basis==True:
        ax.plot([0,K_vec[0][0]],[0,K_vec[0][1]],[0,K_vec[0][2]],color='red')
        ax.plot([0,K_vec[1][0]],[0,K_vec[1][1]],[0,K_vec[1][2]],color='green')
        ax.plot([0,K_vec[2][0]],[0,K_vec[2][1]],[0,K_vec[2][2]],color='blue')
        ax.scatter(K_vec[0][0],K_vec[0][1],K_vec[0][2], color = 'red', marker = "^")
        ax.scatter(K_vec[1][0],K_vec[1][1],K_vec[1][2], color = 'green', marker = "^")
        ax.scatter(K_vec[2][0],K_vec[2][1],K_vec[2][2], color = 'blue', marker = "^")
    # Plot BZ
    v, e, f = __get_brillouin_zone_3d(K_vec)
    for xx in e:
        ax.plot(xx[:, 0], xx[:, 1], xx[:, 2], color='k', lw=line_width)
    if sides==True:
        ax.add_collection3d(Poly3DCollection(e, 
             facecolors='cyan', linewidths=0, edgecolors='black', alpha=.05))
