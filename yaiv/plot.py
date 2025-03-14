#PYTHON module for ploting 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import re
import glob
import spglib as spg

import yaiv.utils as ut
import yaiv.constants as cons
import yaiv.cell_analyzer as cell

# PLOTTING BANDS----------------------------------------------------------------

def __insert_space_before_minus(string):
    """A simple tool to reformat strings and improve reading"""
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
    K_vec=ut.K_basis(vectors)
    path=0
    tick0=ticks[0][:3]
    ticks_pos=np.array(0)
    for i in range(1,ticks.shape[0]):
        tick1=ticks[i][:3]
        if  ticks[i-1][3]==1:
            tick0=tick1
        else:
            if np.any(grid!= None):
                for point in grid:
                    dist=__lineseg_dist(point,tick0,tick1)
                    if np.around(dist,decimals=3)==0:
                        if np.around(np.linalg.norm(point-tick0),decimals=3)==0:
                            delta=0
                        elif np.around(np.linalg.norm(point-tick1),decimals=3)==0:
                            vector=(tick1-tick0)
                            delta=np.linalg.norm(ut.cryst2cartesian(vector,K_vec))
                        else:
                            vector=(point-tick0)
                            delta=np.linalg.norm(ut.cryst2cartesian(vector,K_vec))
                        try:
                            if np.all(grid_ticks!=(path+delta)):  #grid_ticks are not degenerate
                                grid_ticks=np.append(grid_ticks,path+delta)
                        except NameError:
                            grid_ticks=np.array(path+delta)
            vector=(tick1-tick0)
            delta=ut.cryst2cartesian(vector,K_vec)
            path=path+np.linalg.norm(delta)
            ticks_pos=np.append(ticks_pos,path)
            tick0=tick1
    if np.any(grid!= None):
        return ticks_pos, grid_ticks
    else:
        return ticks_pos

def DOS(file,fermi='auto',smearing=0.02,window=[-5,5],steps=500,precision=3,filetype=None,
        title=None,figsize=None,reverse=False,legend=None,color='black',save_as=None,axis=None):
    """
    Plots the Density Of States

    file = File from which to extract the DOS (scf, nscf, bands)
    fermi = Fermi level to shift accordingly
    smearing = Smearing of your normal distribution around each energy
    window = energy window in which to compute the DOS
    steps = Number of values for which to compute the DOS
    precision = Truncation of your normal distrib (truncated from precision*smearing)
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
    title = 'Your nice and original title for the plot'
    figsize = (int,int) => Size and shape of the figure
    reverse = Bolean switching the DOS and energies axis
    save_as = 'wathever.format'
    axis = ax in which to plot, if no axis is present new figure is created
    """
    if filetype == None:
        file = ut.file(file)
    else:
        file = ut.file(file,filetype)
    if fermi == 'auto':
        fermi=ut.grep_fermi(file.file,silent=True)
        if fermi == None:
            fermi=0
    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis
    E,D=ut.grep_DOS(file.file,fermi,smearing,window,steps,precision,filetype)
    if reverse==False:
        ax.plot(E,D,'-',color=color,label=legend)
        ax.set_xlim(E[0],E[-1])
        ax.set_xlabel('energy (eV)')
        ax.set_ylabel('DOS (a.u)')
        ax.set_yticks([])
        ax.set_ylim(0,np.max(D)*1.1)
        if fermi!=None:                       #Fermi energy
            ax.axvline(x=0,color='black',linewidth=0.4)
    else:
        ax.plot(D,E,'-',color=color,label=legend)
        ax.set_ylim(E[0],E[-1])
        ax.set_ylabel('energy (eV)')
        ax.set_xlabel('DOS (a.u)')
        ax.set_xticks([])
        ax.set_xlim(0,np.max(D)*1.05)
        if fermi!=None:                       #Fermi energy
            ax.axhline(y=0,color='black',linewidth=0.4)
    if title!=None:                             #Title option
        ax.set_title(title)
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()


def DOS_projected(file,proj_file,fermi='auto',smearing=0.02,window=[-5,5],steps=500,precision=3,filetype=None,proj_filetype=None,
                  species=None,atoms=None,l=None,j=None,mj=None,title=None,figsize=None,reverse=False,legend=None,color='black',
                  save_as=None,axis=None,silent=False,fill=True,alpha=0.5,linewidth=1.0,symprec=1e-5):
    """
    Plots the projected Density Of States

    file = File from which to extract the DOS (scf, nscf, bands)
    proj_file = File with the projected bands or output from grep_DOS_projected:
                    qe_proj_out (quantum espresso proj.pwo)
                    procar (VASP PROCAR file)
    fermi = Fermi level to shift accordingly
    smearing = Smearing of your normal distribution around each energy
    window = energy window in which to compute the DOS
    steps = Number of values for which to compute the DOS
    precision = Truncation of your normal distrib (truncated from precision*smearing)
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               vaps (VASP OUTCAR file)
               eigenval (VASP EIGENVAL file)
    species = list of atomic species ['Bi','Se'...]
    atoms = list with atoms index [1,2...]
    l = list of orbital atomic numbers:
        qe: [0, 1, 2]
        vasp: ['s','px','py','dxz']  (as written in POSCAR)
    j = total angular mometum. (qe only)
    mj = m_j state. (qe only)
    title = 'Your nice and original title for the plot'
    figsize = (int,int) => Size and shape of the figure
    reverse = Bolean switching the DOS and energies axis
    legend = label for the plot
    color = matplotlib color for the line, or list of colors for different lines
    save_as = 'wathever.format'
    axis = ax in which to plot, if no axis is present new figure is created
    silent = Boolean controling whether you want text output
    fill = Boolean controling whether you want to fill the DOS
    alpha = Opaciety of the fill
    linewidht = linewidth of the lines
    symprec = symprec for spglib detection of wyckoff positions
    """
    if filetype == None:
        file = ut.file(file)
    else:
        file = ut.file(file,filetype)
    if fermi == 'auto':
        fermi=ut.grep_fermi(file.file,silent=True)
        if fermi == None:
            fermi=0
    if type(window) is int or type(window) is float:
        window=[-window,window]
    if type(proj_file)!=str:
        E,DOSs,LABELS = proj_file
    else:
        E,DOSs,LABELS = file.grep_DOS_projected(proj_file,fermi=fermi,smearing=smearing,window=window,steps=steps,
                                            precision=precision,species=species,atoms=atoms,l=l,j=j,mj=mj,symprec=symprec,silent=silent)
    if type(LABELS)!= list:
        DOSs=[DOSs]
        LABELS=[legend]
    if type(color)!= list:
        color = [color]+10*list(mcolors.TABLEAU_COLORS.values())
    
    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis
    if reverse==False:
        if len(LABELS)==1 and fill==True:
            ax.plot(E,DOSs[0],'-',label=LABELS[0],color=color[0],linewidth=linewidth)
            ax.fill_between(E,DOSs[0],color=color[0],alpha=alpha)
        else:
            ax.plot(E,DOSs[0],'-',label=LABELS[0],color=color[0],linewidth=linewidth)
        for i,L in enumerate(LABELS[1:]):
            if fill==True:
                ax.plot(E,DOSs[i+1],'-',color=color[i+1],linewidth=linewidth)
                ax.fill_between(E,DOSs[i+1],'-',color=color[i+1],label=L,alpha=alpha)
            else:
                ax.plot(E,DOSs[i+1],'-',color=color[i+1],label=L)
        ax.set_xlim(E[0],E[-1])
        ax.set_yticks([])
        ax.set_xlabel('energy (eV)')
        ax.set_ylabel('DOS (a.u)')
        ax.set_ylim(0,np.max(DOSs[0])*1.1)
        if fermi!=None:                       #Fermi energy
            ax.axvline(x=0,color='black',linewidth=0.4)
    else:
        if len(LABELS)==1 and fill==True:
            ax.plot(DOSs[0],E,'-',label=LABELS[0],color=color[0],linewidth=linewidth)
            ax.fill_betweenx(DOSs[0],E,color=color[0],alpha=alpha)
        else:
            ax.plot(DOSs[0],E,'-',label=LABELS[0],color=color[0],linewidth=linewidth)
        for i,L in enumerate(LABELS[1:]):
            if fill==True:
                ax.plot(DOSs[i+1],E,'-',color=color[i+1],linewidth=linewidth)
                ax.fill_betweenx(E,DOSs[i+1],color=color[i+1],label=L,alpha=alpha)
            else:
                ax.plot(DOSs[i+1],E,'-',label=L,color=color[i+1],linewidth=linewidth)
        ax.set_ylim(E[0],E[-1])
        ax.set_ylabel('energy (eV)')
        ax.set_xlabel('DOS (a.u)')
        ax.set_xticks([])
        MAX=np.max(DOSs[:,np.where((E>=window[0]) & (E<=window[1]))[0]])
        ax.set_xlim(0,MAX*1.05)
        ax.set_ylim(window[0],window[1])
        
        if fermi!=None:                       #Fermi energy
            ax.axhline(y=0,color='black',linewidth=0.4)
    ax.legend(loc='upper right',fontsize='small')
    if title!=None:                             #Title option
        ax.set_title(title)
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

def __process_electron_bands(filename,filetype=None,vectors=np.array(None),IgnoreWeight=True):
    """Process the bands from various file types with each band separately separated by blank lines
    to a matrix where each column is a band and first column is x axis
    
    filename = File with the bands
    filetype = qe (quantum espresso bands.pwo)
               eigenval (VASP EIGENVAL file)
               data (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    IgnoreWeight = Boolean controlling whether points with non-zero weight would be ignored
    """
    filetype=filetype.lower()

    if filetype=="data" or filetype==None:
        data=np.loadtxt(fname=filename)
        data=data[:,:2]         #select the first two columns to process (for wannier tools)
        rows=1
        position=data[rows,0]
        while position!=0:     #counter will tell me how many points are in the x axis, number of rows
            rows=rows+1
            position=data[rows,0]
        columns=np.int_(2*data.shape[0]/rows)
        data=np.reshape(data,(rows,columns),order='F')
        final_columns=np.int_(columns/2-1)
        data=np.delete(data,np.s_[0:final_columns],1)

    if filetype[:2]=="qe":
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
            if re.search('Writing.*output',line) or re.search('init_run',line):    #Reading is completed
                break
            elif re.search('bands \(ev\)',line):    #New k_point
                if '-' in line:
                    l=__insert_space_before_minus(line)
                    l=l.split()
                else:
                    l=line.split()
                point1=np.array(l[2:5]).astype(float)
                coord1=point1              # Already in reciprocal cartesian coord (not like VASP)
                delta=np.linalg.norm(coord1-coord0)
                if delta>=0.10:                                  #fast fix for broken paths
                    delta=0
                dist=dist+delta
                coord0=coord1
                i=i+1
                j=1
                data[i,0]=dist
            else:                           #Load energies
                line=__insert_space_before_minus(line)
                l=line.split()
                energies=np.array(l).astype(float)
                data[i,j:j+len(energies)]=energies
                j=j+len(energies)
       
    elif filetype=="eigenval":
        file=open(filename,'r')
        lines=file.readlines()
        num_points=int(lines[5].split()[1])
        num_bands=int(lines[5].split()[2])
        data_lines=lines[7:]

        data=np.zeros([num_points,num_bands+1])
        if vectors.all()!=None:                      #If there is no cell in the input
            vectors = vectors/np.linalg.norm(vectors[0])
            K_vec=ut.K_basis(vectors)
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
                point1=np.array(line).astype(float)[0:2]
                coord1=ut.cryst2cartesian(point1,K_vec)
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
                point1=np.array(line).astype(float)[0:3]
                weight=float(line[3])
                if weight == 0 or IgnoreWeight==True:           #Remove weighted points (usefull for HSE and mBJ calculations)
                    coord1=ut.cryst2cartesian(point1,K_vec)
                else:
                    coord1=np.zeros(3)
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
        if IgnoreWeight==False:                                 #Trim non-necessary points
            index=np.argwhere(data[:,0]==0)[-1][0]
            data=data[index:,:]
        data[:,0]=data[:,0]-data[0,0]
    return data

def __plot_electrons(file,filetype=None,vectors=np.array(None),ticks=np.array(None),fermi=None,color=None,style=None,
                     linewidth=None,marker=None,legend=None,num_elec=None,IgnoreWeight=True,save_raw_data=None,ax=None,
                     plot=True):
    """Print the bands given by __process_electron_bands 
    BUT DOES NOT SHOW THE OUTPUT (not plt.show())
    file = Path to the file
    filetype = qe (quantum espresso bands.pwo)
               eigenval (VASP EIGENVAL file)
               data (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks = np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
    fermi = Fermi energy in order to shift the band structure and put it at zero
    color = string with the color for the bands or 'VC' and num_electrons to use blue/red for valence/conduction
    num_elec= Number of electrons
    style = string with the linestyle (solid, dashed, dotted)
    IgnoreWeight = Boolean controlling whether points with non-zero weight would be ignored
    linewidth = width of the line
    save_raw_data = 'File to save the plotable data'
    ax = ax in which to plot
    plot = Boolean controlling whether you want to plot or not
    """
    data=__process_electron_bands(file,filetype,vectors,IgnoreWeight)
    if save_raw_data != None:
        np.savetxt(save_raw_data,data)

    if fermi!=None:                       #Fermi energy
        data[:,1:]=data[:,1:]-fermi

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        x_lim=ticks[ticks.shape[0]-1]                #resize x_data for wannier and other codes output
        data[:,0]=data[:,0]*(x_lim/data[:,0].max())

    if color=='VC':
        color=['tab:blue','tab:red']

    if type(color)==list and plot==True:
        ax.plot(data[:,0],data[:,1],linestyle=style,marker=marker,linewidth=linewidth,color=color[0],label=legend)
        ax.plot(data[:,0],data[:,2:num_elec+1],linestyle=style,marker=marker,linewidth=linewidth,color=color[0])
        ax.plot(data[:,0],data[:,num_elec+1:],linestyle=style,marker=marker,linewidth=linewidth,color=color[1])
    elif plot==True:
        ax.plot(data[:,0],data[:,1],linestyle=style,marker=marker,linewidth=linewidth,color=color,label=legend)
        ax.plot(data[:,0],data[:,2:],linestyle=style,marker=marker,linewidth=linewidth,color=color)

    return [data[:,0].min(),data[:,0].max(),data[:,1:].min(),data[:,1:].max()]

def bands(file,KPATH=None,aux_file=None,title=None,proj_file=None,vectors=np.array(None),ticks=np.array(None),labels=None,
               fermi=None,window=None,plot_DOS=True,DOS_file='aux',num_elec=None,color=None,filetype=None,figsize=(8,4),legend=None,
                style=None,plot_ticks=True,linewidth=1,ratio=0.2,IgnoreWeight=True,save_as=None,save_raw_data=None,axis=None):
    """Plots the:
        bands.pwo file of a band calculation in Quantum Espresso
        EIGENVALUES file of a VASP calculation
        bands.dat.gnu file of bands postprocessing (Quantum Espresso)
        band.dat file in Wannier90
        bulkek.dat in Wanniertools

    Minimal plots can be done with just:
        file = Path to the file with bandstructure
        filetype = (It should be detected automatically)
                   qe (quantum espresso bands.pwo)
                   EIGENVAL (VASP EIGENVAL file)
                   data (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)

    Two aditional files can be provide to autocomplete almost everything:
        KPATH = File with PATH and legends for the HSP in the VASP format as provided by topologicalquantumchemistry.fr
        aux_file = A file from which read the Fermi level, number of electrons and structure.
                   In the case of QE this would be an scf.pwo of nscf.pwo
                   In the case of VASP this is the OUTCAR

    However everything may be introduced manually:
    proj_file = procar (VASP) or proj.pwo (QE) from which you want to extract the DOS
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks = np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
            Ticks in your bandstructure (the usual HSP)
    labels = ["$\Gamma$","$X$","$M$","$\Gamma$"]
    title = 'Your nice and original title for the plot'
    fermi = Fermi energy in order to shift the band structure accordingly
    window = Window of energies to plot around Fermi, it can be a single number or 2
            either window=0.5 or window=[-0.5,0.5] => Same result
    plot_DOS = Whether to plot the DOS as an additional axis.
    DOS_file = The can be plotted either from the "aux" file, or from the "bands" file
    color = Either a color or "VC" to use Valence and Conduction bands with different color
    figsize = (int,int) => Size and shape of the figure
    legend = Legend for the plot
    style = desired line style (solid, dashed, dotted...)
    plot_ticks = Boolean describing wether you want your ticks and labels
    linewidth = desired line width
    ratio = Ratio between DOS and bands plot
    IgnoreWeight = Boolean controlling whether points with non-zero weight would be ignored
    save_as = 'wathever.format'
    save_raw_data = 'file.dat' The processed data ready to plot
    axis = ax in which to plot, if no axis is present new figure is created
    """
    #READ input
    if filetype == None:
        file = ut.file(file)
    else:
        file = ut.file(file,filetype)
    if KPATH != None:
        KPATH=ut.file(KPATH)
    if aux_file != None:
        aux_file=ut.file(aux_file)

    #Select parameters
    if KPATH!=None:
        ticks,labels=KPATH.path,KPATH.labels
    if file.filetype[:2]=='qe' and aux_file==None:
        vectors=file.lattice

    if aux_file!=None:
        v=aux_file.lattice
        f=aux_file.fermi
        n=aux_file.electrons
        if fermi==None:
            fermi=f
        if num_elec==None:
            num_elec=n
        if vectors.any()==None:
            vectors=v

    if fermi!=None:
        if window==None:
            window=1
        if num_elec!=None and color==None:
            color='VC'

    if axis == None:
        fig=plt.figure(figsize=figsize)
        if plot_DOS==False:
            ax = fig.add_subplot(111)
        else:
            gs = fig.add_gridspec(1, 2, hspace=0, wspace=0,width_ratios=[1-ratio, ratio])
            ax,ax_DOS = gs.subplots(sharex='col', sharey='row')
    else:
        plot_DOS = False
        ax=axis
    limits=__plot_electrons(file.file,file.filetype,vectors,ticks,fermi,color=color,num_elec=num_elec,legend=legend,
                            linewidth=linewidth,save_raw_data=save_raw_data,style=style,ax=ax,IgnoreWeight=IgnoreWeight)

    ax.set_ylabel('energy (eV)',labelpad=-1)

    if fermi!=None:                       #Fermi energy
        ax.axhline(y=0,color='black',linewidth=0.4)
        if window!=None:                   #Limits y axis
            if type(window) is int or type(window) is float:
                window=[-window,window]
            ax.set_ylim(window[0],window[1])
        else:
            delta_y=limits[3]-limits[2]
            ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1)

    if vectors.any()!=None and ticks.any()!=None and plot_ticks==True:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        if labels != None :
            ax.set_xticks(ticks,labels)
        else:
            ax.set_xticks(ticks)
        for i in range(1,ticks.shape[0]-1):
            ax.axvline(ticks[i],color='gray',linestyle='--',linewidth=0.4)
    else:
        ax.set_xticks([])
    if plot_DOS==True:
        if DOS_file=='aux':
            fileD=aux_file.file
        elif DOS_file=='bands':
            fileD=file.file
        else:
            print('ERROR:',DOS_file,"option for DOS_file is not implemented.")
        if proj_file==None:
            DOS(fileD,fermi=fermi,window=window,reverse=True,axis=ax_DOS)
        else:
            DOS_projected(fileD,proj_file,fermi=fermi,window=window,reverse=True,axis=ax_DOS,silent=True)
        ax_DOS.set_ylabel('')
        ax_DOS.set_xlabel('DOS')
        
    if title!=None:                             #Title option
        ax.set_title(title)

    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    plt.tight_layout()

    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

def bands_compare(files,KPATH=None,fermi=None,legends=None,title=None,aux_file=None,vectors=np.array(None),
                  ticks=np.array(None),labels=None,window=1,figsize=None,save_as=None,filetypes=None,
                  styles=['-','--','-.',':'],
                  markers=['','','',''],
                  colors=['tab:blue','tab:red','tab:green','tab:orange'],
                  IgnoreWeight=True,
                  axis=None):
    """Plots the:
        bands.pwo file of a band calculation in Quantum Espresso
        EIGENVALUES file of a VASP calculation
        bands.dat.gnu file of bands postprocessing (Quantum Espresso)
        band.dat file in Wannier90
        bulkek.dat in Wanniertools

    Minimal plots can be done with just:
        files = list of files with bandstructures 
        filetypes = list with the filetype for each file (It should be detected automatically)
                   qe (quantum espresso bands.pwo)
                   EIGENVAL (VASP EIGENVAL file)
                   data (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)

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
    IgnoreWeight = Boolean controlling whether points with non-zero weight would be ignored
    axis = ax in which to plot, if no axis is present new figure is created
    """
    styles=styles*4
    colors=colors*4
    markers=markers*4

    #READ input
    if filetypes == None:
        filetypes = [None] * len(files)
        for i in range(len(filetypes)):
            filetypes[i]=ut.grep_filetype(files[i])
    else:
        for i in range(len(filetypes)):
            filetypes[i]=filetypes[i].lower()
    if KPATH != None:
        KPATH=ut.file(KPATH)
    if aux_file != None:
        aux_file=ut.file(aux_file)

    #Select parameters
    if fermi==None:
        fermi=np.zeros(len(files))
    if legends==None:
        legends=['Data ' + str(n+1) for n in range(len(files))]
    if KPATH!=None:
        ticks,labels=KPATH.path,KPATH.labels
    for i,kind in enumerate(filetypes):            #Automatically read vectors if possible
        if kind[:2]=='qe' and aux_file==None and vectors.any()==None:
            vectors=ut.grep_lattice(files[i])
            break
    if aux_file!=None and vectors.any()==None:
        vectors=aux_file.lattice
    
    if axis == None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax = axis
    for num in range(len(files)):
        data_limits=__plot_electrons(files[num],filetypes[num],vectors,ticks,fermi=fermi[num],color=colors[num]
                                       ,style=styles[num],marker=markers[num],legend=legends[num],IgnoreWeight=IgnoreWeight,ax=ax)
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
    
    ax.legend(loc='upper right')
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()


def bands_fat(file,proj_file,KPATH=None,aux_file=None,species=None,atoms=None,l=None,j=None,mj=None,
          title=None,color='Reds',colormap=True,vmin=0,vmax=1,shift=0,size=50,legend=None,only_fat=False,
          vectors=np.array(None),ticks=np.array(None),labels=None,fermi=None,window=None,
          back_color='gray',style=None,linewidth=0.5,filetype=None,proj_filetype=None,
          figsize=(8,4),plot_ticks=True,
          IgnoreWeight=True,save_as=None,axis=None):
    """Plots fat bands over:
        bands.pwo file of a band calculation in Quantum Espresso
        EIGENVALUES file of a VASP calculation
        bands.dat.gnu file of bands postprocessing (Quantum Espresso)
        band.dat file in Wannier90
        bulkek.dat in Wanniertools

    Minimal plots can be done with just:
        file = Path to the file with bandstructure
        proj_file = File with the projected bands, or the output from grep_kpoints_energies_projections
                   qe_proj_out (quantum espresso out for projwfc.x)
                   PROCAR (VASP projections file)

    Two aditional files can be provide to autocomplete almost everything:
        KPATH = File with PATH and legends for the HSP in the VASP format as provided by topologicalquantumchemistry.fr
        aux_file = A file from which read the Fermi level, number of electrons and structure.
                   In the case of QE this would be an scf.pwo of nscf.pwo
                   In the case of VASP this is the OUTCAR
    
    species = list of atomic species ['Bi','Se'...] to project over.
    atoms = list with atoms index [1,2...] to project over.
    l = list of orbital atomic numbers to project over:
        qe: [0, 1, 2]
        vasp: ['s','px','py','dxz']  (as written in POSCAR)
    j = total angular mometum of projected states. (qe only)
    mj = m_j state of projected states. (qe only)

    However everything may be introduced manually:
    
    title = 'Your nice and original title for the plot'
    color = Color for your fat bands
    colormap = Boolean whether you are inputing a color or a colormap.
    vmin = Minimum value of a projection for the colormap
    vmax = Maximum value of a projection for the colormap
    shift = For plotting multiple projections it is handy to sligtly shift them.
    size = factor for which the size of projections is mutiplied.
    legend = Legend for flat bands.
    only_fat = Boolean controlling whether you want just the fat bands (to overlap over other plot).
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    ticks = np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
            Ticks in your bandstructure (the usual HSP)
    labels = ["$\Gamma$","$X$","$M$","$\Gamma$"]
    fermi = Fermi energy in order to shift the band structure accordingly
    window = Window of energies to plot around Fermi, it can be a single number or 2
            either window=0.5 or window=[-0.5,0.5] => Same result
    back_color = Color for your "non-projected" bands
    style = desired line style (solid, dashed, dotted...)
    linewidth = desired line width
    proj_filetype = qe_proj_out (quantum espresso proj.pwo)
                    procar (VASP PROCAR file)
    filetype = Filetype of yoru bandstructure
                   qe (quantum espresso bands.pwo)
                   EIGENVAL (VASP EIGENVAL file)
                   data (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)   
    figsize = (int,int) => Size and shape of the figure
    plot_ticks = Boolean describing wether you want your ticks and labels
    IgnoreWeight = Boolean controlling whether points with non-zero weight would be ignored
    save_as = 'wathever.format'
    axis = ax in which to plot, if no axis is present new figure is created
    """
    #READ input
    if filetype == None:
        file = ut.file(file)
    else:
        file = ut.file(file,filetype)
    if KPATH != None:
        KPATH=ut.file(KPATH)
    if aux_file != None:
        aux_file=ut.file(aux_file)
    if type(proj_file)!=str:
        STATES, KPOINTS, ENERGIES, PROJECTIONS = proj_file
    else:
        proj_file=ut.file(proj_file)
        STATES, KPOINTS, ENERGIES, PROJECTIONS = proj_file.grep_kpoints_energies_projections(proj_file.filetype,IgnoreWeight)

    #Select parameters
    if KPATH!=None:
        ticks,labels=KPATH.path,KPATH.labels
    if file.filetype[:2]=='qe' and aux_file==None:
        vectors=file.lattice

    if aux_file!=None:
        v=aux_file.lattice
        f=aux_file.fermi
        n=aux_file.electrons
        if fermi==None:
            fermi=f
        if vectors.any()==None:
            vectors=v

    if fermi!=None:
        if window==None:
            window=1
    else:
        fermi=0

    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    data=__process_electron_bands(file.file,file.filetype,vectors,IgnoreWeight)
    limits=__plot_electrons(file.file,file.filetype,vectors,ticks,fermi,color=back_color,linewidth=linewidth,
                                 style=style,ax=ax,IgnoreWeight=IgnoreWeight,plot=not only_fat)

    K_len=data[:,0]*limits[1]/data[:,0].max()
    ENERGIES=ENERGIES-fermi

    proj,n = ut.sum_projections(STATES,PROJECTIONS,proj_filetype,species,atoms,l,j,mj)
    print('(',species, atoms,l,j,mj,') ',n, 'states summed')

    proj=proj.transpose()
    for i,E in enumerate(ENERGIES.transpose()):
        if colormap==True:
            scatter=ax.scatter(K_len,E+shift,s=proj[i]*size,c=proj[i],cmap=color,alpha=proj[i],vmin=vmin,vmax=vmax,edgecolors='none')
        else:
            ax.scatter(K_len,E+shift,s=proj[i]*size,c=color,alpha=proj[i],edgecolors='none')
    if legend!=None:
        if colormap==True:
            ax.scatter(-1,0,c=0.7,cmap=color,s=20,label=legend,vmin=0,vmax=1)                #Dummy point (outside the plot) for the legend
        else:
            ax.scatter(-1,0,c=color,s=20,label=legend)
            
    if fermi!=None and only_fat==False:                       #Fermi energy
        ax.axhline(y=0,color='black',linewidth=0.4)
        if window!=None:                   #Limits y axis
            if type(window) is int or type(window) is float:
                window=[-window,window]
            ax.set_ylim(window[0],window[1])
        else:
            delta_y=limits[3]-limits[2]
            ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1)

    if vectors.any()!=None and ticks.any()!=None and plot_ticks==True and only_fat==False:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        if labels != None :
            ax.set_xticks(ticks,labels)
        else:
            ax.set_xticks(ticks)
        for i in range(1,ticks.shape[0]-1):
            ax.axvline(ticks[i],color='gray',linestyle='--',linewidth=0.4)
    if title!=None:                             #Title option
        ax.set_title(title)

    ax.set_ylabel('energy (eV)',labelpad=-1)
    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    plt.tight_layout()

    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()

# PLOTTING PHONONS----------------------------------------------------------------

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

def __process_phonon_bands(gnu_file,QE_path):
    if QE_path.any()==None:
        data=np.loadtxt(fname=gnu_file)
    else:
        #SELECT THE LINES WHERE THE PATH IS SPLITED
        lines=[]
        line=0
        for i in range(len(QE_path)):
            if QE_path[i,3]==1:
                lines=lines+[line]
            line=line+int(QE_path[i,3])

        #LOAD THE DATA AND CORRECT THE SPLITS TO MAKE IT "CONTINOUS"
        data=np.loadtxt(gnu_file)
        for line in lines:
            if line < data.shape[0]-1:
                data[line+1:,0]=data[line+1:,0]-(data[line+1,0]-data[line,0])
    return data

def __plot_phonons(file,linewidth,vectors=np.array(None),ticks=np.array(None),
                        units='cm-1',color=None,style=None,legend=None,QE_path=None,ax=None):
    """Print the phonons.freq.gp file of matdyn output (Quantum Espresso)
    BUT DOES NOT SHOW THE OUTPUT (not plt.show())
    plot_phonons(file,real_vecs,ticks)
    file=Path to the file
    vectors=np.array([[a1,a2,a3],...,[c1,c2,c3]])
    ticks=np.array([[tick1x,tick1y,tick1z],...,[ticknx,tickny,ticknz]])
    units= Eiher 'cm-1' or 'meV'
    color = string with the color for the bands or an array with the color for each band
    style = string with the linestyle (solid, dashed, dotted)
    legend = legend to add for the data set
    QE_path = matdyn file for splitted paths, it will correct the path to make it "continous"
    QE_path = reciprocal space path as given by the grep_ticks_QE or grep_ticks_labels_KPATH. It will fix splitted paths and correct the path to make it "continous"
    ax = ax in which to plot
    """
    data=__process_phonon_bands(file,QE_path)
    if units=='meV':
        data[:,1:]=data[:,1:]*cons.cm2meV

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        ticks=__ticks_generator(vectors,ticks)
        x_lim=ticks[ticks.shape[0]-1]                #resize x_data for wannier and other codes output
        data[:,0]=data[:,0]*(x_lim/data[:,0].max())

    if color==None or type(color)==str:
        ax.plot(data[:,0],data[:,1],linestyle=style,linewidth=linewidth,color=color,label=legend)
        ax.plot(data[:,0],data[:,2:],linestyle=style,linewidth=linewidth,color=color)
    elif len(color)==(len(data[0])-1):
        ax.plot(data[:,0],data[:,1],linestyle=style,linewidth=linewidth,color=color[0],label=legend)
        for i in range(1,len(color)):
            ax.plot(data[:,0],data[:,i+1],linestyle=style,linewidth=linewidth,color=color[i])
    else:
        print('Color flag not correctly set up')

    return [data[:,0].min(),data[:,0].max(),data[:,1:].min(),data[:,1:].max()]

def phonons(file,KPATH=None,ph_out=None,title=None,matdyn_in=None,grid=True,vectors=np.array(None),
                ticks=np.array(None),labels=None,units='cm-1',save_as=None,figsize=None,color=None,style=None,linewidth=1,legend=None,axis=None):
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
    units= Eiher 'cm-1' or 'meV'
    save_as='wathever.format'
    figsize = (int,int) => Size and shape of the figure
    color = linecolor for your plot (it also can be an array with color for each band)
    style = string with the linestyle (solid, dashed, dotted)
    linewidth = linewidth of your plot
    legend = legend to add for the data set
    axis = ax in which to plot, if no axis is present new figure is created
    """

    if KPATH!=None:
        ticks,labels=ut.grep_ticks_labels_KPATH(KPATH)
    if matdyn_in!=None:
        ticks=ut.grep_ticks_QE(matdyn_in)
    if ph_out!=None and vectors.any()==None:
        vectors=ut.grep_lattice(ph_out)
    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    limits=__plot_phonons(file,linewidth,vectors,ticks,units=units,color=color,style=style,QE_path=ticks,legend=legend,ax=ax)
    if units=='cm-1':
        ax.set_ylabel('frequency $(\mathrm{cm^{-1}})$')
    if units=='meV':
        ax.set_ylabel('frequency $(\mathrm{meV})$')
    ax.axhline(y=0,color='gray',linestyle='--',linewidth=0.4)

    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    delta_y=limits[3]-limits[2]
    ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1) #Limits in the y axis

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        if ph_out!=None and grid==True:
            grid_points=ut.grep_ph_grid_points(ph_out,expanded=True,decimals=15) #to correctly find distances
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

def phonons_compare(files,KPATH=None,ph_outs=None,legends=None,title=None,matdyn_in=None,grid=True,
                         vectors=np.array(None),ticks=np.array(None),labels=None,units='cm-1',save_as=None,
                         colors=['tab:blue','tab:red','tab:green','tab:orange'],
                         styles=['solid','dashed','dashdot','dotted'],linewidth=1,figsize=None,axis=None):
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
    units= Eiher 'cm-1' or 'meV'
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
    if matdyn_in!=None:
        ticks=ut.grep_ticks_QE(matdyn_in)
    if ph_outs!=None and vectors.any()==None:
        vectors=ut.grep_lattice(ph_outs[0])

    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis

    for num in range(len(files)):
        data_limits=__plot_phonons(files[num],linewidth,vectors,ticks,units=units,color=colors[num]
                                       ,style=styles[num],legend=legends[num],QE_path=ticks,ax=ax)
        if num==0:
            limits=data_limits
        else:
            limits[0]=min(data_limits[0],limits[0])
            limits[1]=max(data_limits[1],limits[1])
            limits[2]=min(data_limits[2],limits[2])
            limits[3]=max(data_limits[3],limits[3])

    if units=='cm-1':
        ax.set_ylabel('frequency $(\mathrm{cm^{-1}})$')
    if units=='meV':
        ax.set_ylabel('frequency $(\mathrm{meV})$')
    ax.axhline(y=0,color='gray',linestyle='--',linewidth=0.4)

    ax.set_xlim(limits[0],limits[1])   #Limits in the x axis
    delta_y=limits[3]-limits[2]
    ax.set_ylim(limits[2]-delta_y*0.05,limits[3]+delta_y*0.1) #Limits in the y axis

    if vectors.any()!=None and ticks.any()!=None:    #ticks and labels
        if ph_outs!=None and grid==True:
            grid_points=ut.grep_ph_grid_points(ph_outs[0],expanded=True,decimals=15)
            path=ticks
            ticks,grid=__ticks_generator(vectors,ticks,grid_points)
            ax.set_xticks(ticks,labels)
            for num in range(len(ph_outs)):
                grid_points=ut.grep_ph_grid_points(ph_outs[num],expanded=True,decimals=15)
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
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save_as!=None:                             #Saving option
        plt.savefig(save_as, dpi=500)
    if axis == None:
        plt.show()


# PLOTTING MISCELLANY----------------------------------------------------------------
 
def lattice_comparison(folder,title=None,control=None,percentile=True,axis=None,markersize=8,save_as=None,output=False):
    """
    Plots the lattice comparison between different relax procedures, it is usefull to find the best pseudo/interaction matching your system.
    CAUTION: Be aware that it works in the STANDARDICE CELL convention!!! It will convert the files to compare in such setting.
    
    folder = Parent folder from where your relaxation kinds span. (the expected structure is explained below)
    title = Title for your plot
    control = File containing your control structure (for example the experimental one)
    percentile = If control structure is provided, then a percentile error plot is done.
    axis = ax in which to plot, if no axis is present new figure is created
    markersize = Size of the points
    save_as = Path and file type for your plot to be saved
    output = if true then the procedure returns two lists containing the interactions and the lattice parameters for each kind.
    
    ---
    The folowing folder structure is expected:
    Parent folder => Interaction1 => relax1 => output.pwo
                                  => relax2 => output.pwo
                  => Interaction2 => relax1 => output.pwo
                                  => relax2 => output.pwo
                                  => relax3 => output.pwo
                  ...
    The code will automatically select the last relaxation iteration's output.
    """
    #Grep the kinds of interactions from the respective folders
    interactions=[]
    folders=glob.glob(folder+'*')
    for file in folders:
        inter=file.split('/')[-1]
        interactions=interactions+[inter]   
    
    #For each interaction take the last relax output (assuming iterative relaxes named as relax<#num>)
    relax=[]
    for inter in interactions:
        relaxes=glob.glob(folder+'/'+inter+'/*/*pwo')
        relax_iter=0
        for r in relaxes:
            new_iter=int(r.split('/')[-2].split('relax')[1])
            if new_iter>relax_iter:
                last_relax=r
        relax=relax+[last_relax]
    
    #Read the data (in standardize cell)
    lattices=[]
    for file in relax:
        c=cell.read_spg(file)
        lattices=lattices+[spg.standardize_cell(c)[0]]
    
    if axis == None:
        fig=plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax=axis

    #Plot if None experimental (then just plotting the results, not the comparison)
    if control==None or percentile==False:
        if control!=None:
            c_lattice=spg.standardize_cell(cell.read_spg(control))[0]
            ax.plot(0,np.linalg.norm(c_lattice[0]),'o',color='tab:red',label='a',markersize=markersize)
            ax.plot(0,np.linalg.norm(c_lattice[1]),'o',color='tab:green',label='b',markersize=markersize)
            ax.plot(0,np.linalg.norm(c_lattice[2]),'o',color='tab:blue',label='c',markersize=markersize)
            n=0
        else:
            ax.plot(0,np.linalg.norm(lattices[0][0]),'o',color='tab:red',label='a',markersize=markersize)
            ax.plot(0,np.linalg.norm(lattices[0][1]),'o',color='tab:green',label='b',markersize=markersize)
            ax.plot(0,np.linalg.norm(lattices[0][2]),'o',color='tab:blue',label='c',markersize=markersize)
            n=1
        for i,d in enumerate(lattices[n:]):
            ax.plot(i+1,np.linalg.norm(d[0]),'o',color='tab:red',markersize=markersize)
            ax.plot(i+1,np.linalg.norm(d[1]),'o',color='tab:green',markersize=markersize)
            ax.plot(i+1,np.linalg.norm(d[2]),'o',color='tab:blue',markersize=markersize)
        
        ax.set_ylabel('angstrom ($\mathrm{\AA}$)')
        if control==None:
            ax.set_xticks(range(len(interactions)),labels=interactions,rotation=50)
        else:
            ax.set_xticks(range(len(interactions)+1),labels=['EXP']+interactions,rotation=50)

    #Plot with experimental structure as control (show percentile error)
    if control!=None and percentile==True:
        c_lattice=spg.standardize_cell(cell.read_spg(control))[0]
        c0=np.linalg.norm(c_lattice[0])
        c1=np.linalg.norm(c_lattice[1])
        c2=np.linalg.norm(c_lattice[2])
        ax.axhline(y=0,color='tab:red',linestyle='-',linewidth=0.5)
        ax.plot(0,(np.linalg.norm(lattices[0][0])-c0)*100/c0,'o',color='tab:red',label='a',markersize=markersize)
        ax.plot(0,(np.linalg.norm(lattices[0][1])-c1)*100/c1,'o',color='tab:green',label='b',markersize=markersize)
        ax.plot(0,(np.linalg.norm(lattices[0][2])-c2)*100/c2,'o',color='tab:blue',label='c',markersize=markersize)
        for i,d in enumerate(lattices[1:]):
            ax.plot(i+1,(np.linalg.norm(d[0])-c0)*100/c0,'o',color='tab:red',markersize=markersize)
            ax.plot(i+1,(np.linalg.norm(d[1])-c1)*100/c1,'o',color='tab:green',markersize=markersize)
            ax.plot(i+1,(np.linalg.norm(d[2])-c2)*100/c2,'o',color='tab:blue',markersize=markersize)
        ax.set_ylabel('percentile error (%)')
        ax.set_xticks(range(len(interactions)),labels=interactions,rotation=50)
        
    if title!=None:
        ax.set_title(title)
    ax.legend()
    ax.grid()
    if axis == None:
        plt.tight_layout()
        plt.show()
        if save_as!=None:
            plt.savefig(save_as,dpi=300)
    if output==True:
        return interactions,lattices
