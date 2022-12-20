#PYTHON module for ploting 

import numpy as np
import matplotlib.pyplot as plt
import re

import yaiv.utils as ut

# PLOTTING BANDS**************************************************************************************

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
        columns=np.int_(2*data.shape[0]/rows)
        data=np.reshape(data,(rows,columns),order='F')
        final_columns=np.int_(columns/2-1)
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
                point1=np.array(l[2:5]).astype(np.float_)
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
                l=line.split()
                energies=np.array(l).astype(np.float_)
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
                point1=np.array(line).astype(np.float_)[0:2]
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
                point1=np.array(line).astype(np.float_)[0:3]
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

def __plot_electrons(file,filetype=None,vectors=np.array(None),ticks=np.array(None),fermi=None,color=None,style=None,marker=None,legend=None,num_elec=None,save_raw_data=None,ax=None):
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
        if num_elec!=None and color==None:
            color='VC'


    if axis == None:
        fig=plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        ax=axis
    limits=__plot_electrons(file,filetype,vectors,ticks,fermi,color=color,num_elec=num_elec,save_raw_data=save_raw_data,ax=ax)

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
    markers=markers*4

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
        data_limits=__plot_electrons(files[num],filetypes[num],vectors,ticks,fermi=fermi[num],color=colors[num]
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
