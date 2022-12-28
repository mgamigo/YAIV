#PYTHON module for ploting bands, two main functions:
#p)lot_phonons(file,vectors,ticks,labels,title)
#plot_bands(file,vectors,ticks,labels,title,fermi,window)

import numpy as np
import matplotlib.pyplot as plt
import re
import glob
import os
import subprocess
import yaiv.plot as plot
import yaiv.experimental.cell_analyzer as cell
import yaiv.constants as cons
import spglib as spg

#Utilities*****************************************************************************************

def fermi_surface(file):
    """Just launches fermisurfer for you from the python notebook"""
    cmd='fermisurfer '+file
    subprocess.call(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def lattice_comparison(folder,title=None,control=None,percentile=True,save_as=None,output=False):
    """
    Plots the lattice comparison between different relax procedures, it is usefull to find the best pseudo/interaction matching your system.
    CAUTION: Be aware that it works in the STANDARDICE CELL convention!!! It will convert the files to compare in such setting.
    
    folder = Parent folder from where your relaxation kinds span. (the expected structure is explained below)
    title = Title for your plot
    control = File containing your control structure (for example the experimental one)
    percentile = If control structure is provided, then a percentile error plot is done.
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
    
    plt.figure()
    #Plot if None experimental (then just plotting the results, not the comparison)
    if control==None or percentile==False:
        if control!=None:
            c_lattice=spg.standardize_cell(cell.read_spg(control))[0]
            plt.plot(0,np.linalg.norm(c_lattice[2]),'o',color='tab:blue',label='c')
            plt.plot(0,np.linalg.norm(c_lattice[1]),'o',color='tab:green',label='b')
            plt.plot(0,np.linalg.norm(c_lattice[0]),'o',color='tab:red',label='a')
            n=0
        else:
            plt.plot(0,np.linalg.norm(lattices[0][2]),'o',color='tab:blue',label='c')
            plt.plot(0,np.linalg.norm(lattices[0][1]),'o',color='tab:green',label='b')
            plt.plot(0,np.linalg.norm(lattices[0][0]),'o',color='tab:red',label='a')
            n=1
        for i,d in enumerate(lattices[n:]):
            plt.plot(i+1,np.linalg.norm(d[2]),'o',color='tab:blue')
            plt.plot(i+1,np.linalg.norm(d[1]),'o',color='tab:green')
            plt.plot(i+1,np.linalg.norm(d[0]),'o',color='tab:red')
        
        plt.ylabel('Angstrom')
        if control==None:
            plt.xticks(range(len(interactions)),labels=interactions,rotation=50)
        else:
            plt.xticks(range(len(interactions)+1),labels=['EXP']+interactions,rotation=50)

    #Plot with experimental structure as control (show percentile error)
    if control!=None and percentile==True:
        c_lattice=spg.standardize_cell(cell.read_spg(control))[0]
        c0=np.linalg.norm(c_lattice[0])
        c1=np.linalg.norm(c_lattice[1])
        c2=np.linalg.norm(c_lattice[2])
        plt.axhline(y=0,color='tab:red',linestyle='-',linewidth=0.5)
        plt.plot(0,(np.linalg.norm(lattices[0][2])-c2)*100/c2,'o',color='tab:blue',label='c')
        plt.plot(0,(np.linalg.norm(lattices[0][1])-c1)*100/c1,'o',color='tab:green',label='b')
        plt.plot(0,(np.linalg.norm(lattices[0][0])-c0)*100/c0,'o',color='tab:red',label='a')
        for i,d in enumerate(lattices[1:]):
            plt.plot(i+1,(np.linalg.norm(d[2])-c2)*100/c2,'o',color='tab:blue')
            plt.plot(i+1,(np.linalg.norm(d[1])-c1)*100/c1,'o',color='tab:green')
            plt.plot(i+1,(np.linalg.norm(d[0])-c0)*100/c0,'o',color='tab:red')
        plt.ylabel('Percentile error (%)')
        plt.xticks(range(len(interactions)),labels=interactions,rotation=50)
        
    if title!=None:
        plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if save_as!=None:
        plt.save_as(save_as,dpi=300)
    plt.show()
    if output==True:
        return interactions,lattices

def adapt_path(vectors1,vectors2,KPATH):
    """Adapts the KPATH in TQC format to another set of coordinates in order to match paths with differente subsets
    vectors1 = original set of vectors (use grep_vectors)
    vectors2 = original set of vectors (use grep_vectors)
    KPATH = file with the KPATH in VASP format
    """
    #reciprocal vectors in columns
    rec1=np.linalg.inv(vectors1)
    rec2=np.linalg.inv(vectors2)
    M=np.matmul(vectors2,rec1)
    
    PATH=open(KPATH,'r')
    QE_path=np.zeros(0)
    q0=np.zeros(0)
    repeat=False
    path_section=False
    for line in PATH:
        if path_section==True:
            if len(line.split())!=0:
                X=float(line.split()[0])
                Y=float(line.split()[1])
                Z=float(line.split()[2])
                q1=np.array([X,Y,Z,100])
                q1[0:3]=np.matmul(M,q1[0:3])
                label=line.split('!')[1]
                if not np.array_equal(q0,q1):
                    if len(QE_path)==0:
                        QE_path=q1
                    else:
                        if repeat==False:
                            if len(QE_path.shape)>1:
                                if QE_path[len(QE_path)-2,3]!=1:
                                    QE_path[len(QE_path)-1,3]=1
                            QE_path=np.vstack((QE_path,q1))
                            q0=q1
                        else:
                            QE_path=np.vstack((QE_path,q1))
                            q0=q1
                            repeat=False
                else:
                    repeat=True
        if re.search('Reciprocal',line,flags=re.IGNORECASE):
            path_section=True
    QE_path[len(QE_path)-1,3]=1

    num_q=len(QE_path[:,0])

    np.savetxt('tmp.path',QE_path,fmt='%1.5f %1.5f %1.5f    %1.0f')

    f = open('QE_path','w')
    f.write(str(num_q)+"\n")
    tmp=open('tmp.path')
    for line in tmp:
        f.write(line)
    f.close()
    os.remove('tmp.path')

def count_number_of_bands(file,window=None,fermi=0):
    """Counts the number of bands in an energy window for the bands.dat.gnu file that QE produces"""
    data=np.loadtxt(fname=file)
    data=data[:,:2]         #select the first two columns to process (for wannier tools)
    rows=1
    position=data[rows,0]
    data=data[:,:2]         #select the first two columns to process (for wannier tools)

    while position!=0:     #counter will tell me how many points are in the x axis, number of rows
        rows=rows+1
        position=data[rows,0]
    columns=np.int(2*data.shape[0]/rows)
    data=np.reshape(data,(rows,columns),order='F')
    final_columns=np.int(columns/2-1)
    data=np.delete(data,np.s_[0:final_columns],1)
    if window==None:
        bands=data.shape[1]-1
        print("the total number of bands is",bands)
    else:
        bands=0
        first_q=data[0,1:]-fermi
        for item in first_q:
            if item>=window[0] and item<=window[1]:
                bands=bands+1
        print("The number of bands between",str(window[0])+"eV and",str(window[1])+"eV is",bands)
    return bands



# GREPPING utilities***********************************************************************

def grep_vectors(file,filetype='qe'):
    """Greps the real vectors from a scf.pwo, bands.pwo (in the alat units) or file VASP OUTCAR 
    (it may work with other output files of QE)
    OUTPUT= np.array([vec1,vec2,vec3])
    """
    filetype=filetype.lower()
    count=0
    lattice_lines=False
    lines=open(file,'r')

    if filetype=='qe':
        for line in lines:
            if lattice_lines==True:
                X=float(line.split()[3])
                Y=float(line.split()[4])
                Z=float(line.split()[5])
                vec=np.array([X,Y,Z])
                if count==0:
                    vectors=vec
                else:
                    vectors=np.vstack((vectors,vec))            
                count=count+1
                if count>=3:
                    break
            if re.search('crystal axes',line,flags=re.IGNORECASE):
                lattice_lines=True
    elif filetype=='vasp':
        for line in lines:
            if re.search('direct lattice vectors',line):
                lattice_lines=True
            elif count>=3:
                break
            elif lattice_lines==True:
                count=count+1
                if count>0:
                    l=line.split()
                    X=float(l[0])
                    Y=float(l[1])
                    Z=float(l[2])
                    vec=np.array([X,Y,Z])
                    try:
                        vectors=np.vstack([vectors,vec])
                    except NameError:
                        vectors=vec
        norm=np.linalg.norm(vectors[0])
        vectors=vectors/norm
    else:
        print('FILETYPE NOT AVAILABLE')
        print('could not grep vectors')
    return vectors

def grep_ticks_labels_KPATH(file):
    """Greps ticks and labels of the ticks from a KPATH file of VASP.
    It expects the file to have the structure:
    0 0 0 !GM
    0 0.5 0 !X
    0 0.5 0 !X
    0.5 0.5 0 !

    It generates the correct ticks to plot with my scripts (even with splitted paths)

    return ticks, labels
    """
    KPATH=open(file,'r')
    ticks=np.zeros(0)
    q0=np.zeros(0)
    repeat=False
    labels=[]
    path_section=False
    for line in KPATH:
        if path_section==True:
            if len(line.split())!=0:
                X=float(line.split()[0])
                Y=float(line.split()[1])
                Z=float(line.split()[2])
                q1=np.array([X,Y,Z,100])
                label=line.split('!')[1].split()[0]
                if not np.array_equal(q0,q1):
                    labels=labels+[label]
                    if len(ticks)==0:
                        ticks=q1
                    else:
                        if repeat==False:
                            if len(ticks.shape)>1:
                                if ticks[len(ticks)-2,3]!=1:
                                    ticks[len(ticks)-1,3]=1
                            ticks=np.vstack((ticks,q1))
                            q0=q1
                        else:
                            ticks=np.vstack((ticks,q1))
                            q0=q1
                            repeat=False
                else:
                    repeat=True
        if re.search('Reciprocal',line,flags=re.IGNORECASE):
            path_section=True
    ticks[len(ticks)-1,3]=1

    num_q=len(ticks[:,0])

    num_labels=num_q
    path=True
    for i in range(num_q):
        if path==False:
            diff=ticks[i,:3]-ticks[i-1,:3]
            labels[i-1]=labels[i-1]+'|'+labels[i]
            labels[i]='000'
        if ticks[i,3]==1:
            path=False
            num_labels=num_labels-1
        else:
            path=True
#    print("you need to introduce",num_labels+1,"labels")
    while labels.count('000')>0:
        labels.remove('000')
    for i in range(len(labels)):
        if 'Gamma' in labels[i]:
            labels[i]=labels[i].split('Gamma')[0]+'\Gamma'+labels[i].split('Gamma')[1]
        labels[i]='$'+labels[i]+'$'
    return ticks, labels

def grep_ticks_QE(file):
    """Greps the path and generates ticks from a bands.pwi file.
    It takes into account when the distance between two points is 1 and therefore there is a splitted path for the bandstructure. It also informs for the number of labels needed.

    (it may work with other output files of QE, easy to addapt to matdyn)

    OUTPUT= np.array([tick1,tick2,tick3...])
    """
    KPATH=open(file)
    ticks=np.zeros(0)
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
                    if len(ticks)==0:
                        ticks=q1
                    else:
                        ticks=np.vstack((ticks,q1))
                i=i+1
        if re.search('K_POINTS.*crystal_b',line,flags=re.IGNORECASE):
            path_section=True

    num_labels=num_q
    path=True
    for i in range(num_q):
        if ticks[i,3]==1:
            path=False
            num_labels=num_labels-1
        else:
            path=True
    print("you need to introduce",num_labels+1,"labels")
    return ticks

def grep_fermi(file,filetype='qe'):
    """Greps the Fermi Energy from a scf.pwo, nscf.pwo ... or OUTCAR (VASP) file.
   returns the Fermi energy in eV
    """
    filetype=filetype.lower()
    if filetype=='qe':
        scf_out=open(file,'r')
        for line in scf_out:
            if re.search('Fermi energy is',line):
                E_f=float(line.split()[4])
            if re.search('highest occupied',line):
                if re.search('unoccupied',line):
                    E1=float(line.split()[6])
                    E2=float(line.split()[7])
                    print('The gap is',(E2-E1)*1000,'meV')
                    E_f=E1+(E2-E1)/2
                else:
                    E_f=float(line.split()[4])
    elif filetype=='vasp':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('E-fermi',line):
                E_f=float(line.split()[2])
    return E_f

def grep_electrons(file,filetype='qe'):
    """Greps the number of electrons from a scf.pwo or OUTCAR file.
   return num_elec
    """
    num_elec=None
    filetype=filetype.lower()
    if filetype=='qe':
        scf_out=open(file,'r')
        for line in scf_out:
            if re.search('number of electrons',line):
                num_elec=int(float(line.split()[4]))
    elif filetype=='vasp':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('NELECT',line):
                num_elec=int(float(line.split()[2]))
    return num_elec

def __grep_fermi_and_electrons(file,filetype='qe'):
    """Greps the Fermi Energy and number of electrons from a scf.pwo or OUTCAR file.
   returns the Fermi energy in eV
   (The advantage is we just read the file once)
    """
    num_elec=None
    filetype=filetype.lower()
    if filetype=='qe':
        scf_out=open(file,'r')
        for line in scf_out:
            if re.search('Fermi energy is',line):
                E_f=float(line.split()[4])
            if re.search('highest occupied',line):
                if re.search('unoccupied',line):
                    E1=float(line.split()[6])
                    E2=float(line.split()[7])
                    print('The gap is',(E2-E1)*1000,'meV')
                    E_f=E1+(E2-E1)/2
                else:
                    E_f=float(line.split()[4])
            if re.search('number of electrons',line):
                num_elec=int(float(line.split()[4]))
    elif filetype=='vasp':
        OUTCAR=open(file,'r')
        for line in OUTCAR:
            if re.search('E-fermi',line):
                E_f=float(line.split()[2])
            if re.search('NELECT',line):
                num_elec=int(float(line.split()[2]))
    return E_f, num_elec

def grep_grid_points(file,expanded=False,decimals=3):
    """Greps the grid points from a ph.pwo file, it reads the points and the star of those
    points given in the QE ouput and expresses them in reciprocal space lattice vectors. This grid
    can be further expanded to equivalent points line (0,0,0.5) and (0,0,-0.5).

    file='material.ph.pwo'
    """
    vectors=grep_vectors(file)
    text=open(file)
    read_text=False
    for line in text:
        if read_text==True and line.split()[0]==str(i):
            split=(line.split())
            point=np.array([float(split[1]),float(split[2]),float(split[3])])
            point=np.matmul(point,vectors.transpose())
            try:
                grid_points=np.vstack([grid_points,point])
            except NameError:
                grid_points=point
            i=i+1
            if i > num_star:
                read_text=False
        elif re.search('Number of q in the star',line):
            num_star=int(line.split()[7])
            i=1
            read_text=True
        elif re.search('In addition there is the',line):
            i=1
            read_text=True
    grid_points=np.around(grid_points,decimals=decimals) #Fix for detecting the grid in the paths
    
    if expanded==True:
        initial_grid=grid_points
        for point in initial_grid:
            expanded_star=__expand_star(point)
            try:
                grid=np.vstack([grid,expanded_star])
            except NameError:
                grid=expanded_star
    else:
        grid=grid_points
    return grid

def grep_stress_tensor(file,kbar=False):
    """greps the total stress tensor in (Ry/bohr**3) or (kbar) of scf.pwo file
    returns either the stress tensor or a False boolean if the pressure was not found"""
    lines=open(file,'r')
    pressure=False
    stress=None
    for line in lines:
        if pressure==True:
            l=line.split()
            l=[float(item) for item in l]
            vec=np.array(l[:3])
            try:
                stress=np.vstack([stress,vec])
                if len(stress)==3:
                    pressure=False
            except NameError:
                stress=vec
        if re.search('total.*stress',line):
            pressure=True
            del stress
    if kbar==True:
        stress=stress*(cons.Ry2jul/(cons.bohr2metre**3))*cons.pas2bar/1000
    return stress

def grep_total_energy(file,meV=False):
    """greps the total energy (in Ry or meV) of scf.pwo file
    returns either the energy or a False boolean if the energy was not found"""
    lines=open(file,'r')
    energy=False
    for line in lines:
        if re.search('!',line):
            l=line.split()
            energy=float(l[4])
    if meV==True:
        energy=energy*cons.Ry2eV*1000
    return energy

def __expand_star(q_point):
    """Expands the "star" of each point to equivalent points inside (111) (related by lattice) in the border of the BZ.
    This is usefull when a High Sym point is the (0.5,0,0), this generates (-0.5,0,0), which is not in the star
    since is it the same point.    
    """
    output=[q_point]
    for i in range(3):
        for point in output:
            related1=np.array(point)
            related2=np.array(point)
            related1[i]=point[i]+1
            related2[i]=point[i]-1
            output=np.vstack([output,related1,related2])
    return output

def integrate_DOS(file,fermi,shift=None,doping=None,force_positive=False):
    """Integrates the density of states given by QE (the summed DOS is generated when running the plot.pdos routine)
    This method serves a couple of porpuses and depending on the input gives different output. The integration is done with the trapezoidal method.
    JUST INTEGRATION
        file = total.DOS.file
        fermi = Energy up to which the integration is done
        returns the integrated DOS (aka number of electrons)
    Looking to shift the fermi level?
        shift = Float with the desired shift
        returns the necessary doping per cell (negative being electron doping)
    Looking to achieve certain doping?
        doping = Doping per cell (in electron units) (negative being electron doping)
        returns = Stimated Fermi level for that doping
    """
    data=np.loadtxt(file)
    #Force possitive DOS
    if force_positive==True:
        for i,num in enumerate(data[:,1]):
            if num<0:
                data[i,1]=0
                
    #Compute number of electrons up to charge neutrality        
    for i,num in enumerate(data[:,0]):
        if num>fermi:
            limit=i
            break
    num_elec=np.trapz(data[:limit,1],data[:limit,0])
    
    #If you are looking for doping in order to achieve certain shift in the Fermi level
    if shift!=None:
        for i,num in enumerate(data[:,0]):
            if num>fermi+shift:
                limit=i
                break
        doping=num_elec-np.trapz(data[:limit,1],data[:limit,0])
        return doping
    #If you want to predict the chemical potential shift from certain doping
    if doping!=None:
        tota_elec=num_elec-doping
        low=0
        for limit,dum in enumerate(data[:,0]):
            elec=np.trapz(data[:limit,1],data[:limit,0])
            high=data[limit,0]
            if elec>=tota_elec:
                break
            else:
                low=high
        return low+(high-low)/2
    return num_elec

def grep_kpoints_energies(filename,filetype='qe',vectors=np.array(None)):
    """Process the kpoints and energies for different file kinds, it outputs per rows:
    The first three numbers are the K-point coordinate in 2/pi*a units
    The rest of the numbers are the energies for that K point.

    filename = File with the bands
    filetype = qe (quantum espresso bands.pwo, scf.pwo, nscf.pwo)
               vaps (VASP EIGENVAL file)
               gnu (wannier90 band.dat, wantools bulkek.dat, QE bands.dat.gnu)
    vectors = np.array([[a1,a2,a3],...,[c1,c2,c3]])
              Real space lattice vectors in order to convert VASP K points (in crystal coord) to cartesian coord
    """
    filetype=filetype.lower()

    if filetype=="gnu" or filetype==None:
        print('NOT IMPLEMENTED')
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
            if re.search('End of .* calculation',line):
                results_line=i+1
                break
        data=np.zeros([num_points,num_bands+3])
        data_lines=lines[results_line:]
        i=-1
        j=1
        coord0=np.zeros(3)
        read_energies=False
        for line in data_lines:
            if re.search('Writing output',line):    #Reading is completed
                break
            elif re.search('bands \(ev\)',line):    #New k_point
                if '-' in line:
                    l=plot.__insert_space_before_minus(line)
                    l=l.split()
                else:
                    l=line.split()
                coord=np.array(l[2:5]).astype(np.float)  # Already in reciprocal cartesian coord (not like VASP)
                i=i+1
                j=3
                data[i,0:3]=coord
                read_energies=True
            elif re.search('occupation numbers',line):          #Stop reading energies when occupations
                read_energies=False
            elif read_energies==True:                           #Load energies
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

        data=np.zeros([num_points,num_bands+3])
        if vectors.all()!=None:                      #If there is no cell in the input
            K_vec=np.linalg.inv(vectors).transpose() #reciprocal vectors in columns
        else:
            K_vec=np.identity(3)
            print('CAUTIION: There was no cell introduced, therefore this are crystaline units')
        for i,num in enumerate(range(0,len(data_lines),num_bands+2)):    #load the x position
            line=data_lines[num]
            line=line.split()
            point=np.array(line).astype(np.float)[0:3]
            coord=np.matmul(point,K_vec)
            data[i,:3]=coord
        for band in range(1,num_bands+1):                             #load the bands
            i=0
            for num in range(band,len(data_lines),num_bands+2):
                line=data_lines[num]
                line=line.split()
                data[i,band+2]=line[1]
                i=i+1
    return data

def rotation(phi,u,radians=False):
    """
    Rotation matrix for an angle phi in a direction u
    """
    if radians==False:
        phi=(phi/360)*2*np.pi
    u=u/np.linalg.norm(u)
    x,y,z=u
    sin=np.sin(phi)
    cos=np.cos(phi)
    R=np.array(
    [[cos+(x**2)*(1-cos),x*y*(1-cos)-z*sin,x*z*(1-cos)+y*sin],
     [y*x*(1-cos)+z*sin,cos+(y**2)*(1-cos),y*z*(1-cos)-x*sin],
     [z*x*(1-cos)-y*sin,z*y*(1-cos)+x*sin,cos+(z**2)*(1-cos)]]
    )
    return R
