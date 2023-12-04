#PYTHON module with experimental code that is not yet ready to publish (ignored in the rest of branches)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
import os
import re

import yaiv.constants as const
import yaiv.utils as ut

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


def brillouin_zone_3d(cell,axis=None,basis=True,sides=True,line_width=1,reciprocal=False):
    """
    Plot Brillouin zone in a 3D ax, it uses as input the real space cell or a QE output file containing it
    ax = ax over with to plot (ax = fig.add_subplot...)
    basis = Whether to plot the basis
    sides = Whether to plot or not the sides
    line_width = Line width for the edges
    reciprocal = Whether or not the lattice is in reciprocal space
    """
    labels=False
    if type(cell)==str:
        cell=ut.grep_lattice(cell)
        labels=True
    if reciprocal==False:
        K_vec=ut.K_basis(cell)
    else:
        K_vec=cell

    if axis == None:
        plt.figure()
        ax=plt.axes(projection='3d')
    else:
        ax=axis

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

    if labels==True:
        ax.set_xlabel('$K_x\ (\AA^{-1})$')
        ax.set_ylabel('$K_y\ (\AA^{-1})$')
        ax.set_zlabel('$K_z\ (\AA^{-1})$')
    if axis == None:
#        axisEqual3D(ax)
        plt.show()
    axisEqual3D(ax)


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def miller_plane(miller,lattice,axis,label=None,size=0.15,alpha=0.4):
    """
    Displayes the desired Miller Plane.

    miller = Miller plane indices.
    lattice = Real space lattice.
    axis = 3D axis in which you want your Miller plane to be displayed
    label = Desired label for the Miller plane
    size = size of the miller_plane
    alpha = Transparency of the miller plane
    """
    ax=axis
    lim1=-size
    lim2=size
    reciprocal=ut.K_basis(lattice)
    miller=ut.cryst2cartesian(miller,reciprocal)
    if miller[0]!=0:
        y = np.linspace(lim1,lim2, 10)
        z = np.linspace(lim1,lim2, 10)
        y,z = np.meshgrid(y, z)
        x=-(miller[1]*y+miller[2]*z)/miller[0]
    elif miller[1]!=0:
        x = np.linspace(lim1,lim2, 10)
        z = np.linspace(lim1,lim2, 10)
        x,z = np.meshgrid(x, z)
        y=-(miller[0]*x+miller[2]*z)/miller[1]
    elif miller[2]!=0:
        x = np.linspace(lim1,lim2, 10)
        y = np.linspace(lim1,lim2, 10)
        x,y = np.meshgrid(x, y)
        z=-(miller[0]*x+miller[1]*y)/miller[2]
    ax.plot_surface(x, y, z, alpha=alpha,color='pink')
    if label!=None:
        ax.plot([0,0],[0,0],[0,0],label=label,color='pink')


def __load_surfdos_data(file,only_surfdos=False,save_interp=True,silent=True):
    """Loads the surfdos data of WannierTools and interpolates the data to plot in a imshow way
    returns Z, data
    being "Z" the interpolated set of data to plot with imshow and "data" the provided data in a numpy array.
    
    file = dat.dos file from WT, or the raw data as read by np.loadtxt()
    only_surfdos = if you want only the contribution of the surface
    save_interp = Boolean controlling if you want to save the interpolation (to save time in future plots)
    silent = Boolean if you want feedback of the interpolation.
    """
    interpolate = True
    if type(file)==str:
        if only_surfdos == False and os.path.exists(file+'_interp'):
            data=np.loadtxt(file)
            Z=np.loadtxt(file+'_interp')
            interpolate = False
        elif only_surfdos == True and os.path.exists(file+'_interp_surfonly'):
            data=np.loadtxt(file)
            Z=np.loadtxt(file+'_interp_surfonly')
            interpolate = False
        else:
            data=np.loadtxt(file)
    else:
        data=file
        save_interp=False

    if interpolate==True:
        if silent==False:
            print('Interpolating...')
        E=len(np.unique(data[:,1]))
        K=int(data.shape[0]/E)
        if K*E != data.shape[0]:
            print("There is an error reading the data CHECK!!!")

        #INTERPOLATION (to the same grid..., just a change of data format)
        x = np.linspace(np.min(data[:,0]),np.max(data[:,0]), K)
        y = np.linspace(np.min(data[:,1]),np.max(data[:,1]), E)
        #y = np.linspace(data[0,1],data[-1,1], E)
        X, Y = np.meshgrid(x, y)
        if only_surfdos == False:
            Z = griddata((data[:,0],data[:,1]),data[:,2],(X, Y), method='cubic')
            if save_interp == True:
                np.savetxt(file+'_interp',Z)
        elif only_surfdos == True:
            Z = griddata((data[:,0],data[:,1]),data[:,3],(X, Y), method='cubic')
            if save_interp == True:
                np.savetxt(file+'_interp_surfonly',Z)
    return Z, data

def surfdos(file,title=None,only_surfdos=False,axis=None,save_interp=True,colormap='plasma',display_ticks=True,figsize=None,save_as=None):
    """Plots and interpolates the surface DOS generated by WannierTools
    file = dos.dat file from WT or the output of __load_surfdos_data
    title = string with the title for the plot
    only_surfdos = if you want only the contribution of the surface
    axis = Matplotlib axis in which you want to plot
    save_interp = Boolean controlling if you want to save the interpolation (to save time in future plots)
    colormap = colormap (plasma, viridis, inferno, cividis...)
    display_ticks = Bolean (Whether you the ticks to be displayed) 
                            (read from the surfdos_bulk.gnu file)
    figsize = (int,int) => Size and shape of the figure
    save_as = string with the saving file and extensions

    It might get faster if we remove interpolation from plt.imshow
    """
    if type(file) == str:
        Z, data = __load_surfdos_data(file,only_surfdos,save_interp=save_interp)
    else:
        Z,data = file
        display_ticks=False

    #GNUFILE for xticks
    if display_ticks == True:
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
    if axis == None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax=axis
    pos=ax.imshow(Z,interpolation='bilinear',aspect='auto',cmap=colormap,
                  origin='lower',extent=(np.min(data[:,0]),np.max(data[:,0]),
                                         np.min(data[:,1]),np.max(data[:,1])))
    # draw gridlines
    if display_ticks == True:
        ax.set_xticks(ticks,labels)
        ax.axhline(y=0,color='gray',linestyle='-',linewidth=0.5)
        for tick in ticks:
            ax.axvline(tick,color='black',linestyle='-',linewidth=0.3)
    ax.set_ylabel('Energy (eV)')

    if title!=None:
        ax.set_title(title)
    if axis==None:
        plt.tight_layout()
        if save_as!=None:
            plt.savefig(save_as,dpi=500)
        plt.show()


def surfdos_all(folder,title=None,only_surfdos=False,save_interp=True,save_as=None,figsize=(9,3.5),colormap='plasma',display_ticks=True):
    """Plots and interpolates a group of surface DOS files generated by WannierTools
    folder = WT results folder where the dos.dat files are stored
    title = string with the title for the plot
    only_surfdos = if you want only the contribution of the surface
    axis = Matplotlib axis in which you want to plot
    save_interp = Boolean controlling if you want to save the interpolation (to save time in future plots)
    save_as = string with the saving file and extensions
    figsize = (int,int) => Size and shape of the figure
    colormap = colormap (plasma, viridis, inferno, cividis...)
    display_ticks = Bolean (Whether you the ticks to be displayed) 
                            (read from the surfdos_bulk.gnu file)

    It might get faster if we remove interpolation from plt.imshow
    """

    BULK =folder+'/dos.dat_bulk'
    SL =folder+'/dos.dat_l'
    SR =folder+'/dos.dat_r'

    #PLOTING
    fig,axs = plt.subplots(1,3,figsize=figsize)
    if title!=None:
        fig.suptitle(title)

    surfdos(BULK,'Bulk',axis=axs[0],save_interp=save_interp,colormap=colormap,display_ticks=display_ticks)
    surfdos(SL,'L surf.',only_surfdos,axs[1],save_interp,colormap,display_ticks)
    surfdos(SR,'R surf.',only_surfdos,axs[2],save_interp,colormap,display_ticks)

    axs[1].set_ylabel('')
    axs[2].set_ylabel('')

    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=300)
    plt.show()



def E_cut(file,HSP=None,title=None,axis=None,save_interp=True,save_as=None,figsize=None,colormap='plasma',text_color='black'):
    """Plots and interpolates the surface  generated by WannierTools
    file = dos.dat file from WT or the output of __load_surfdos_data
    HSP = The High symmetry points at (0,0), (0,0.5),(0.5,0.5) and (0.5,0). At that order...
    title = string with the title for the plot
    axis = Matplotlib axis in which you want to plot
    save_interp = Boolean controlling if you want to save the interpolation (to save time in future plots)
    save_as = string with the saving file and extensions
    figsize = (int,int) => Size and shape of the figure
    colormap = colormap (plasma, viridis, inferno, cividis...)
    text_color = Color for the text and HSP.

    It might get faster if we remove interpolation from plt.imshow
    """
    if type(file) == str:
        Z, data = __load_surfdos_data(file,only_surfdos=False,save_interp=save_interp)
    else:
        Z,data = file

    #PLOTING
    if axis == None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax=axis
        
    xmin,xmax,ymin,ymax=np.min(data[:,0]),np.max(data[:,0]),np.min(data[:,1]),np.max(data[:,1])
    pos=ax.imshow(Z,interpolation='bilinear',aspect='equal',cmap=colormap,
                  origin='lower',extent=(xmin,xmax,ymin,ymax))
    # draw gridlines
    ax.set_xlabel('$k_x$'), ax.set_xticks([])
    ax.set_ylabel('$k_y$'), ax.set_yticks([])
   
    if HSP!=None:
        c=text_color
        shift=np.max([xmax,ymax])*0.05
        xs=shift*1.3
        ys=shift
        
        ax.plot(0,0,'o', markersize=2,color=c) 
        ax.text(x=-xs, y=-ys, s=HSP[0], fontsize=16,color=c)
        ax.plot(0,ymax-ys*0.15,'o', markersize=2,color=c) 
        ax.text(x=-xs, y=ymax-ys*1.4, s=HSP[1], fontsize=16,color=c)
        ax.plot(xmax-xs*0.15,ymax-ys*0.15,'o', markersize=2,color=c) 
        ax.text(x=xmax-xs*1.4, y=ymax-ys*1.4, s=HSP[2], fontsize=16,color=c)
        ax.plot(xmax-xs*0.15,0,'o', markersize=2,color=c) 
        ax.text(x=xmax-xs*1.4, y=-ys, s=HSP[3], fontsize=16,color=c)

    if title!=None:
        ax.set_title(title)
    if axis==None:
        plt.tight_layout()
        if save_as!=None:
            plt.savefig(save_as,dpi=500)
        plt.show()


def read_relax(relax,kbar=False):
    relaxed=cell.read_spg(relax)
    species=relaxed[2]
    READ_stress=False
    READ_lat=False
    READ_atoms=False
    stress=None
    lattice=None
    atoms=None
    structures=[]
    stresses=[]
    
    lines = open(relax)
    for line in lines:
        if READ_stress==True:
            l=line.split()
            l=[float(item) for item in l]
            vec=np.array(l[:3])
            try:
                stress=np.vstack([stress,vec])
                if len(stress)==3:
                    READ_stress=False
            except NameError:
                stress=vec
        if READ_lat==True:
            l=line.split()
            l=[float(item) for item in l]
            vec=np.array(l[:3])
            try:
                lattice=np.vstack([lattice,vec])
                if len(lattice)==3:
                    READ_lat=False
            except NameError:
                lattice=vec
        if READ_atoms==True:
            l=line.split()
            l=[float(item) for item in l[1:]]
            vec=np.array(l)
            try:
                atoms=np.vstack([atoms,vec])
                if len(atoms)==len(species):
                    READ_atoms=False
                    struc=(lattice,atoms,species)
                    structures=structures+[struc]
                    stresses=stresses+[stress]
            except NameError:
                atoms=vec
        if re.search('total * stress',line):
            READ_stress=True
            del stress
        if re.search('CELL_PARAMETERS',line):
            READ_lat=True
            del lattice
        if re.search('ATOMIC_POSITIONS',line):
            READ_atoms=True
            del atoms
    
    if kbar==True:
        stresses=[stress*(const.Ry2jul/(const.bohr2metre**3))*const.pas2bar/1000 for stress in stresses]
    #structures=np.array(structures)
    stresses=np.array(stresses)
    
    return structures, stresses


def test_GAP(folder,steps=None,ranges=None,title=None,save_as=None):
    """For testing GAP results, so far all is preaty self-explanatory
    steps = 3d list with the number of steps for the histogram.
    ranges = 6d list with the bounds for the histogram.
    title = string with the title for the plot
    """
    if steps==None:
        steps=[100,100,100]
    
    GAP2GPa=(1/const.jul2eV)/(const.ang2metre**3)*1e-9
    energies=np.loadtxt(folder+'/energy')*1000
    forces=np.loadtxt(folder+'/forces')
    stress=np.loadtxt(folder+'/stress')*GAP2GPa
    
    if ranges==None:
        ranges=[np.min(energies[:,2]),np.max(energies[:,2]),
                np.min([np.min(forces[:,2]),np.min(forces[:,5]),np.min(forces[:,8])]),
                np.max([np.max(forces[:,2]),np.max(forces[:,5]),np.max(forces[:,8])]),
                np.min([np.min(stress[:,2]),np.min(stress[:,5]),np.min(stress[:,8]),np.min(stress[:,11]),np.min(stress[:,14]),np.min(stress[:,17])]),
                np.max([np.max(stress[:,2]),np.max(stress[:,5]),np.max(stress[:,8]),np.max(stress[:,11]),np.max(stress[:,14]),np.max(stress[:,17])])]
    configs=len(energies[:,0])
    atoms=int(len(forces[:,0])/configs)

    fig,[[ax1,ax2,ax3],[ax4,ax5,ax6]]=plt.subplots(2,3,figsize=(10,6))
    if title != None:
        t=title+'\n'+'Tested in '+str(configs)+' configs with '+str(atoms)+' atoms each'
    else:
        t='Tested in '+str(configs)+' configs with '+str(atoms)+' atoms each'
    fig.suptitle(t)
    shift=(np.max(energies[:,0])+np.min(energies[:,0]))/2
    ax1.plot((energies[:,0]-shift),(energies[:,1]-shift),'.')
    ax1.set_xlabel('DFT')
    ax1.set_ylabel('GAP')
    ax1.set_title('Energies (meV/atom)')
    ax1.grid()
    ax2.plot(forces[:,0],forces[:,1],'.',color='tab:red',alpha=0.7,label='x')
    ax2.plot(forces[:,0+3],forces[:,1+3],'.',color='tab:green',alpha=0.7,label='y')
    ax2.plot(forces[:,0+6],forces[:,1+6],'.',color='tab:blue',alpha=0.7,label='z')
    ax2.legend()
    ax2.set_xlabel('DFT')
    ax2.set_ylabel('GAP')
    ax2.set_title('Forces(eV/ang)')
    ax2.grid()
    ax3.plot(stress[:,0],stress[:,1],'.',color='tab:red',label='xx')
    ax3.plot(stress[:,0+3],stress[:,1+3],'.',color='tab:green',label='yy')
    ax3.plot(stress[:,0+6],stress[:,1+6],'.',color='tab:blue',label='zz')
    ax3.plot(stress[:,0+9],stress[:,1+9],'.',color='turquoise',label='yz')
    ax3.plot(stress[:,0+12],stress[:,1+12],'.',color='purple',label='xz')
    ax3.plot(stress[:,0+15],stress[:,1+15],'.',color='sienna',label='xy')
    ax3.legend()
    ax3.set_xlabel('DFT')
    ax3.set_ylabel('GAP')
    ax3.set_title('Stress (GPa)')
    ax3.grid()
    for ax in [ax1,ax2,ax3]:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ## now plot both limits against eachother
        ax.plot(lims, lims, '--', alpha=0.75, zorder=0,color='gray')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    x,y=np.histogram(energies[:,2],bins=steps[0],range=(ranges[0],ranges[1]),density=True)
    ax4.stairs(x,y,fill=True,alpha=0.5)
    ax4.set_xlabel('error (meV/atom)')
    ax4.set_ylabel('probability')
    ax4.set_yticks([])
    ax4.axvline(0,linestyle='--',color='gray')

    x,y=np.histogram(forces[:,2],bins=steps[1],range=(ranges[2],ranges[3]),density=True)
    ax5.stairs(x,y,color='tab:red',fill=True,alpha=0.5)
    x,y=np.histogram(forces[:,2+3],bins=steps[1],range=(ranges[2],ranges[3]),density=True)
    ax5.stairs(x,y,color='tab:green',fill=True,alpha=0.5)
    x,y=np.histogram(forces[:,2+6],bins=steps[1],range=(ranges[2],ranges[3]),density=True)
    ax5.stairs(x,y,color='tab:blue',fill=True,alpha=0.5)
    ax5.set_xlabel('error (eV/ang)')
    ax5.set_yticks([])
    ax5.axvline(0,linestyle='--',color='gray')
    
    x,y=np.histogram(stress[:,2],bins=steps[2],range=(ranges[4],ranges[5]),density=True)
    ax6.stairs(x,y,color='tab:red',fill=True,alpha=0.5)
    x,y=np.histogram(stress[:,2+3],bins=steps[2],range=(ranges[4],ranges[5]),density=True)
    ax6.stairs(x,y,color='tab:green',fill=True,alpha=0.5)
    x,y=np.histogram(stress[:,2+6],bins=steps[2],range=(ranges[4],ranges[5]),density=True)
    ax6.stairs(x,y,color='tab:blue',fill=True,alpha=0.5)
    x,y=np.histogram(stress[:,2+9],bins=steps[2],range=(ranges[4],ranges[5]),density=True)
    ax6.stairs(x,y,color='turquoise',fill=True,alpha=0.5)
    x,y=np.histogram(stress[:,2+12],bins=steps[2],range=(ranges[4],ranges[5]),density=True)
    ax6.stairs(x,y,color='purple',fill=True,alpha=0.5)
    x,y=np.histogram(stress[:,2+15],bins=steps[2],range=(ranges[4],ranges[5]),density=True)
    ax6.stairs(x,y,color='sienna',fill=True,alpha=0.5)
    ax6.set_yticks([])
    ax6.set_xlabel('error (GPa)')
    ax6.axvline(0,linestyle='--',color='gray')
    
    plt.tight_layout()
    if save_as!=None:
        plt.savefig(save_as,dpi=500)
    plt.show()

def grep_weyl_windings(file):
    """
    file : wanniercenter3D_Weyl.dat from a WannierTools FindWeylChirality calculation
    return CHIRALITY, PHASES
    """
    lines=open(file)
    for line in lines:
        if re.search('Chirality',line):
            CHIRALITY=np.array(line.split()[2:],dtype=float)
    PHASES=np.loadtxt(file,usecols=np.arange(len(CHIRALITY)+1))
    return CHIRALITY, PHASES

def plot_weyl_winding(data,point=1):
    """
    data : Either
            wanniercenter3D_Weyl.dat from a WannierTools FindWeylChirality calculation
            output of grep_weyl_winding(file)
    point : Point of which you want to plot the winding.
    """
    if type(data)==str:
        C,P=grep_weyl_windings(data)
    else:
        C,P=data[0],data[1]
    plt.figure()
    plt.plot(P[:,0],P[:,point],'o')
    plt.xlim(0,1),plt.ylim(0,1)
    plt.title('Point '+str(point)+' - Chern number '+str(C[point-1]))
    plt.tight_layout()
    plt.show()

def __grep_weyl_nodes(file,meV=True):
    """
    file : Nodes.dat from a WannierTools FindNodes calculation
    meV : Whether to return the energy and gap in meV (default is Hartrees)
    return Kcryst, Kcart, GAP, ENERGY
    """
    data=np.loadtxt(file)
    Kcart,GAP,E,Kcryst=data[:,:3],data[:,3],data[:,4],data[:,5:]
    if meV==True:
        E=const.hartree2meV*E
        GAP=const.hartree2eV*GAP
    for x in [Kcart,GAP,E,Kcryst]:
        x = np.array(x)
    return Kcryst, Kcart, GAP, E

def grep_weyl_nodes(file,meV=True):
    """
    file : Nodes.dat from a WannierTools FindNodes calculation (single or list of files)
    meV : Whether to return the energy and gap in meV (default is Hartrees)
    return Kcryst, Kcart, GAP, ENERGY
    """
    if type(file)== str:
        file=[file]
    for f in file:
        new = __grep_weyl_nodes(f,meV)
        try:
            Kcryst = np.vstack((Kcryst,new[0]))
            Kcart = np.vstack((Kcart,new[1]))
            GAP = np.hstack((GAP,new[2]))
            E = np.hstack((E,new[3]))
        except NameError:
            Kcryst,Kcart,GAP,E = new
    return Kcryst, Kcart, GAP, E

def __grep_weyl_chirality(file):
    """
    file/files : WT.OUT from a WannierTools FindWeylChirality calculation
    return Kcryst, Kcart, chirality
    """
    READ_CHIRALITIES=False
    lines=open(file)
    for line in lines:
        if re.search('Time cost for Weyl',line):
            READ_CHIRALITIES=False
        elif READ_CHIRALITIES==True:
            l=np.array(line.split(),dtype=float)
            try:
                data=np.vstack((data,l))
            except NameError:
                data=l
        elif re.search('Chirality',line) and re.search('#',line):
            READ_CHIRALITIES=True
    Kcryst,Kcart,chirality=data[:,:3],data[:,3:6],data[:,6]
    for x in [Kcryst,Kcart,chirality]:
        x = np.array(x)
    return Kcryst, Kcart, chirality

def grep_weyl_chirality(file):
    """
    file/files : WT.OUT from a WannierTools FindWeylChirality calculation (single or list of files)
    return Kcryst, Kcart, chirality
    """
    if type(file)== str:
        file=[file]
    for f in file:
        new = __grep_weyl_chirality(f)
        try:
            Kcryst = np.vstack((Kcryst,new[0]))
            Kcart = np.vstack((Kcart,new[1]))
            chirality = np.hstack((chirality,new[2]))
        except NameError:
            Kcryst,Kcart,chirality = new
    return Kcryst, Kcart, chirality


def filter_weyl_nodes(k,gap,energy,chirality,E_range=1000,GAP_range=0):
    """
    Given a list of Weyl points it filters to the ones in a certain energy range and GAP range
    k = List of k points
    gap = List of gaps
    energy = List of energies
    chirality = List of chiralities
    return K, GAP, E, C           
    """
    if type(E_range)!=list:
        E_range=[-E_range,E_range]
    for i in range(len(chirality)):
        if E_range[0] <= energy[i] <= E_range[1]:
            if gap[i]<=GAP_range:
                try:
                    K=np.vstack((K,k[i]))
                    E=np.hstack((E,energy[i]))
                    GAP=np.hstack((GAP,gap[i]))
                    C=np.hstack((C,chirality[i]))
                except NameError:
                    K,GAP,E,C=k[i],gap[i],energy[i],chirality[i]
    return K, GAP, E, C        


def symmetrize_weyl_nodes(k,gap,energy,chirality,lattice,symmetry,precision=2,truncate=None,silent=False):
    """
    Given a list of Weyl points it filters to the ones in a certain energy range and GAP range.
    CAUTION: This only works for rotations (which keep the chirality invariant)
    k = List of k points in cartesian units
    gap = List of gaps
    energy = List of energies
    chirality = List of chiralities
    lattice = Lattice in order to find which points lay inside the Brillouin zone
    symmetry = Symmetry operation
    precision = Number of decimals in order to check distance between two Weyl points and decide whether they are same point or not.
    truncate = Remove all points with higher distance to GM than this value. (usefull to fix unwanted Weyl points outside BZ)
    silent = Bolean controling whether you want printed output
    return K, GAP, E, C
    """
    kpoints=k
    initial=len(kpoints)
    #Get vectores related by symmetry
    p=precision #precision
    Kin,Gin,Ein,Cin,S=k,gap,energy,chirality,symmetry #input
    K,GAP,E,C=[],[],[],[]
    for i,k in enumerate(kpoints):
        K,GAP,E,C=K+[k],GAP+[Gin[i]],E+[Ein[i]],C+[Cin[i]]
        new=np.matmul(S,k)
        while np.around(np.linalg.norm(new-k),decimals=p) != 0:
            K,GAP,E,C=K+[new],GAP+[Gin[i]],E+[Ein[i]],C+[Cin[i]]
            new=np.matmul(S,new)
    for x in [K,GAP,E,C]:
        x=np.array(x)

    #Get vectores related by lattice
    rlat=ut.K_basis(lattice)
    K=ut.cartesian2cryst(K,rlat,list_of_vec=True)
    for i,k in enumerate(K):
        if abs(np.around(k[0],decimals=p))==0.5 or abs(np.around(k[1],decimals=p))==0.5 or abs(np.around(k[2],decimals=p))==0.5:
            new=ut.__expand_star(k)
            try:
                NEW_K=np.vstack((NEW_K,new))
                NEW_E=np.hstack((NEW_E,[E[i]]*len(new)))
                NEW_C=np.hstack((NEW_C,[C[i]]*len(new)))
                NEW_GAP=np.hstack((NEW_GAP,[GAP[i]]*len(new)))
            except NameError:
                NEW_K=new
                NEW_E=[E[i]]*len(new)
                NEW_C=[C[i]]*len(new)
                NEW_GAP=[GAP[i]]*len(new)
    #remove outside BZ
    tmp_list=[]
    for i,k in enumerate(NEW_K):
        if abs(np.around(k[0],decimals=p))>0.5 or abs(np.around(k[1],decimals=p))>0.5 or abs(np.around(k[2],decimals=p))>0.5:
            tmp_list=tmp_list+[i]
    NEW_K=np.delete(NEW_K,tmp_list,0)
    NEW_GAP,NEW_E,NEW_C=np.delete(NEW_GAP,tmp_list),np.delete(NEW_E,tmp_list),np.delete(NEW_C,tmp_list)
    
    K,E,C,GAP=np.vstack((K,NEW_K)),np.hstack((E,NEW_E)),np.hstack((C,NEW_C)),np.hstack((GAP,NEW_GAP))
    K=ut.cryst2cartesian(K,rlat,list_of_vec=True)

    #remove duplicates
    i=0
    while i<len(K):
        diff_K = np.array([np.around(np.linalg.norm(x-K[i]),decimals=p) for x in K])
        diff_E = np.array([np.around(np.linalg.norm(x-E[i]),decimals=p) for x in E])
        diff_C = np.array([np.around(np.linalg.norm(x-C[i]),decimals=p) for x in C])
        diff_GAP = np.array([np.around(np.linalg.norm(x-GAP[i]),decimals=p) for x in GAP])
        tmp_list=np.where(np.around(diff_K,decimals=p)==0.0)[0]
        aux_list=np.where(diff_C==0.0)[0]
        tmp_list=[item for item in tmp_list if item in aux_list]
#        aux_list=np.where(diff_E<10)[0]  Problematic to check energies, because of the lack os symmetries...
#        tmp_list=[item for item in tmp_list if item in aux_list]
        K=np.delete(K,tmp_list[1:],0)
        GAP,E,C=np.delete(GAP,tmp_list[1:]),np.delete(E,tmp_list[1:]),np.delete(C,tmp_list[1:])
        i=i+1
    if truncate!=None:
        dist = np.array([np.linalg.norm(x) for x in K])
        tmp_list=np.where(dist>truncate)[0]
        K=np.delete(K,tmp_list,0)
        GAP,E,C=np.delete(GAP,tmp_list),np.delete(E,tmp_list),np.delete(C,tmp_list)
    final=len(K)
    if silent==False:
        print('Symemtrizing from',initial,'to',final,'Weyl points')
    
    return K,GAP,E,C
