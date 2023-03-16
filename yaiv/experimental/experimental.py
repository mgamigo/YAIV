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


def brillouin_zone_3d(cell,axis=None,basis=True,sides=True,line_width=1,reciprocal=True):
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


def miller_plane(miller,lattice,axis,label=None):
    """
    Displayes the desired Miller Plane.

    miller = Miller plane indices.
    lattice = Real space lattice.
    axis = 3D axis in which you want your Miller plane to be displayed
    label = Desired label for the Miller plane
    """
    ax=axis
    lim1=-0.15
    lim2=0.15
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
    ax.plot_surface(x, y, z, alpha=0.4,color='pink')
    if label!=None:
        ax.plot([0,0],[0,0],[0,0],label=label,color='pink')


def __load_surfdos_data(file,only_surfdos=False,save_interp=True):
    """Loads the surfdos data of WannierTools and interpolates the data to plot in a imshow way
    returns Z, data
    being "Z" the interpolated set of data to plot with imshow and "data" the provided data in a numpy array.
    
    file = dat.dos file from WT
    only_surfdos = if you want only the contribution of the surface
    save_interp = Boolean controlling if you want to save the interpolation (to save time in future plots)
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
        print('Interpolating...')
        E=1
        previous=data[0,0]
        for elem in data[1:,0]:
            if elem==previous:
                E=E+1
            else:
                break
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
        print('Done')
    return Z, data

def surfdos(file,title=None,only_surfdos=False,axis=None,save_interp=True,save_as=None,figsize=None,colormap='plasma',display_ticks=True):
    """Plots and interpolates the surface DOS generated by WannierTools
    file = dos.dat file from WT or the output of __load_surfdos_data
    title = string with the title for the plot
    only_surfdos = if you want only the contribution of the surface
    axis = Matplotlib axis in which you want to plot
    save_interp = Boolean controlling if you want to save the interpolation (to save time in future plots)
    save_as = string with the saving file and extensions
    figsize = (int,int) => Size and shape of the figure
    colormap = colormap (plasma, viridis, inferno, cividis...)
    display_ticks = Bolean (Whether you the ticks to be displayed)

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
        plt.xticks(ticks,labels)
        plt.axhline(y=0,color='gray',linestyle='-',linewidth=0.5)
        for tick in ticks:
            plt.axvline(tick,color='black',linestyle='-',linewidth=0.3)
    ax.set_ylabel('Energy (eV)')

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


def test_GAP(folder,steps=None,ranges=None,save_as=None):
    """For testing GAP results, so far all is preaty self-explanatory
    steps = 3d list with the number of steps for the histogram.
    ranges = 6d list with the bounds for the histogram.
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
    fig.suptitle('Trained in '+str(configs)+' configs with '+str(atoms)+' atoms each')
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
