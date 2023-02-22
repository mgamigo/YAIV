#PYTHON module with experimental code that is not yet ready to publish (ignored in the rest of branches)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
import os
import re

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
        #data must be a [E,K]=[Energy,K_path] grid
        #E=1
        #previous=data[0,1]
        #for elem in data[1:,1]:
        #    if elem!=previous:
        #        E=E+1
        #    else:
        #        break
        #K=int(data.shape[0]/E)
        E=1
        previous=data[0,0]
        for elem in data[1:,0]:
            if elem==previous:
                E=E+1
            else:
                break
        K=int(data.shape[0]/E)
        print(K)
        print(E)
        if K*E != data.shape[0]:
            print("There is an error reading the data CHECK!!!")

        #INTERPOLATION
        x = np.linspace(data[0,0],data[-1,0], K)
        y = np.linspace(data[0,1],data[-1,1], E)
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


def surfdos(file,title=None,axis=None,save_interp=True,save_as=None,only_surfdos=False,figsize=None,colormap='plasma'):
    """Plots and interpolates the surface DOS generated by WannierTools
    file = dos.dat file from WT
    title = string with the title for the plot
    axis = Matplotlib axis in which you want to plot
    only_surfdos = if you want only the contribution of the surface
    colormap = colormap (plasma, viridis, inferno, cividis...)
    save_interp = Boolean controlling if you want to save the interpolation (to save time in future plots)
    save_as = string with the saving file and extensions

    It might get faster if we remove interpolation from plt.imshow
    """

    Z, data = __load_surfdos_data(file,only_surfdos,save_interp=save_interp)

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
    if axis == None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax=axis

    pos=ax.imshow(Z,interpolation='bilinear',aspect='auto',cmap=colormap,
                 origin='lower',extent=(data[0,0],data[-1,0],data[1,1],data[-1,1]))
    #cbar=fig.colorbar(pos, ax=ax)
    #cbar.set_ticks([])

    # draw gridlines
    plt.axhline(y=0,color='gray',linestyle='-',linewidth=0.5)
    plt.xticks(ticks,labels)
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
