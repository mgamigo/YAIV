#PYTHON module with experimental code that is not yet ready to publish (ignored in the rest of branches)

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


def miller_plane(miller,ax,label=None):
    lim1=-0.15
    lim2=0.15
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
