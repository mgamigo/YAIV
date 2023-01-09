#PYTHON module for cutoff convergence analysis

import re
import glob
import numpy as np
import matplotlib.pyplot as plt

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

