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
import yaiv.cell_analyzer as cell
import yaiv.constants as cons
import spglib as spg

#Utilities*****************************************************************************************

def fermi_surface(file):
    """Just launches fermisurfer for you from the python notebook"""
    cmd='fermisurfer '+file
    subprocess.call(cmd,shell=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
