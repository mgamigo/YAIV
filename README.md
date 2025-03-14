<a name="readme-top"></a>
<!-- PROJECT LOGO -->
<div align="center">
  <h1 align="center">YAIV</h3>
  <h3 align="center">Yet another Ab Initio Visualizer...</h3>
  <p align="center">
    A general purpose tool for condensed matter data analysis in jupyterlab.
    <br />
    <a href="https://github.com/mgamigo/YAIV/issues">Report Bug</a>
    ·
    <a href="https://github.com/mgamigo/YAIV/issues">Request Feature</a>
    <br />
	___
    <br />
    <a href="https://github.com/mgamigo/YAIV/tree/main/Tutorial"><strong>Explore the tutorials:</strong></a>
    <br />
    <a href="https://github.com/mgamigo/YAIV/blob/main/Tutorial/Plot_module.ipynb">Plotting</a>
    ·
    <a href="https://github.com/mgamigo/YAIV/blob/main/Tutorial/Convergence_module.ipynb">Convergence</a>  
    ·
    <a href="https://github.com/mgamigo/YAIV/blob/main/Tutorial/Utils_module.ipynb">Utilities</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#why">Why?</a></li>
      </ul>
    </li>
    <li><a href="#installation">Installation</a></li>
    <li>
	<a href="#current-tools">Current tools</a>
	<ul>
            <li><a href="#i-plot-module">Plotting tools</a></li>
	    <li><a href="#ii-convergence-module">Convergence analysis</a></li>
	    <li><a href="#iii-utils-module">Other utilities</a></li>
        </ul>
    </li>
    <li><a href="#examples">Examples</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project
YAIV is a collection of tools for plotting results of condensed matter ab initio codes such as *Quantum Espresso, VASP, Wannier90, Wannier Tools...* Although it can be used from the command line, the main intention of YAIV is to be used within JupyterLab, thereby allowing users to centralize the data analysis of a whole project into a single file. The goal is to provide both *(1)* fast and easy plotting defaults to glance over results, while *(2)* being flexible and powerful enough to generate publication-ready figures.

![gif demo](../media/demo.gif?raw=true)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Why?
> *A single file to rule them all...*

Most of the tools contained on YAIV are nothing more than glorified python scripts I needed during my PhD. Although python libraries for ab initio data analysis already exist, I found many of them being disigned to work within the command line (often required to be run from a certain directory). YAIV is aimed at providing useful ab initio analysis functionalities to those people willing to use a single JupyterLab file to organize their projects.

YAIV also intends to provide enough flexibility and modularity for most scenarios. To this end, useful [utilities](Tutorial/Utils_module.ipynb) are also provided in order to scrape data from the output of a variety of codes. Then, users can either further process the raw data or plot it in any desired way.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Installation

#### Create an isolated python enviroment
In case you want to create your own python enviroment and have it available in JupyterLab.
```sh
    virtualenv yaiv_env                                 #Create yor new enviroment
    source yaiv_env/bin/activate                        #Load the enviroment
    pip install ipykernel                               #In order to create a Kernel for this enviroment
    python -m ipykernel install --user --name=YAIV      #Install your new kernel with your desired name
    jupyter kernelspec list                             #Check that the new installed kernel appears
```
Now your new installed Kernel should be available in JupyterLab. You can select Kernel clicking at the top-right corner of JupyterLab.

#### Installing YAIV
You can either install from pip as:
```sh
   pip install yaiv
```

   Or cloning the git repository:
   
```sh
   git clone https://github.com/mgamigo/YAIV.git
   cd YAIV
   pip install .
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Current tools

All the functions are properly documented (remember that in JupyterLab all the documentation can be conviniently accesed with the **shift + tab** shortcut). All the tools are demostrated in the **[tutorials](Tutorial)**, here is a brief summary of the main modules of YAIV and their current tools:

### I. Plot module
Contains most of the plotting functionalities, currently being:
- **Electronic band structures** from Quantum Espresso, Vasp, Wannier90 and WannierTools.
    - Plotting results from different codes against each other.
- **Phonon spectra** from Quantum Espresso.
    - Plotting different phonon spectra. It can highlight the DFPT phonons from which the whole spectrum is interpolated.

### II. Convergence module
A variety of tools for the purpose of inspecting the convergence of different calculations by plotting the results in a digestible way. Currently supports:
- **Self-consistent calculations:** Given a folder with the quantum-espresso outputs it gives tools for the convergence analysis of various quantities respect to the cutoff, Kgrid and smearing.
- **DFPT Phonons:** Like the self-consisten convergence analyzer (also respect to cutoff, Kgrid and smeargin). But for the phonon frequencies.
- **Wannierizations:** Tools for the convergence analysis of the Wannier minimizations done with wannier90.

### III. Utils module
The utils module has a variety of utilities mostly focussed on scraping data from output files of different codes. This tools combined can be usefull for various porpuses. All the functions are demostrated in this [tutorial](/Tutorial/Utils_module.ipynb).
So far the code supports:
- **Grepping** tools (either by calling the function or using the **file class**):
	- Grepping the **number of electrons** from Quantum Espresso and VASP outputs.
	- Grepping the **Fermi level**.
	- Grep the **lattice parameters**.
	- Grep the **path** from a Quantum Espresso bands.pwi or madtyn.in input.
	- Grep the **path and HSP labels** from a KPATH in the [TQC website](https://www.topologicalquantumchemistry.fr/#/) format. (Enter in any compound and click in the "Download KPATH" link).
	- Grep the **phonon grid** from a Quantum Espresso ph.x output.
	- Grep the **total energy** from a Quantum Espresso ph.x output.
- **Transforming** tools (mainly usefull changes of coordinates):
	- **K_basis**: Obtaining the reciprocal lattice vectors.
	- **cartesian2cryst**: From cartesian to crystal coordinates.
	- **cryst2cartesian**: From crystal to cartesian coordinates.
	- **cartesian2spherical**: From cartesian to spherical coordinates.
	- **cryst2spherical**: From crystal to spherical coordinates.
---
## Examples
Here are some simple examples:
```py
plot.bands(file='DATA/bands/QE/results_bands/CsV3Sb5.bands.pwo',  #raw Quantum Espresso output file with the band structure
           KPATH='DATA/bands/KPATH',   #File with the Kpath (in order to plot the ticks at the High symmetry points)
           aux_file='DATA/bands/QE/results_scf/CsV3Sb5.scf.pwo', #File needed to read the number of electrons and lattice parameters
           title='Electronic bandstructures')    # A title of your liking
```
<img src="../media/bands.png" width="600">

```py
plot.phonons(file='DATA/phonons/2x2x2/results_matdyn/CsV3Sb5.freq.gp', #raw data file with the phonon spectrum
            KPATH='DATA/bands/KPATH',                                 #File with the Kpath (in order to plot the ticks at the High symmetry points)
            ph_out='DATA/phonons/2x2x2/results_ph/CsV3Sb5.ph.pwo',    #File with the phonon grid points and lattice vectors.
            title='Phonon spectra with the (2x2x2) grid highlighted!',   # A title of your liking
            grid=True,color='navy',linewidth=1)                        #Non-mandatory customization
```
<img src="../media/phonon.png" width="600">


```py
conv.kgrid.analysis(data='DATA/convergence/Kgrid/',         #Folder with your DFT outputs
		    title='K-grid convergence analysis')    #A title of your liking
```
<img src="../media/convergence.png" width="800">


```py
conv.wannier.w90(data='DATA/convergence/wannier90/NbGe2.wout',     #Wannier90 output file
                 title='Wannier minimization (66 WF)')             #A title of your liking
```
<img src="../media/wannier.png" width="800">

Combining YAIV tools with the usual **matplotlib sintax** one can generate complex plots as this one (check the [tutorial](Tutorial/Plot_module.ipynb)):

<img src="../media/collage.png" width="800">


_(For more examples, please refer to the [Tutorials](Tutorial))._

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [x] Plot module
    - [x] Plotting electronic band strucutres
    - [x] Plotting phonon spectra
    - [ ] Plotting densities of states (DOS)
    - [ ] ...
<!---
    - [ ] Plotting surface DOS generated by WannierTools (ARPES simulations)
    - [ ] Plotting contour energy DOS generated by WannierTools
    - [ ] 3D Band structure plots
-->
- [x] Utils module
    - [x] Grep tools to scrape data form OUTPUT files
    - [x] Transformation tools for easy changing of coordinates
    - [ ] ...
- [x] Convergence analysis tools
    - [x] Quantum Espresso self consistent calculations
    - [x] Quantum Espresso phonon spectra
    - [x] Wannierizations for Wannier90
    - [ ] ...
<!---
- [ ] Crystall structure analysis tools
    - [ ] Symmetry analysis
    - [ ] Visualization tools
    - [ ] ...
- [ ] Charge density wave analysis
    - [ ] Reading Quantum Espresso outputs
    - [ ] Distort crystal structures according to a given phonon
    - [ ] Linear combinations of condensing modes
    - [ ] Computing Born–Oppenheimer energy landscapes
    - [ ] ...
-->
- [ ] ...

##### Built With

[![NumPy][numpy.js]][numpy-url]  [![Matplotlib][matplo.js]][matplo-url]  [![ASE][ase.js]][ase-url] 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[numpy-url]: https://numpy.org/
[numpy.js]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white

[matplo-url]: https://matplotlib.org/
[matplo.js]: https://img.shields.io/badge/Matplotlib-%23000000.svg?style=for-the-badge&logo=Matplotlib&logoColor=white

[ase-url]: https://wiki.fysik.dtu.dk/ase/
[ase.js]: https://img.shields.io/badge/ASE-%23006f5c.svg?style=for-the-badge&logoColor=FF6719
