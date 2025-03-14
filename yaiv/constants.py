#PYTHON module for storing usefull constants:

Ry2eV = 13.6056980659
Boltz=1.380649e-23
u2Kg = 1.66054e-27
me = 9.1093837e-31
GHz2eV=4.13566553853598E-06
GHz2meV=4.13566553853598E-03
hartree2eV=27.2114
hartree2meV=27.2114*1000
Ry2jul= 2.179872e-18
eV2cm = 8065.544
Ry2cm_QE = 132.064879/1.20346372e-03 #fitted to match freqs
au2ang= 0.52917721067121
ang2metre=1e-10
hz2cm = 3.33565e-11  #checked

Ry2meV = Ry2eV*1000
Ry2K=Ry2jul/Boltz
K2Ry=1/Ry2K
cm2meV=1000/eV2cm
Ry2cm = Ry2eV*eV2cm
jul2eV = Ry2eV/Ry2jul
jul2meV = Ry2meV/Ry2jul
meV2jul = 1/jul2meV
bohr2ang=au2ang
bohr2metre=bohr2ang*ang2metre
au2metre=bohr2metre
pas2bar=1e-5
Kbar2Gpa=1/10
meV2hartree=1/hartree2meV
def smear2temp(smear):
    """Smearing is expected in Ry"""
    temp=Ry2K*smear
    return temp

def temp2smear(temp):
    """Smearing is returned in Ry"""
    smear=K2Ry*temp
    return smear

def F_Ry2meV(e):
    return e*Ry2eV*1000

def F_meV2Ry(e):
    return e/(Ry2eV*1000)
