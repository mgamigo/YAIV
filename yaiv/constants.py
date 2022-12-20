#PYTHON module for storing various constants:

Ry2eV = 13.6056980659
Ry2jul= 2.179872e-18
eV2cm = 8065.544
Ry2cm = Ry2eV*eV2cm
jul2eV = Ry2eV/Ry2jul
Ry2cm_QE = 132.064879/1.20346372e-03 #fitted to match freqs
au2ang= 0.52917721067121
ang2metre=1e-10
hz2cm = 3.33565e-11  #checked
bohr2ang=au2ang
bohr2metre=bohr2ang*ang2metre
au2metre=bohr2metre
pas2bar=1e-5
Boltz=1.380649e-23
u2Kg = 1.66054e-27
me = 9.1093837e-31

def smear2temp(smear):
    temp=(Ry2jul*smear)/Boltz
    return temp

def temp2smear(temp):
    smear=(Boltz*temp)/Ry2jul
    return smear

def F_Ry2meV(e):
    return e*Ry2eV*1000

def F_meV2Ry(e):
    return e/(Ry2eV*1000)
