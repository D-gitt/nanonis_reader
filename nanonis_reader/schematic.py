import numpy as np
from scipy.interpolate import UnivariateSpline

def band_EI (x, Ec, Ev, sign = '+', delta = 100): # delta: order parameter
    Erec = (Ec + Ev + np.sqrt ( (Ec - Ev)**2 + 4*(delta**2) )) / 2
    Erev = (Ec + Ev - np.sqrt ( (Ec - Ev)**2 + 4*(delta**2) )) / 2
    if sign == '+':
        return Erec
    elif sign == '-':
        return Erev

def DOS (band, E, thermal):
    dos = np.zeros(len(E))
    dE = E[1]-E[0]
    for i, energy in enumerate(E):
        if thermal >= 0:
            # tmp1 = []
            # for Ei in np.arange(energy-thermal, energy+thermal, dE):
            #     # tmp = np.where((energy-thermal <= band) & (band <= energy+thermal))[0]
            #     # tmp = UnivariateSpline(k, band-energy, s=0).roots()
            #     tmp2 = UnivariateSpline(k, band-Ei, s=0).roots()
            #     tmp1.append(len(tmp2))
            # # dos[i] = len(tmp)
            # dos[i] = np.sum(tmp1)
            
            tmp = np.where((energy-thermal <= band) & (band <= energy+thermal))[0]
            dos[i] = len(tmp)
            

        else:
            raise Exception ('Thermal broadening must be positive.')
    
    return dos