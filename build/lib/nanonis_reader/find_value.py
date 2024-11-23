def barrier_height (kappa): # Input: kappa in 1/m unit.
    m_eV = 0.51099895e+6 # Electron mass in eV unit. (NOT eV/c^2 unit. Use m_e instead of m_e*c**2.)
    hbar = 6.582119569e-16 # hbar in eV*s unit.
    c = 2.99792458e+8 # Speed of light in m/s unit.
    return ( hbar*c*kappa )**2 / ( 2*m_eV )

def kappa (workfunction): # Input: work function in eV unit.
    m_eV = 0.51099895e+6 # Electron mass in eV unit. (NOT eV/c^2 unit. Use m_e instead of m_e*c**2.)
    hbar = 6.582119569e-16 # hbar in eV*s unit.
    c = 2.99792458e+8 # Speed of light.
    import numpy as np
    return np.sqrt(2 * m_eV * workfunction) / (hbar * c)

def nearest(array, value):
    import numpy as np
    array = np.asarray(array)
    # idx = (np.abs(array - value)).argmin()
    idx = np.where((np.abs(array - value)) == np.min((np.abs(array - value))))[0]
    return idx, array[idx]