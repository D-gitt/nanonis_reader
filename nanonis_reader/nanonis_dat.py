import nanonispy as nap
import numpy as np
import os
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz


# Only forward sweeps can be plotted right now. Need to add codes for backward.
class Load:
    def __init__(self, filepath):
        self.fname = os.path.basename(filepath)
        self.header = nap.read.Spec(filepath).header
        self.signals = nap.read.Spec(filepath).signals

class spectrum:
    
    '''
    Args:
        filepath : str
            Name of the Nanonis spectrum file to be loaded.
        sts_channel : str
            Channel name corresponding to the dI/dV value.
            'LI Demod 1 X (A)' by default.
        sweep_direction : str
            The sweep direction in which the dI/dV value is measured.
            'fwd' by default.
    
    Attributes (name : type):
        file : nanonispy.read.NanonisFile class
            Base class for Nanonis data files (grid, scan, point spectroscopy).
            Handles methods and parsing tasks common to all Nanonis files.
            https://github.com/underchemist/nanonispy/blob/master/nanonispy/read.py
        header : dict
            Header information of spectrum data.
        signals : dict
            Measured values in spectrum data.
        channel : str
            Channel name corresponding to the dI/dV value.
            'LI Demod 1 X (A)' by default.
        sweep_dir : str
            The sweep direction in which the dI/dV value is measured.
            'fwd' by default.

    Methods:
        didv_scaled(self)
            Returns the tuple: (Bias (V), dIdV (S))
        didv_numerical(self)
            Returns the tuple: (Bias (V), numerical dIdV (S))
        didv_normalized(self)
            Returns the tuple: (Bias (V), normalized dIdV)
        iv_raw(self)
            Returns the tuple: (Bias (V), Current (A))
    '''
    
    def __init__(self, instance, sts_channel = 'LI Demod 1 X (A)', sweep_direction = 'fwd'):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        self.channel = sts_channel # 'LI Demod 1 X (A)' or 'LI Demod 2 X (A)'
        self.sweep_dir = sweep_direction # 'fwd' or 'bwd'

    # def __init__(self, filepath, sts_channel = 'LI Demod 1 X (A)', sweep_direction = 'fwd'):
    #     import nanonispy as nap
    #     import os
    #     self.fname = os.path.basename(filepath)
    #     self.header = nap.read.Spec(filepath).header
    #     self.signals = nap.read.Spec(filepath).signals
    #     self.channel = sts_channel # 'LI Demod 1 X (A)' or 'LI Demod 2 X (A)'
    #     self.sweep_dir = sweep_direction # 'fwd' or 'bwd'

    def didv_raw(self):
        '''
        Returns
        -------
        tuple
            (Bias (V), raw dIdV (a.u.))
        '''
        if 'Current [AVG] (A)' in self.signals.keys():
            avg_channel = self.channel.replace(' (A)', ' [AVG] (A)')
            return self.signals['Bias calc (V)'], self.signals[avg_channel]
        else:
            return self.signals['Bias calc (V)'], self.signals[self.channel]
    
    def didv_scaled(self):
        '''
        Returns
        -------
        tuple
            (Bias (V), dIdV (S))
        '''
        if 'Current [AVG] (A)' in self.signals.keys():
            avg_channel = self.channel.replace(' (A)', ' [AVG] (A)')
            return self.signals['Bias calc (V)'], np.median(self.didv_numerical()[1]/self.signals[avg_channel])*self.signals[avg_channel]
        else:
            return self.signals['Bias calc (V)'], np.median(self.didv_numerical()[1]/self.signals[self.channel])*self.signals[self.channel]

    
    def didv_numerical(self):
        '''
        Returns
        -------
        tuple
            (Bias (V), numerical dIdV (S))
        '''        
        step = self.signals['Bias calc (V)'][1] - self.signals['Bias calc (V)'][0]
        if 'Current [AVG] (A)' in self.signals.keys():
            didv = np.gradient(self.signals['Current [AVG] (A)'], step, edge_order=2) # I-V curve를 직접 미분.
            return self.signals['Bias calc (V)'], didv
        else:
            didv = np.gradient(self.signals['Current (A)'], step, edge_order=2) # I-V curve를 직접 미분.
            return self.signals['Bias calc (V)'], didv
    
    def didv_normalized(self, factor=0.2, delete_zero_bias=True):
        '''
        Returns
        -------
        tuple
            (Bias (V), normalized dIdV)
        '''               
        # dIdV, V = self.signals[a.channel], self.signals['Bias calc (V)']
        V, dIdV = self.didv_scaled()
        I_cal = cumtrapz(dIdV, V, initial = 0)
        zero = np.argwhere ( abs(V) == np.min(abs(V)) )[0, 0] # The index where V = 0 or nearest to 0.
        popt, pcov = curve_fit (lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
        I_cal -= popt[1]

        # get total conductance I/V
        with np.errstate(divide='ignore'): # Ignore the warning of 'division by zero'.
            IV_cal = I_cal/V

        # Normalized_dIdV = dIdV / IV_cal
        # return np.delete(V, zero), np.delete(Normalized_dIdV, zero)

        delta = factor*np.median(IV_cal)
        Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
        if delete_zero_bias == False:
            return V, Normalized_dIdV
        else:
            return np.delete(V, zero), np.delete(Normalized_dIdV, zero)
    
    def iv_raw(self, save_all = False):
        '''
        Returns
        -------
        tuple
            (Bias (V), Current (A))
        '''        
        if 'Current [AVG] (A)' in self.signals.keys():
            return self.signals['Bias calc (V)'], self.signals['Current [AVG] (A)']
        else:
            return self.signals['Bias calc (V)'], self.signals['Current (A)']
            
        
    def dzdv_numerical(self):
        '''
        Returns
        -------
        tuple
            (Bias (V), numerical dZdV (nm/V))
        '''        
        step = self.signals['Bias calc (V)'][1] - self.signals['Bias calc (V)'][0]            
        dzdv = np.gradient(self.signals['Z (m)']*1e9, step, edge_order=2)
        
        return self.signals['Bias calc (V)'], dzdv

class z_spectrum:
    
    '''
    Args:
        filepath : str
            Name of the Nanonis spectrum file to be loaded.
        sweep_direction : str
            The sweep direction in which the I-z spectrum is measured.
            'AVG' by default.
    
    Attributes (name : type):
        file : nanonispy.read.NanonisFile class
            Base class for Nanonis data files (I-z spectroscopy).
        header : dict
            Header information of spectrum data.
        signals : dict
            Measured values in spectrum data.
        sweep_dir : str
            The sweep direction in which the I-z spectrum is measured.
            'AVG' by default.

    Methods:
        get(self)
            Returns the tuple: (Z rel (m), Current (A))
    '''
    
    def __init__(self, instance, sweep_direction = 'AVG'):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals    
        self.sweep_dir = sweep_direction # 'fwd' or 'bwd'

    # def __init__(self, filepath, sweep_direction = 'AVG'):
    #     import nanonispy as nap
    #     self.file = nap.read.NanonisFile(filepath) # Create an object corresponding to a specific data file.
    #     self.header = getattr(nap.read, self.file.filetype.capitalize())(self.file.fname).header
    #     self.signals = getattr(nap.read, self.file.filetype.capitalize())(self.file.fname).signals
    #     self.sweep_dir = sweep_direction # 'fwd' or 'bwd'
        
    def get_iz(self): # Better naming is welcome.
        '''
        Returns
        -------
        tuple
            (Z rel (m), Current (A))
        '''        
        if self.sweep_dir == 'fwd':
            I = self.signals['Current (A)']
        elif self.sweep_dir == 'bwd':
            I = self.signals['Current [bwd] (A)']
        elif self.sweep_dir == 'AVG':
            I = np.mean( [self.signals['Current (A)'], self.signals['Current [bwd] (A)']], axis = 0 )
        elif self.sweep_dir == 'save all':
            I = np.array([self.signals[channel] for channel in np.sort(list(self.signals.keys()))[:-3]])
                
        return self.signals['Z rel (m)'], I
    
    def get_apparent_barrier_height(self, fitting_current_range=(1e-12, 10e-12)): # fitting_current_range: current range in A unit.
        '''
        Returns
        -------
        float
            (apparent barrier height (eV), error (eV), z-spec slope (m**-1))
        '''
        def linear(x, barr, b):
            return -2*( np.sqrt(2*0.51099895e+6*barr)/(6.582119569e-16*2.99792458e+8) )*x + b
    
        ############################## Set fitting range ##############################
        z, I = self.get_iz()[0], abs(self.get_iz()[1])
        idx = np.where( (fitting_current_range[0] <= I) & (I <= fitting_current_range[1]) ) # Filter with I
        ############################## Set fitting range ##############################
        
        popt, pcov = curve_fit (linear, z[idx], np.log(I[idx]), p0 = [1.2, 1.2])
        apparent_barrier_height, err = popt[0], np.sqrt(np.diag(pcov))[0]
        slope = -2*np.sqrt(2*0.51099895e+6*apparent_barrier_height)/(6.582119569e-16*2.99792458e+8)

        return apparent_barrier_height, err, slope

        

class noise_spectrum:

    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def get_noise(self):
        '''
        Returns
        -------
        tuble
            (Frequency (Hz), Current PSD (A/sqrt(Hz)) or Z PSD (m/sqrt(Hz)))
        '''
        if 'Current PSD (A/sqrt(Hz))' in self.signals.keys():
            PSD = self.signals['Current PSD (A/sqrt(Hz))']
        elif 'Z PSD (m/sqrt(Hz))' in self.signals.keys():
            PSD = self.signals['Z PSD (m/sqrt(Hz))']
        return self.signals['Frequency (Hz)'], PSD