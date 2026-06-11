# nanonispy dependency removed — parsing handled by base.py via _vendor
import numpy as np
import os
from scipy.optimize import curve_fit
try:
    from scipy.integrate import cumtrapz # scipy old version (before 1.14.0)
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz # scipy new version (after 1.14.0)



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
    
    def __init__(self, instance, sts_channel='LI Demod 1 X (A)', sweep_direction='fwd'):
        # Input validation
        if sts_channel not in ['LI Demod 1 X (A)', 'LI Demod 2 X (A)', 'LIX 1 omega (A)']:
            raise ValueError("sts_channel must be 'LI Demod 1 X (A)' or 'LI Demod 2 X (A)'")
        if sweep_direction not in ['fwd', 'bwd']:
            raise ValueError("sweep_direction must be 'fwd' or 'bwd'")
        
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        self.channel = sts_channel
        self.sweep_dir = sweep_direction

    def _resolve_channel(self, base_channel, sweep_direction=None, sweep_index=None):
        """Resolve a channel name (delegates to spectral_analysis.resolve_channel)."""
        from . import spectral_analysis as sa
        sd = sweep_direction if sweep_direction is not None else self.sweep_dir
        return sa.resolve_channel(self.signals, base_channel, sd, sweep_index)

    def _get_data(self, base_channel, sweep_direction=None, sweep_index=None):
        """
        Get signal data for a channel, supporting sweep_index='all' (returns stacked array).
        """
        resolved = self._resolve_channel(base_channel, sweep_direction, sweep_index)
        if isinstance(resolved, list):  # sweep_index='all'
            return np.stack([self.signals[ch] for ch in resolved])
        else:
            return self.signals[resolved]

    def didv_raw(self, sweep_direction=None, sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Bias (V), raw dIdV (a.u.))
            If sweep_index='all': (Bias (V), stacked 2D array)
        '''
        data = self._get_data(self.channel, sweep_direction, sweep_index)
        return self.signals['Bias calc (V)'], data
    
    def didv_scaled(self, sweep_direction=None, sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Bias (V), dIdV (S))
            If sweep_index='all': (Bias (V), stacked 2D array)
        '''
        raw = self._get_data(self.channel, sweep_direction, sweep_index)
        V, numerical_didv = self.didv_numerical(sweep_direction=sweep_direction, sweep_index=sweep_index)
        if raw.ndim == 2:  # sweep_index='all'
            scale = np.median(numerical_didv / raw, axis=1, keepdims=True)
            return V, scale * raw
        else:
            return V, np.median(numerical_didv / raw) * raw
    
    def didv_numerical(self, sweep_direction=None, sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Bias (V), numerical dIdV (S))
            If sweep_index='all': (Bias (V), stacked 2D array)
        '''        
        step = self.signals['Bias calc (V)'][1] - self.signals['Bias calc (V)'][0]
        current = self._get_data('Current (A)', sweep_direction, sweep_index)
        if current.ndim == 2:  # sweep_index='all'
            didv = np.gradient(current, step, axis=1, edge_order=2)
        else:
            didv = np.gradient(current, step, edge_order=2)
        return self.signals['Bias calc (V)'], didv
    
    def didv_normalized(self, factor=0.2, delete_zero_bias=True, sweep_direction=None, sweep_index=None):
        """
        Returns
        -------
        tuple
            (Bias (V), normalized dIdV)
            If sweep_index='all': (Bias (V), stacked 2D array)
        """
        V, dIdV = self.didv_scaled(sweep_direction=sweep_direction, sweep_index=sweep_index)
        
        if dIdV.ndim == 2:  # sweep_index='all'
            results = []
            for i in range(dIdV.shape[0]):
                _, norm = self._normalize_single(V, dIdV[i], factor, delete_zero_bias)
                results.append(norm)
            if delete_zero_bias:
                zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
                V = np.delete(V, zero)
            return V, np.stack(results)
        else:
            return self._normalize_single(V, dIdV, factor, delete_zero_bias)
    
    def _normalize_single(self, V, dIdV, factor=0.2, delete_zero_bias=True):
        """Normalize a single dIdV spectrum."""
        I_cal = cumtrapz(dIdV, V, initial=0)

        zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
        popt, _ = curve_fit(lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
        I_cal -= popt[1]

        with np.errstate(divide='ignore', invalid='ignore'):
            IV_cal = I_cal/V
            
            # 극한값(로피탈의 정리)에 의해 V->0 일 때 I/V = dI/dV 이며, 이는 피팅된 직선의 기울기(popt[0])와 같습니다.
            if abs(V[zero]) < 1e-5:
                IV_cal[zero] = popt[0]

        delta = factor*np.nanmedian(IV_cal)
        Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
        
        if delete_zero_bias:
            return np.delete(V, zero), np.delete(Normalized_dIdV, zero)
        else:
            return V, Normalized_dIdV

    def iv_raw(self, sweep_direction=None, sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Bias (V), Current (A))
            If sweep_index='all': (Bias (V), stacked 2D array)
        '''        
        data = self._get_data('Current (A)', sweep_direction, sweep_index)
        return self.signals['Bias calc (V)'], data
    
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
        
    def get_iz(self, sweep_direction=None, sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Z rel (m), Current (A))
            If sweep_index='all': (Z rel (m), stacked 2D array)
        '''
        from . import spectral_analysis as sa
        sd = sweep_direction if sweep_direction is not None else self.sweep_dir
        
        # Backward compatibility: sweep_dir='save all' → sweep_index='all'
        if sd == 'save all':
            sd = 'fwd'
            sweep_index = 'all'
        
        if sweep_index == 'all':
            channels = sa.find_sweep_channels(self.signals, 'Current (A)', sd)
            I = np.stack([self.signals[ch] for ch in channels])
            return self.signals['Z rel (m)'], I
        elif sweep_index is not None:
            ch = sa.get_channel_name('Current (A)', sweep_direction=sd, sweep_index=sweep_index)
            return self.signals['Z rel (m)'], self.signals[ch]
        elif sd == 'AVG':
            # fwd + bwd average
            if sa.has_averaged_data(self.signals):
                I_fwd = self.signals.get('Current [AVG] (A)')
                I_bwd = self.signals.get('Current [AVG] [bwd] (A)')
            else:
                I_fwd = self.signals.get('Current (A)')
                I_bwd = self.signals.get('Current [bwd] (A)')
            if I_fwd is not None and I_bwd is not None:
                I = np.mean([I_fwd, I_bwd], axis=0)
            elif I_fwd is not None:
                I = I_fwd
            else:
                I = I_bwd
            return self.signals['Z rel (m)'], I
        else:
            # fwd or bwd
            if sa.has_averaged_data(self.signals):
                ch = f"Current [AVG] (A)" if sd == 'fwd' else f"Current [AVG] [bwd] (A)"
            else:
                ch = f"Current (A)" if sd == 'fwd' else f"Current [bwd] (A)"
            return self.signals['Z rel (m)'], self.signals[ch]

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
        tuple
            (Frequency (Hz), Current PSD (A/sqrt(Hz)) or Z PSD (m/sqrt(Hz)))
        '''
        if 'Current PSD (A/sqrt(Hz))' in self.signals.keys():
            PSD = self.signals['Current PSD (A/sqrt(Hz))']
        elif 'Z PSD (m/sqrt(Hz))' in self.signals.keys():
            PSD = self.signals['Z PSD (m/sqrt(Hz))']
        return self.signals['Frequency (Hz)'], PSD

class history_data:

    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def get_history(self, channel='Z (m)'):
        '''
        Returns
        -------
        tuple
            (Time (ms), History data)
            History data:
                'Current (A)', 'Z (m)', 'Input 2 (V)', ...
                'Z (m)' is the default history data.
        '''
        sample_period = np.int64(self.header['Sample Period (ms)'])
        
        history = self.signals[channel]
        time = np.arange(np.shape(history)[0]) * sample_period

        return time, history

class longterm_data:

    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def get_z_longterm_chart(self):
        '''
        Returns
        -------
        tuple
            (Rel. Time (s), Z (m))
        '''
        t = self.signals['Rel. Time (s)']
        z = self.signals['Z (m)']

        return t, z