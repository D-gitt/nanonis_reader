# nanonispy dependency removed — parsing handled by base.py via _vendor
import numpy as np
import os
import warnings
from scipy.optimize import curve_fit
try:
    from scipy.integrate import cumtrapz # scipy old version (before 1.14.0)
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz # scipy new version (after 1.14.0)


# ═══════════════════════════════════════════════════════════════════
#  STS spectroscopy (.dat)
# ═══════════════════════════════════════════════════════════════════

class sts:
    
    '''
    STS (Scanning Tunneling Spectroscopy) processor for .dat files.
    
    Args:
        instance : base.load
            Loaded Nanonis data file.
        sts_channel : str
            Channel name corresponding to the dI/dV value.
            'LI Demod 1 X (A)' by default.
    
    Methods:
        raw()       → (Bias (V), raw dIdV (a.u.))
        scaled()    → (Bias (V), dIdV (S))
        numerical() → (Bias (V), numerical dIdV (S))
        normalized()→ (Bias (V), normalized dIdV)
        iv()        → (Bias (V), Current (A))
    '''
    
    def __init__(self, instance, sts_channel='LI Demod 1 X (A)'):
        # Input validation
        if sts_channel not in ['LI Demod 1 X (A)', 'LI Demod 2 X (A)', 'LIX 1 omega (A)']:
            raise ValueError("sts_channel must be 'LI Demod 1 X (A)' or 'LI Demod 2 X (A)'")
        
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        self.channel = sts_channel

    def _resolve_channel(self, base_channel, sweep_direction='fwd', sweep_index=None):
        """Resolve a channel name (delegates to spectral_analysis.resolve_channel)."""
        from . import spectral_analysis as sa
        return sa.resolve_channel(self.signals, base_channel, sweep_direction, sweep_index)

    def _get_data(self, base_channel, sweep_direction='fwd', sweep_index=None):
        """
        Get signal data for a channel, supporting sweep_index='all' (returns stacked array).
        """
        resolved = self._resolve_channel(base_channel, sweep_direction, sweep_index)
        if isinstance(resolved, list):  # sweep_index='all'
            return np.stack([self.signals[ch] for ch in resolved])
        else:
            return self.signals[resolved]

    def raw(self, sweep_direction='fwd', sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Bias (V), raw dIdV (a.u.))
            If sweep_index='all': (Bias (V), stacked 2D array)
        '''
        data = self._get_data(self.channel, sweep_direction, sweep_index)
        return self.signals['Bias calc (V)'], data
    
    def scaled(self, sweep_direction='fwd', sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Bias (V), dIdV (S))
            If sweep_index='all': (Bias (V), stacked 2D array)
        '''
        raw = self._get_data(self.channel, sweep_direction, sweep_index)
        V, numerical_didv = self.numerical(sweep_direction=sweep_direction, sweep_index=sweep_index)
        if raw.ndim == 2:  # sweep_index='all'
            scale = np.median(numerical_didv / raw, axis=1, keepdims=True)
            return V, scale * raw
        else:
            return V, np.median(numerical_didv / raw) * raw
    
    def numerical(self, sweep_direction='fwd', sweep_index=None):
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
    
    def normalized(self, factor=0.2, delete_zero_bias=False, sweep_direction='fwd', sweep_index=None):
        """
        Returns
        -------
        tuple
            (Bias (V), normalized dIdV)
            If sweep_index='all': (Bias (V), stacked 2D array)
        """
        from . import spectral_analysis as sa
        V, dIdV = self.scaled(sweep_direction=sweep_direction, sweep_index=sweep_index)
        
        if dIdV.ndim == 2:  # sweep_index='all'
            results = []
            for i in range(dIdV.shape[0]):
                _, norm = sa.normalize_didv(V, dIdV[i], factor, delete_zero_bias)
                results.append(norm)
            if delete_zero_bias:
                zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
                V = np.delete(V, zero)
            return V, np.stack(results)
        else:
            return sa.normalize_didv(V, dIdV, factor, delete_zero_bias)
    
    def iv(self, sweep_direction='fwd', sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Bias (V), Current (A))
            If sweep_index='all': (Bias (V), stacked 2D array)
        '''        
        data = self._get_data('Current (A)', sweep_direction, sweep_index)
        return self.signals['Bias calc (V)'], data


# ═══════════════════════════════════════════════════════════════════
# FER (Field Emission Resonance) — inherits from sts
# ═══════════════════════════════════════════════════════════════════

class fer(sts):
    '''
    FER (Field Emission Resonance) spectroscopy processor for .dat files.
    Inherits all dI/dV methods from sts, and adds dZ/dV analysis.
    
    FER = Z-controller ON during bias sweep (Z-controller hold = FALSE).
    
    Additional Methods:
        dzdv_numerical() → (Bias (V), numerical dZdV (nm/V))
    '''
    
    def dzdv_numerical(self):
        '''
        Numerical dZ/dV from Z channel differentiation.
        Only meaningful for FER (Z-controller active during sweep).
        
        Returns
        -------
        tuple
            (Bias (V), numerical dZdV (nm/V))
        '''        
        step = self.signals['Bias calc (V)'][1] - self.signals['Bias calc (V)'][0]            
        dzdv = np.gradient(self.signals['Z (m)'] * 1e9, step, edge_order=2)
        return self.signals['Bias calc (V)'], dzdv


# ── Deprecated wrapper ──────────────────────────────────────────

class spectrum(sts):
    """Deprecated: use ``sts`` instead.
    
    Old: ``d.spec.didv_scaled()``
    New: ``d.sts.scaled()``
    """
    def __init__(self, instance, sts_channel='LI Demod 1 X (A)'):
        warnings.warn(
            "dat.spectrum is deprecated. Use dat.sts instead: "
            "d.sts.scaled(), d.sts.normalized(), etc.",
            DeprecationWarning, stacklevel=2
        )
        super().__init__(instance, sts_channel)

    def didv_raw(self, *args, **kwargs):
        return self.raw(*args, **kwargs)

    def didv_scaled(self, *args, **kwargs):
        return self.scaled(*args, **kwargs)

    def didv_numerical(self, *args, **kwargs):
        return self.numerical(*args, **kwargs)

    def didv_normalized(self, factor=0.2, delete_zero_bias=True,
                        sweep_direction=None, sweep_index=None):
        # Preserve old default: delete_zero_bias=True
        return self.normalized(factor, delete_zero_bias, sweep_direction, sweep_index)

    def iv_raw(self, *args, **kwargs):
        return self.iv(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════
#  I-z spectroscopy (.dat)
# ═══════════════════════════════════════════════════════════════════

class iz:
    
    '''
    I-z spectroscopy processor for .dat files.
    
    Args:
        instance : base.load
            Loaded Nanonis data file.
    
    Methods:
        raw()            → (Z rel (m), Current (A))
        barrier_height() → (apparent barrier height (eV), error (eV), slope (m⁻¹))
    '''
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals

    def _resolve_channel(self, base_channel, sweep_direction='fwd', sweep_index=None):
        """Resolve a channel name (delegates to spectral_analysis.resolve_channel)."""
        from . import spectral_analysis as sa
        return sa.resolve_channel(self.signals, base_channel, sweep_direction, sweep_index)
        
    def raw(self, sweep_direction='fwd', sweep_index=None):
        '''
        Returns
        -------
        tuple
            (Z rel (m), Current (A))
            If sweep_index='all': (Z rel (m), stacked 2D array)
        '''
        from . import spectral_analysis as sa
        sd = sweep_direction
        
        # Backward compatibility: sweep_dir='save all' → sweep_index='all'
        if sd == 'save all':
            sd = 'fwd'
            sweep_index = 'all'
        
        # AVG mode: average fwd + bwd
        if sd == 'AVG':
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
        
        # fwd/bwd with optional sweep_index — use resolve_channel
        resolved = self._resolve_channel('Current (A)', sd, sweep_index)
        if isinstance(resolved, list):  # sweep_index='all'
            I = np.stack([self.signals[ch] for ch in resolved])
        else:
            I = self.signals[resolved]
        return self.signals['Z rel (m)'], I

    def barrier_height(self, fitting_current_range=(1e-12, 10e-12)): # fitting_current_range: current range in A unit.
        '''
        Returns
        -------
        float
            (apparent barrier height (eV), error (eV), z-spec slope (m**-1))
        '''
        def linear(x, barr, b):
            return -2*( np.sqrt(2*0.51099895e+6*barr)/(6.582119569e-16*2.99792458e+8) )*x + b
    
        ############################## Set fitting range ##############################
        z, I = self.raw()[0], abs(self.raw()[1])
        idx = np.where( (fitting_current_range[0] <= I) & (I <= fitting_current_range[1]) ) # Filter with I
        ############################## Set fitting range ##############################
        
        popt, pcov = curve_fit (linear, z[idx], np.log(I[idx]), p0 = [1.2, 1.2])
        apparent_barrier_height, err = popt[0], np.sqrt(np.diag(pcov))[0]
        slope = -2*np.sqrt(2*0.51099895e+6*apparent_barrier_height)/(6.582119569e-16*2.99792458e+8)

        return apparent_barrier_height, err, slope


# ── Deprecated wrapper ──────────────────────────────────────────

class z_spectrum(iz):
    """Deprecated: use ``iz`` instead.
    
    Old: ``d.z_spec.get_iz()``
    New: ``d.iz.raw()``
    """
    def __init__(self, instance):
        warnings.warn(
            "dat.z_spectrum is deprecated. Use dat.iz instead: "
            "d.iz.raw(), d.iz.barrier_height(), etc.",
            DeprecationWarning, stacklevel=2
        )
        super().__init__(instance)

    def get_iz(self, *args, **kwargs):
        return self.raw(*args, **kwargs)

    def get_apparent_barrier_height(self, *args, **kwargs):
        return self.barrier_height(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════
#  Other .dat types (unchanged)
# ═══════════════════════════════════════════════════════════════════

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