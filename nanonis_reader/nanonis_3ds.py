class Load:
    def __init__(self, filepath):
        import nanonispy as nap
        import os
        self.fname = os.path.basename(filepath)
        self.header = nap.read.Grid(filepath).header
        self.signals = nap.read.Grid(filepath).signals

        
class Topo:
        
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def get_z (self, processing = 'raw'):
        if processing == 'raw':
            return self.raw()
        elif processing == 'subtract average':
            return self.subtract_average()
        elif processing == 'subtract linear fit':
            return self.subtract_linear_fit()
        elif processing == 'subtract parabolic fit':
            return self.subtract_parabolic_fit()
        elif processing == 'differentiate':
            return self.differentiate()
        
    def raw (self):
        import numpy as np
        tmp = self.signals['topo']
        z = np.where(tmp == 0, np.nan, tmp)
        return z
    
    def subtract_average (self):
        import warnings
        import numpy as np
        warnings.filterwarnings(action='ignore')
        z = self.raw()
        z_subav = np.zeros(np.shape(z))
        lines = np.shape(z)[0]
        for i in range(lines):
            z_subav[i] = z[i] - np.nanmean(z[i])
        return z_subav

    def subtract_linear_fit (self):
        import numpy as np
        from scipy.optimize import curve_fit
        def f_lin(x, a, b): return a*x + b
        xrange = round(self.header['size_xy'][0] * 1e9)*1e-9
        print (xrange)
        z = self.raw()
        z_sublf = np.zeros(np.shape(z))
        lines, pixels = np.shape(z)
        for i in range(lines):
            if np.shape(np.where(np.isnan(z))[0])[0] != 0: # image에 nan값이 포함되어 있을 경우 (== scan을 도중에 멈추었을 경우)
                if i < np.min(np.where(np.isnan(z))[0]):
                    x = np.linspace(0, xrange, pixels)
                    popt, pcov = curve_fit(f_lin, x, z[i])
                    z_sublf[i] = z[i] - f_lin(x, *popt)
                else:
                    z_sublf[i] = np.nan
            else:
                x = np.linspace(0, xrange, pixels)
                popt, pcov = curve_fit(f_lin, x, z[i]) # x - ith line: linear fitting
                z_sublf[i] = z[i] - f_lin(x, *popt)

        return z_sublf

    def subtract_parabolic_fit (self):
        import numpy as np
        from scipy.optimize import curve_fit
        def f_parab(x, a, b, c): return a*(x**2) + b*x + c
        xrange = round(self.header['size_xy'][0] * 1e9)*1e-9
        print (xrange)
        z = self.raw()
        z_subpf = np.zeros(np.shape(z))
        lines, pixels = np.shape(z)
        for i in range(lines):
            if np.shape(np.where(np.isnan(z))[0])[0] != 0: # image에 nan값이 포함되어 있을 경우 (== scan을 도중에 멈추었을 경우)
                if i < np.min(np.where(np.isnan(z))[0]):
                    x = np.linspace(0, xrange, pixels)
                    popt, pcov = curve_fit(f_parab, x, z[i])
                    z_subpf[i] = z[i] - f_parab(x, *popt)
                else:
                    z_subpf[i] = np.nan
            else:
                x = np.linspace(0, xrange, pixels)
                popt, pcov = curve_fit(f_parab, x, z[i]) # x - ith line: linear fitting
                z_subpf[i] = z[i] - f_parab(x, *popt)

        return z_subpf
    
    def differentiate (self):
        import numpy as np
        xrange, pixels = round(self.header['size_xy'][0] * 1e9)*1e-9, int(self.header['dim_px'][0])
        dx = xrange / pixels
        z = self.raw()
        z_deriv = np.zeros(np.shape(z))
        lines = np.shape(z)[0]
        for i in range(lines):
            z_deriv[i] = np.gradient(z[i], dx, edge_order = 2) # dI/dV curve를 직접 미분. --> d^2I/dV^2
        
        return z_deriv

    
class Map: # dIdV, I-z spec, apparent barrier map
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        
    def get_didvmap (self, sweep_idx, channel = 'LI Demod 1 X (A)'):
        didv = self.signals[channel][:, :, sweep_idx]
        return didv
    
    def get_currentmap (self, sweep_idx, sweep_direction = 'fwd'):
        if sweep_direction == 'fwd':
            current = self.signals['Current (A)'][:, :, sweep_idx]
        elif sweep_direction == 'bwd':
            current = self.signals['Current [bwd] (A)'][:, :, sweep_idx]
        elif sweep_direction == 'AVG':
            current = np.nanmean ( [self.signals['Current (A)'], self.signals['Current [bwd] (A)']], axis = 2 ) [:, :, sweep_idx]
        return current
    
    def get_apparent_barrier_height (self, line, pixel, sweep_direction='fwd', fitting_current_range=(1e-12, 10e-12)):
        # fitting_current_range: current range in A unit.
        import numpy as np
        from scipy.optimize import curve_fit
        def linear(x, barr, b):
            return -2*( np.sqrt(2*0.51099895e+6*barr)/(6.582119569e-16*2.99792458e+8) )*x + b
        
        z = self.signals['sweep_signal']
        if sweep_direction == 'fwd':
            I = np.abs(self.signals['Current (A)'][line, pixel])
        elif sweep_direction == 'bwd':
            I = np.abs(self.signals['Current [bwd] (A)'][line, pixel])
        elif sweep_direction == 'AVG':
            I = np.abs(np.nanmean ( [self.signals['Current (A)'][line, pixel], \
                              self.signals['Current [bwd] (A)'][line, pixel]], \
                           axis = 0))
        
        ############################## Set fitting range ##############################
        idx = np.where( (fitting_current_range[0] <= I) & (I <= fitting_current_range[1]) ) # Filter with I
        ############################## Set fitting range ##############################
        popt, pcov = curve_fit (linear, z[idx], np.log(I[idx]), p0 = [1.2, 1.2])
        apparent_barrier_height = popt[0]
        err = np.sqrt(np.diag(pcov))[0]
        # err = np.sqrt(np.diag(pcov))[0]
        # slope = -2*np.sqrt(2*0.51099895e+6*apparent_barrier_height)/(6.582119569e-16*2.99792458e+8)
        # return apparent_barrier_height, err, slope
       
        return apparent_barrier_height, err
    
    def get_apparent_barrier_height_map (self, sweep_direction='fwd', fitting_current_range=(1e-12, 10e-12)):
        import numpy as np
        lines, pixels = self.header['dim_px'][1], self.header['dim_px'][0]
        arr = np.zeros((lines, pixels))
        err = np.zeros((lines, pixels))
        for i in range (lines):
            for j in range (pixels):
                try:
                    arr[i, j] = self.get_apparent_barrier_height (i, j, sweep_direction, fitting_current_range)[0]
                    err[i, j] = self.get_apparent_barrier_height (i, j, sweep_direction, fitting_current_range)[1]
                except:
                    print (f'Estimation error at: {i, j}. Investigate z-spectrum at {i, j} for detailed info.')
                    arr[i, j] = np.nan
                    err[i, j] = np.nan
        return arr, err
    
    def get_sweepsignal (self, sweep_idx):
        return self.signals['sweep_signal'][sweep_idx]

    
class PtSpec: # any spectrum (dIdV, Z, I, ...) vs sweep_signal at any point.
        
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals

    def get_didv_raw (self, line, pixel, channel = 'none', offset = 'none'):
        import numpy as np
        if channel == 'none':
            if 'LI Demod 2 X (A)' in self.signals.keys():
                channel = 'LI Demod 2 X (A)'
            elif 'LI Demod 1 X (A)' in self.signals.keys():
                channel = 'LI Demod 1 X (A)'
        else:
            channel = channel
        if isinstance(offset, np.ndarray):
            didv = self.signals[channel][line, pixel] - offset
        else:
            didv = self.signals[channel][line, pixel]
        return self.signals['sweep_signal'], didv

    
    def get_dzdv_numerical (self, line, pixel):
        import numpy as np
        z = self.signals['Z (m)'][line, pixel]
        dzdv_numerical = np.gradient(z, edge_order=2)
        return self.signals['sweep_signal'], dzdv_numerical

    def get_apparent_barrier_height (self, line, pixel, sweep_direction='fwd', fitting_current_range=(1e-12, 10e-12)):
        # fitting_current_range: current range in A unit.
        import numpy as np
        from scipy.optimize import curve_fit
        def linear(x, barr, b):
            return -2*( np.sqrt(2*0.51099895e+6*barr)/(6.582119569e-16*2.99792458e+8) )*x + b
        
        z = self.signals['sweep_signal']
        if sweep_direction == 'fwd':
            I = np.abs(self.signals['Current (A)'][line, pixel])
        elif sweep_direction == 'bwd':
            I = np.abs(self.signals['Current [bwd] (A)'][line, pixel])
        elif sweep_direction == 'AVG':
            I = np.abs(np.nanmean ( [self.signals['Current (A)'][line, pixel], \
                              self.signals['Current [bwd] (A)'][line, pixel]], \
                           axis = 0))
        
        ############################## Set fitting range ##############################
        idx = np.where( (fitting_current_range[0] <= I) & (I <= fitting_current_range[1]) ) # Filter with I
        ############################## Set fitting range ##############################
        popt, pcov = curve_fit (linear, z[idx], np.log(I[idx]), p0 = [1.2, 1.2])
        apparent_barrier_height = popt[0]
        # err = np.sqrt(np.diag(pcov))[0]
        # slope = -2*np.sqrt(2*0.51099895e+6*apparent_barrier_height)/(6.582119569e-16*2.99792458e+8)
        # return apparent_barrier_height, err, slope
       
        return apparent_barrier_height
    
    def get_didv_scaled (self, line, pixel, channel = 'LI Demod 2 X (A)', offset = 'none'):
        '''
        Returns
        -------
        tuple
            (Bias (V), dIdV (S))
        '''
        import numpy as np
        # return self.signals['sweep_signal'], np.median(self.get_didv_numerical(line, pixel)[1]/self.signals[channel][line, pixel])*self.signals[channel][line, pixel]
        return self.signals['sweep_signal'], \
        np.median(self.get_didv_numerical(line, pixel)[1]/self.get_didv_raw(line, pixel, channel, offset)[1])\
        *self.get_didv_raw(line, pixel, channel, offset)[1]
    
    def get_didv_normalized (self, line, pixel, channel='LI Demod 2 X (A)', factor=0.2, offset='none', delete_zero_bias=False):
        '''
        Returns
        -------
        tuple
            (Bias (V), normalized dIdV)
        '''        
        import numpy as np
        from scipy.optimize import curve_fit
        try:
            from scipy.integrate import cumtrapz
        except:
            from scipy.integrate import cumulative_trapezoid
        
        # dIdV, V = self.get_didv_scaled(line, pixel, channel)[1], self.signals['sweep_signal']
        V, dIdV = self.get_didv_scaled(line, pixel, channel, offset = 'none')
        try:
            I_cal = cumtrapz(dIdV, V, initial = 0)
        except:
            I_cal = cumulative_trapezoid(dIdV, V, initial = 0)
        zero = np.argwhere ( abs(V) == np.min(abs(V)) )[0, 0] # The index where V = 0 or nearest to 0.
        popt, pcov = curve_fit (lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
        I_cal -= popt[1]

        # get total conductance I/V
        with np.errstate(divide='ignore'): # Ignore the warning of 'division by zero'.
            IV_cal = I_cal/V

        # I_cal/V = 0/0으로 계산되는 경우
        # nan으로 처리됨. 이 값 제외를 위해 nanmedian 사용.
        delta = factor*np.nanmedian(IV_cal)
        Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
        if delete_zero_bias == False:
            return V, Normalized_dIdV
        else:
            return np.delete(V, zero), np.delete(Normalized_dIdV, zero)

        

    
    def get_didv_numerical (self, line, pixel):
        '''
        Returns
        -------
        tuple
            (Bias (V), numerical dIdV (S))
        '''        
        import numpy as np
        step = self.signals['sweep_signal'][1] - self.signals['sweep_signal'][0]
        didv = np.gradient(self.signals['Current (A)'][line, pixel], step, edge_order=2) # I-V curve를 직접 미분.
        return self.signals['sweep_signal'], didv
    
    def get_iv_raw (self, line, pixel):
        '''
        Returns
        -------
        tuple
            (Bias (V), Current (A))
        '''        
        return self.signals['sweep_signal'], self.signals['Current (A)']


class LineSpec: # any spectrum (dIdV, Z, I, ...) vs sweep_signal at any point.
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    # def get (self, line, sts='scaled', channel='LI Demod 2 X (A)', factor=0.2, offset='none', delete_zero_bias=False):
    def get (self, line, processing='scaled', **kwargs):
        import numpy as np
        if processing == 'scaled':
            spec = self.get_didv_scaled
        elif processing == 'raw':
            spec = self.get_didv_raw
        elif processing == 'numerical':
            spec = self.get_didv_numerical
        elif processing == 'normalized':
            spec = self.get_didv_normalized
        linespec = np.array([ spec(line, pixel, **kwargs)[1] for pixel \
                     in range (self.header['dim_px'][0]) ]).T
        return linespec

    def get_didv_raw (self, line, pixel, channel = 'none', offset = 'none'):
        if channel == 'none':
            if 'LI Demod 2 X (A)' in self.signals.keys():
                channel = 'LI Demod 2 X (A)'
            elif 'LI Demod 1 X (A)' in self.signals.keys():
                channel = 'LI Demod 1 X (A)'
        else:
            channel = channel
        if offset != 'none':
            didv = self.signals[channel][line, pixel] - offset
        else:
            didv = self.signals[channel][line, pixel]
        return self.signals['sweep_signal'], didv

    def get_didv_scaled (self, line, pixel, channel = 'LI Demod 2 X (A)', offset = 'none'):
        '''
        Returns
        -------
        tuple
            (Bias (V), dIdV (S))
        '''
        import numpy as np
        # return self.signals['sweep_signal'], np.median(self.get_didv_numerical(line, pixel)[1]/self.signals[channel][line, pixel])*self.signals[channel][line, pixel]
        return self.signals['sweep_signal'], \
        np.median(self.get_didv_numerical(line, pixel)[1]/self.get_didv_raw(line, pixel, channel, offset)[1])\
        *self.get_didv_raw(line, pixel, channel, offset)[1]
    
    def get_didv_normalized (self, line, pixel, channel='LI Demod 2 X (A)', factor=0.2, offset='none', delete_zero_bias=False):
        '''
        Returns
        -------
        tuple
            (Bias (V), normalized dIdV)
        '''        
        import numpy as np
        from scipy.optimize import curve_fit
        try:
            from scipy.integrate import cumtrapz
        except:
            from scipy.integrate import cumulative_trapezoid
        
        # dIdV, V = self.get_didv_scaled(line, pixel, channel)[1], self.signals['sweep_signal']
        V, dIdV = self.get_didv_scaled(line, pixel, channel, offset = 'none')
        try:
            I_cal = cumtrapz(dIdV, V, initial = 0)
        except:
            I_cal = cumulative_trapezoid(dIdV, V, initial = 0)
        zero = np.argwhere ( abs(V) == np.min(abs(V)) )[0, 0] # The index where V = 0 or nearest to 0.
        popt, pcov = curve_fit (lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
        I_cal -= popt[1]

        # get total conductance I/V
        with np.errstate(divide='ignore'): # Ignore the warning of 'division by zero'.
            IV_cal = I_cal/V

        # I_cal/V = 0/0으로 계산되는 경우
        # nan으로 처리됨. 이 값 제외를 위해 nanmedian 사용.
        delta = factor*np.nanmedian(IV_cal)
        Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
        if delete_zero_bias == False:
            return V, Normalized_dIdV
        else:
            return np.delete(V, zero), np.delete(Normalized_dIdV, zero)

        

    
    def get_didv_numerical (self, line, pixel):
        '''
        Returns
        -------
        tuple
            (Bias (V), numerical dIdV (S))
        '''        
        import numpy as np
        step = self.signals['sweep_signal'][1] - self.signals['sweep_signal'][0]
        didv = np.gradient(self.signals['Current (A)'][line, pixel], step, edge_order=2) # I-V curve를 직접 미분.
        return self.signals['sweep_signal'], didv
































        
        
            