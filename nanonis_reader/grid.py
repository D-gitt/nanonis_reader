# nanonispy dependency removed — parsing handled by base.py via _vendor
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
try:
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz


# class load:
#     def __init__(self, filepath):
#         self.fname = os.path.basename(filepath)
#         self.header = nap.read.Grid(filepath).header
#         self.signals = nap.read.Grid(filepath).signals


class topography:
        
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
        tmp = self.signals['topo']
        z = np.where(tmp == 0, np.nan, tmp)
        return z
    
    def subtract_average (self):
        from . import image_processing as ip
        return ip.subtract_average(self.raw())

    def subtract_linear_fit(self):        
        from . import image_processing as ip
        return ip.subtract_linear_fit(self.raw())

    # def subtract_parabolic_fit (self):
    #     def f_parab(x, a, b, c): return a*(x**2) + b*x + c
    #     xrange = round(self.header['size_xy'][0] * 1e9)*1e-9
    #     print (xrange)
    #     z = self.raw()
    #     z_subpf = np.zeros(np.shape(z))
    #     lines, pixels = np.shape(z)
    #     for i in range(lines):
    #         if np.shape(np.where(np.isnan(z))[0])[0] != 0: # image에 nan값이 포함되어 있을 경우 (== scan을 도중에 멈추었을 경우)
    #             if i < np.min(np.where(np.isnan(z))[0]):
    #                 x = np.linspace(0, xrange, pixels)
    #                 popt, pcov = curve_fit(f_parab, x, z[i])
    #                 z_subpf[i] = z[i] - f_parab(x, *popt)
    #             else:
    #                 z_subpf[i] = np.nan
    #         else:
    #             x = np.linspace(0, xrange, pixels)
    #             popt, pcov = curve_fit(f_parab, x, z[i]) # x - ith line: linear fitting
    #             z_subpf[i] = z[i] - f_parab(x, *popt)
    #     return z_subpf

    def subtract_parabolic_fit(self):        
        from . import image_processing as ip
        return ip.subtract_parabolic_fit(self.raw())
    
    # def differentiate (self):
    #     xrange, pixels = round(self.header['size_xy'][0] * 1e9)*1e-9, int(self.header['dim_px'][0])
    #     dx = xrange / pixels
    #     z = self.raw()
    #     z_deriv = np.zeros(np.shape(z))
    #     lines = np.shape(z)[0]
    #     for i in range(lines):
    #         z_deriv[i] = np.gradient(z[i], dx, edge_order = 2) # dI/dV curve를 직접 미분. --> d^2I/dV^2
    #     return z_deriv

    def differentiate(self):
        from . import image_processing as ip
        xrange = round(self.header['size_xy'][0] * 1e9) * 1e-9
        pixels = int(self.header['dim_px'][0])
        dx = xrange / pixels
        return ip.differentiate(self.raw(), dx)


class point_didv:
    def __init__(self, instance, sweep_direction='fwd'):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        self.sweep_dir = sweep_direction

    def get_channel_name(self, base_channel, include_avg=False, bwd=None, sweep_direction=None):
        """
        Parameters:
        -----------
        base_channel : str
            Base channel name (e.g., 'LI Demod 1 X (A)' or 'Current (A)')
        include_avg : bool
            Whether to include the [AVG] tag
        bwd : bool or None
            If True, forces [bwd] tag. If None, uses sweep_direction/self.sweep_dir
        sweep_direction : str or None
            'fwd' or 'bwd'. If None, uses the instance default (self.sweep_dir).
        """
        sd = sweep_direction if sweep_direction is not None else self.sweep_dir
        channel_base = base_channel.replace(' (A)', '')
        
        tags = []
        if include_avg:
            tags.append('[AVG]')
        if bwd is True or (bwd is None and sd == 'bwd'):
            tags.append('[bwd]')
            
        if tags:
            channel_name = f"{channel_base} {' '.join(tags)} (A)"
        else:
            channel_name = f"{channel_base} (A)"
            
        return channel_name

    def has_averaged_data(self):
        """
        Checks if the dataset contains averaged signals.
        """
        return 'Current [AVG] (A)' in self.signals.keys()

    def get_didv_raw(self, line, pixel, channel='LI Demod 1 X (A)', offset='none', sweep_direction=None):
        channel_name = self.get_channel_name(channel, include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        if isinstance(offset, np.ndarray):
            didv = self.signals[channel_name][line, pixel] - offset
        else:
            didv = self.signals[channel_name][line, pixel]
            
        return self.signals['sweep_signal'], didv

    def get_didv_scaled(self, line, pixel, channel='LI Demod 1 X', offset='none', sweep_direction=None):
        """
        Returns
        -------
        tuple
            (Bias (V), dIdV (S))
        """
        return self.signals['sweep_signal'], \
                np.median(self.get_didv_numerical(line, pixel, sweep_direction=sweep_direction)[1]/self.get_didv_raw(line, pixel, channel, offset, sweep_direction=sweep_direction)[1])\
                *self.get_didv_raw(line, pixel, channel, offset, sweep_direction=sweep_direction)[1]
    
    # def get_didv_normalized(self, line, pixel, channel='LI Demod 1 X', factor=0.2, offset='none', delete_zero_bias=True):
    #     """
    #     Returns
    #     -------
    #     tuple
    #         (Bias (V), normalized dIdV)
    #     """
    #     V, dIdV = self.get_didv_scaled(line, pixel, channel, offset='none')
    #     I_cal = cumtrapz(dIdV, V, initial=0)
    # 
    #     zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
    #     popt, pcov = curve_fit(lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
    #     I_cal -= popt[1]
    # 
    #     with np.errstate(divide='ignore'):
    #         IV_cal = I_cal/V
    # 
    #     delta = factor*np.nanmedian(IV_cal)
    #     Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
    #     
    #     if delete_zero_bias:
    #         return np.delete(V, zero), np.delete(Normalized_dIdV, zero)
    #     return V, Normalized_dIdV
    # 
    # def get_didv_normalized_rev(self, line, pixel, channel='LI Demod 1 X', factor=0.2, offset='none', delete_zero_bias=True):
    #     """
    #     Returns
    #     -------
    #     tuple
    #         (Bias (V), normalized dIdV)
    #     """
    #     V, dIdV = self.get_didv_scaled(line, pixel, channel, offset='none')
    #     I_cal = cumtrapz(dIdV, V, initial=0)
    # 
    #     zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
    # 
    #     with np.errstate(divide='ignore'): # Ignore the warning of zero division.
    #         if V[zero] == 0: # The case V has 0 as an element.
    #             I_cal -= I_cal[zero]  # Offset for I(V=0) = 0
    #             IV_cal = I_cal/V
    #             
    #             # linear interpolation for I/V at 0 V: y = mx + b
    #             m = (IV_cal[zero+1] - IV_cal[zero-1]) / (V[zero+1] - V[zero-1])
    #             b = IV_cal[zero+1] - m * V[zero+1]
    #             IV_cal[zero] = b
    #         else:
    #             popt, _ = curve_fit(lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
    #             I_cal -= popt[1]
    #             IV_cal = I_cal/V
    #     
    #     delta = factor*np.nanmedian(IV_cal)
    #     Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
    #     
    #     if delete_zero_bias:
    #         return np.delete(V, zero), np.delete(Normalized_dIdV, zero)
    #     else:
    #         return V, Normalized_dIdV

    def get_didv_normalized(self, line, pixel, channel='LI Demod 1 X', factor=0.2, offset='none', delete_zero_bias=True, sweep_direction=None):
        """
        Returns
        -------
        tuple
            (Bias (V), normalized dIdV)
        """
        V, dIdV = self.get_didv_scaled(line, pixel, channel, offset=offset, sweep_direction=sweep_direction)
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

    def get_didv_numerical(self, line, pixel, sweep_direction=None):
        """
        Returns
        -------
        tuple
            (Bias (V), numerical dIdV (S))
        """
        step = self.signals['sweep_signal'][1] - self.signals['sweep_signal'][0]
        current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        didv = np.gradient(self.signals[current_channel][line, pixel], step, edge_order=2)
        return self.signals['sweep_signal'], didv
    
    def get_iv_raw(self, line, pixel, sweep_direction=None):
        """
        Returns
        -------
        tuple
            (Bias (V), Current (A))
        """        
        current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        return self.signals['sweep_signal'], self.signals[current_channel][line, pixel]

class didvmap(point_didv):  # dIdV
    def __init__(self, instance, sweep_direction='fwd'):
        super().__init__(instance, sweep_direction)

    def get_didvmap(self, sweep_idx, processing='raw', channel='LI Demod 1 X (A)', sweep_direction=None, **kwargs):
        """
        Get dI/dV map at a specific sweep index with various processing options.

        Parameters:
        -----------
        sweep_idx : int
            Index of the sweep signal
        processing : str
            Processing method: 'raw', 'scaled', 'normalized'
        channel : str
            Channel name (e.g., 'LI Demod 1 X (A)')
        sweep_direction : str or None
            'fwd' or 'bwd'. If None, uses the instance default.
        **kwargs : dict
            Additional parameters for processing (e.g., factor, offset, delete_zero_bias)

        Returns:
        --------
        np.ndarray
            2D array of dI/dV values at the specified sweep index
        """
        if processing == 'raw':
            return self.raw(sweep_idx, channel, sweep_direction=sweep_direction)
        elif processing == 'scaled':
            return self.scaled(sweep_idx, channel, sweep_direction=sweep_direction, **kwargs)
        elif processing == 'normalized':
            return self.normalized(sweep_idx, channel, sweep_direction=sweep_direction, **kwargs)
        else:
            raise ValueError(f"Unknown processing method: {processing}")

    def raw(self, sweep_idx, channel='LI Demod 1 X (A)', sweep_direction=None):
        """Get raw dI/dV map at a specific sweep index."""
        channel_name = self.get_channel_name(channel, include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        didv = self.signals[channel_name][:, :, sweep_idx]
        return didv

    def scaled(self, sweep_idx, channel='LI Demod 1 X', offset='none', sweep_direction=None):
        """Get scaled dI/dV map at a specific sweep index."""
        channel_name = self.get_channel_name(channel, include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        didv_raw = np.copy(self.signals[channel_name])
        if offset != 'none':
            didv_raw = didv_raw - offset
            
        step = self.signals['sweep_signal'][1] - self.signals['sweep_signal'][0]
        current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        
        # Calculate numerical derivatives and scale factor across the entire 3D grid
        didv_numerical = np.gradient(self.signals[current_channel], step, axis=2, edge_order=2)
        scale_factors = np.nanmedian(didv_numerical / didv_raw, axis=2)
        
        didv_map = scale_factors * didv_raw[:, :, sweep_idx]
        return didv_map

    def normalized(self, sweep_idx, channel='LI Demod 1 X', factor=0.2, offset='none', delete_zero_bias=True, sweep_direction=None):
        """Get normalized dI/dV map at a specific sweep index (Vectorized 3D Volume)."""
        channel_name = self.get_channel_name(channel, include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        didv_raw = np.copy(self.signals[channel_name])
        if offset != 'none':
            didv_raw = didv_raw - offset
        
        V = self.signals['sweep_signal']
        step = V[1] - V[0]
        current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        
        didv_numerical = np.gradient(self.signals[current_channel], step, axis=2, edge_order=2)
        
        # dI/dV Scaled 3D matrix
        scale_factors = np.nanmedian(didv_numerical / didv_raw, axis=2)[..., np.newaxis]
        dIdV = scale_factors * didv_raw
        
        I_cal = cumtrapz(dIdV, V, initial=0, axis=2)
            
        zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
        
        # curve_fit linear offset correction on the V=0 near points (3 points) vectorized
        x = V[zero-1:zero+2]
        y = I_cal[:, :, zero-1:zero+2]
        
        lines, pixels, sweeps = I_cal.shape
        y_reshaped = y.reshape(lines * pixels, 3).T
        
        # NaN 방어 (스캔 도중 중지되어 일부 픽셀에 NaN이 있는 경우 전체 polyfit 연산이 터지는 것 방지)
        valid_cols = ~np.isnan(y_reshaped).any(axis=0)
        b_offset = np.full(lines * pixels, np.nan)
        
        if np.any(valid_cols):
            # Fit degree 1: returns shape (2, valid_pixels)
            popt = np.polyfit(x, y_reshaped[:, valid_cols], 1)
            b_offset[valid_cols] = popt[1]
        
        # popt[1] is the intercept b
        b_offset = b_offset.reshape(lines, pixels, 1)
        I_cal = I_cal - b_offset
        
        with np.errstate(divide='ignore'):
            IV_cal = I_cal / V
            
        delta = factor * np.nanmedian(IV_cal, axis=2)[..., np.newaxis]
        Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
        
        if delete_zero_bias:
            Normalized_dIdV = np.delete(Normalized_dIdV, zero, axis=2)
            
        return Normalized_dIdV[:, :, sweep_idx]

    def get_currentmap(self, sweep_idx, sweep_direction=None):
        """Get current map at a specific sweep index."""
        current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        return self.signals[current_channel][:, :, sweep_idx]

    # def get_currentmap(self, sweep_idx):
    #     if self.sweep_dir == 'AVG':
    #         fwd_current = self.signals[self.get_channel_name('Current', include_avg=False)][:, :, sweep_idx]
    #         bwd_current = self.signals[self.get_channel_name('Current', include_avg=False, bwd=True)][:, :, sweep_idx]
    #         current = np.nanmean([fwd_current, bwd_current], axis=0)
    #     else:
    #         current = self.signals[self.get_channel_name('Current',
    #                                                     include_avg=self.has_averaged_data())][:, :, sweep_idx]
    #     return current
    
    def get_sweepsignal (self, sweep_idx):
        return self.signals['sweep_signal'][sweep_idx]

class point_iz:
    def __init__(self, instance, sweep_direction='fwd'):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        self.sweep_dir = sweep_direction

    def get_channel_name(self, base_channel, include_avg=False, bwd=None, sweep_direction=None):
        """
        Parameters:
        -----------
        base_channel : str
            Base channel name (e.g., 'LI Demod 1 X (A)' or 'Current (A)')
        include_avg : bool
            Whether to include the [AVG] tag
        bwd : bool or None
            If True, forces [bwd] tag. If None, uses sweep_direction/self.sweep_dir
        sweep_direction : str or None
            'fwd' or 'bwd'. If None, uses the instance default (self.sweep_dir).
        """
        sd = sweep_direction if sweep_direction is not None else self.sweep_dir
        channel_base = base_channel.replace(' (A)', '')
        
        tags = []
        if include_avg:
            tags.append('[AVG]')
        if bwd is True or (bwd is None and sd == 'bwd'):
            tags.append('[bwd]')
            
        if tags:
            channel_name = f"{channel_base} {' '.join(tags)} (A)"
        else:
            channel_name = f"{channel_base} (A)"
            
        return channel_name
    
    def has_averaged_data(self):
        """
        Checks if the dataset contains averaged signals.
        """
        return 'Current [AVG] (A)' in self.signals.keys()

    def get_iz_raw(self, line, pixel, sweep_direction=None):
        """
        Returns
        -------
        tuple
            (Z (m), Current (A))
        """        
        current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction=sweep_direction)
        return self.signals['sweep_signal'], self.signals[current_channel][line, pixel]

    def get_apparent_barrier_height(self, line, pixel, fitting_current_range=(1e-12, 10e-12), sweep_direction=None):
        sd = sweep_direction if sweep_direction is not None else self.sweep_dir
        def linear(x, barr, b):
            return -2*(np.sqrt(2*0.51099895e+6*barr)/(6.582119569e-16*2.99792458e+8))*x + b
        
        z = self.signals['sweep_signal']
        if sd == 'AVG':
            fwd_current = self.signals[self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction='fwd')][line, pixel]
            bwd_current = self.signals[self.get_channel_name('Current', include_avg=self.has_averaged_data(), bwd=True, sweep_direction='fwd')][line, pixel]
            I = np.abs(np.nanmean([fwd_current, bwd_current], axis=0))
        else:
            channel_name = self.get_channel_name('Current', include_avg=self.has_averaged_data(), sweep_direction=sd)
            I = np.abs(self.signals[channel_name][line, pixel])

        idx = np.where((fitting_current_range[0] <= I) & (I <= fitting_current_range[1]))
        idxz0 = np.argwhere(z == np.min(np.abs(z)))[0, 0]
        dI = np.log(fitting_current_range[1]) - np.log(fitting_current_range[0])
        dz = z[idx][-1] - z[idx][0]
        slope = dI/dz
        barr = ((slope * (6.582119569e-16*2.99792458e+8)/(-2))**2)/(2*0.51099895e+6)

        popt, pcov = curve_fit(linear, z[idx], np.log(I[idx]), p0=[barr, np.log(I[idxz0])])
        # popt, pcov = curve_fit(linear, z[idx], np.log(I[idx]), p0=[5.5, 1.2])
        return popt[0]  # apparent_barrier_height

class izmap(point_iz):  # I-z spec, apparent barrier map
    def __init__(self, instance, sweep_direction='fwd'):
        super().__init__(instance, sweep_direction)
    
    def get_izmap(self, sweep_idx, sweep_direction=None):
        sd = sweep_direction if sweep_direction is not None else self.sweep_dir
        if sd == 'AVG':
            fwd_current = self.signals[self.get_channel_name('Current', include_avg=False, sweep_direction='fwd')][:, :, sweep_idx]
            bwd_current = self.signals[self.get_channel_name('Current', include_avg=False, bwd=True, sweep_direction='fwd')][:, :, sweep_idx]
            current = np.nanmean([fwd_current, bwd_current], axis=0)
        else:
            current = self.signals[self.get_channel_name('Current', 
                                                        include_avg=self.has_averaged_data(), sweep_direction=sd)][:, :, sweep_idx]
        return current

    # def get_currentmap(self, sweep_idx):
    #     if self.sweep_dir == 'AVG':
    #         fwd_current = self.signals[self.get_channel_name('Current', include_avg=False)][:, :, sweep_idx]
    #         bwd_current = self.signals[self.get_channel_name('Current', include_avg=False, bwd=True)][:, :, sweep_idx]
    #         current = np.nanmean([fwd_current, bwd_current], axis=0)
    #     else:
    #         current = self.signals[self.get_channel_name('Current', 
    #                                                     include_avg=self.has_averaged_data())][:, :, sweep_idx]
    #     return current
    
    def get_apparent_barrier_height_map(self, fitting_current_range=(1e-12, 10e-12), sweep_direction=None):
        lines, pixels = self.header['dim_px'][1], self.header['dim_px'][0]
        arr = np.zeros((lines, pixels))
        # err = np.zeros((lines, pixels))
        for i in range(lines):
            for j in range(pixels):
                try:
                    # arr[i, j], err[i, j] = self.get_apparent_barrier_height(i, j, fitting_current_range)
                    arr[i, j] = self.get_apparent_barrier_height(i, j, fitting_current_range, sweep_direction=sweep_direction)
                except Exception as e:
                    print(f'Estimation error at: {i, j}. {str(e)}')
                    arr[i, j] = np.nan
                    # err[i, j] = np.nan
        # return arr, err
        return arr
    
    def get_sweepsignal (self, sweep_idx):
        return self.signals['sweep_signal'][sweep_idx]

# map class until ver. 0.1.7.
# class map:  # dIdV, I-z spec, apparent barrier map
#     def __init__(self, instance, sweep_direction='fwd'):
#         self.fname = instance.fname
#         self.header = instance.header
#         self.signals = instance.signals
#         self.sweep_dir = sweep_direction
        
#     def get_channel_name(self, base_channel, include_avg=False, bwd=None):
#         """
#         Parameters:
#         -----------
#         base_channel : str
#             Base channel name (e.g., 'LI Demod 1 X (A)' or 'Current (A)')
#         include_avg : bool
#             Whether to include the [AVG] tag
#         bwd : bool or None
#             If True, forces [bwd] tag. If None, uses self.sweep_dir
#         """
#         channel_base = base_channel.replace(' (A)', '')
        
#         tags = []
#         if include_avg:
#             tags.append('[AVG]')
#         if bwd or (bwd is None and self.sweep_dir == 'bwd'):
#             tags.append('[bwd]')
            
#         if tags:
#             channel_name = f"{channel_base} {' '.join(tags)} (A)"
#         else:
#             channel_name = f"{channel_base} (A)"
            
#         return channel_name

#     def has_averaged_data(self):
#         """
#         Checks if the dataset contains averaged signals.
#         """
#         return 'Current [AVG] (A)' in self.signals.keys()
        
#     def get_didvmap(self, sweep_idx, channel='LI Demod 1 X (A)'):
#         channel_name = self.get_channel_name(channel, include_avg=self.has_averaged_data())
#         didv = self.signals[channel_name][:, :, sweep_idx]
#         return didv
    
#     def get_currentmap(self, sweep_idx):
#         if self.sweep_dir == 'AVG':
#             fwd_current = self.signals[self.get_channel_name('Current', include_avg=False)][:, :, sweep_idx]
#             bwd_current = self.signals[self.get_channel_name('Current', include_avg=False, bwd=True)][:, :, sweep_idx]
#             current = np.nanmean([fwd_current, bwd_current], axis=0)
#         else:
#             current = self.signals[self.get_channel_name('Current', 
#                                                         include_avg=self.has_averaged_data())][:, :, sweep_idx]
#         return current

#     def get_apparent_barrier_height(self, line, pixel, fitting_current_range=(1e-12, 10e-12)):
#         def linear(x, barr, b):
#             return -2*(np.sqrt(2*0.51099895e+6*barr)/(6.582119569e-16*2.99792458e+8))*x + b
        
#         z = self.signals['sweep_signal']
#         if self.sweep_dir == 'AVG':
#             fwd_current = self.signals[self.get_channel_name('Current', include_avg=False)][line, pixel]
#             bwd_current = self.signals[self.get_channel_name('Current', include_avg=False, bwd=True)][line, pixel]
#             I = np.abs(np.nanmean([fwd_current, bwd_current], axis=0))
#         else:
#             channel_name = self.get_channel_name('Current', include_avg=self.has_averaged_data())
#             I = np.abs(self.signals[channel_name][line, pixel])

#         idx = np.where((fitting_current_range[0] <= I) & (I <= fitting_current_range[1]))
#         popt, pcov = curve_fit(linear, z[idx], np.log(I[idx]), p0=[5.5, 1.2])
#         apparent_barrier_height = popt[0]
#         err = np.sqrt(np.diag(pcov))[0]

#         return apparent_barrier_height, err
    
#     def get_apparent_barrier_height_map(self, fitting_current_range=(1e-12, 10e-12)):
#         lines, pixels = self.header['dim_px'][1], self.header['dim_px'][0]
#         arr = np.zeros((lines, pixels))
#         err = np.zeros((lines, pixels))
#         for i in range(lines):
#             for j in range(pixels):
#                 try:
#                     arr[i, j], err[i, j] = self.get_apparent_barrier_height(i, j, fitting_current_range)
#                 except Exception as e:
#                     print(f'Estimation error at: {i, j}. {str(e)}')
#                     arr[i, j] = np.nan
#                     err[i, j] = np.nan
#         return arr, err
    
#     def get_sweepsignal (self, sweep_idx):
#         return self.signals['sweep_signal'][sweep_idx]

# point_spectrum class until ver. 0.1.7.
# class point_spectrum:  # any spectrum (dIdV, Z, I, ...) vs sweep_signal at any point.
        
#     def __init__(self, instance, sweep_direction='fwd'):
#         self.fname = instance.fname
#         self.header = instance.header
#         self.signals = instance.signals
#         self.sweep_dir = sweep_direction

#     def get_channel_name(self, base_channel, include_avg=False, bwd=None):
#         """
#         Parameters:
#         -----------
#         base_channel : str
#             Base channel name (e.g., 'LI Demod 1 X (A)' or 'Current (A)')
#         include_avg : bool
#             Whether to include the [AVG] tag
#         bwd : bool or None
#             If True, forces [bwd] tag. If None, uses self.sweep_dir
#         """
#         channel_base = base_channel.replace(' (A)', '')
        
#         tags = []
#         if include_avg:
#             tags.append('[AVG]')
#         if bwd or (bwd is None and self.sweep_dir == 'bwd'):
#             tags.append('[bwd]')
            
#         if tags:
#             channel_name = f"{channel_base} {' '.join(tags)} (A)"
#         else:
#             channel_name = f"{channel_base} (A)"
            
#         return channel_name

#     def has_averaged_data(self):
#         """
#         Checks if the dataset contains averaged signals.
#         """
#         return 'Current [AVG] (A)' in self.signals.keys()

#     def get_didv_raw(self, line, pixel, channel='LI Demod 1 X (A)', offset='none'):
#         channel_name = self.get_channel_name(channel, include_avg=self.has_averaged_data())
#         if isinstance(offset, np.ndarray):
#             didv = self.signals[channel_name][line, pixel] - offset
#         else:
#             didv = self.signals[channel_name][line, pixel]
            
#         return self.signals['sweep_signal'], didv

#     def get_dzdv_numerical(self, line, pixel):
#         z = self.signals['Z (m)'][line, pixel]
#         dzdv_numerical = np.gradient(z, edge_order=2)
#         return self.signals['sweep_signal'], dzdv_numerical

#     def get_apparent_barrier_height(self, line, pixel, fitting_current_range=(1e-12, 10e-12)):
#         def linear(x, barr, b):
#             return -2*(np.sqrt(2*0.51099895e+6*barr)/(6.582119569e-16*2.99792458e+8))*x + b
        
#         z = self.signals['sweep_signal']
#         if self.sweep_dir == 'AVG':
#             fwd_current = self.signals[self.get_channel_name('Current', include_avg=self.has_averaged_data())][line, pixel]
#             bwd_current = self.signals[self.get_channel_name('Current', include_avg=self.has_averaged_data(), bwd=True)][line, pixel]
#             I = np.abs(np.nanmean([fwd_current, bwd_current], axis=0))
#         else:
#             channel_name = self.get_channel_name('Current', include_avg=self.has_averaged_data())
#             I = np.abs(self.signals[channel_name][line, pixel])

#         idx = np.where((fitting_current_range[0] <= I) & (I <= fitting_current_range[1]))
#         popt, pcov = curve_fit(linear, z[idx], np.log(I[idx]), p0=[5.5, 1.2])
#         return popt[0]  # apparent_barrier_height

#     def get_didv_scaled(self, line, pixel, channel='LI Demod 1 X', offset='none'):
#         """
#         Returns
#         -------
#         tuple
#             (Bias (V), dIdV (S))
#         """
#         return self.signals['sweep_signal'], \
#                 np.median(self.get_didv_numerical(line, pixel)[1]/self.get_didv_raw(line, pixel, channel, offset)[1])\
#                 *self.get_didv_raw(line, pixel, channel, offset)[1]

#     def get_didv_normalized(self, line, pixel, channel='LI Demod 1 X', factor=0.2, offset='none', delete_zero_bias=True):
#         """
#         Returns
#         -------
#         tuple
#             (Bias (V), normalized dIdV)
#         """
#         V, dIdV = self.get_didv_scaled(line, pixel, channel, offset='none')
#         I_cal = cumtrapz(dIdV, V, initial=0)
#
#         zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
#         popt, pcov = curve_fit(lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
#         I_cal -= popt[1]

#         with np.errstate(divide='ignore'):
#             IV_cal = I_cal/V

#         delta = factor*np.nanmedian(IV_cal)
#         Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
        
#         if delete_zero_bias:
#             return np.delete(V, zero), np.delete(Normalized_dIdV, zero)
#         return V, Normalized_dIdV

#     def get_didv_numerical(self, line, pixel):
#         """
#         Returns
#         -------
#         tuple
#             (Bias (V), numerical dIdV (S))
#         """
#         step = self.signals['sweep_signal'][1] - self.signals['sweep_signal'][0]
#         current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data())
#         didv = np.gradient(self.signals[current_channel][line, pixel], step, edge_order=2)
#         return self.signals['sweep_signal'], didv
    
#     def get_iv_raw(self, line, pixel):
#         """
#         Returns
#         -------
#         tuple
#             (Bias (V), Current (A))
#         """        
#         current_channel = self.get_channel_name('Current', include_avg=self.has_averaged_data())
#         return self.signals['sweep_signal'], self.signals[current_channel][line, pixel]


class line_spectrum: # any spectrum (dIdV, Z, I, ...) vs sweep_signal at any point.
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    # def get (self, line, sts='scaled', channel='LI Demod 1 X (A)', factor=0.2, offset='none', delete_zero_bias=True):
    def get (self, line, processing='scaled', **kwargs):
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
        # return self.signals['sweep_signal'], np.median(self.get_didv_numerical(line, pixel)[1]/self.signals[channel][line, pixel])*self.signals[channel][line, pixel]
        return self.signals['sweep_signal'], \
        np.median(self.get_didv_numerical(line, pixel)[1]/self.get_didv_raw(line, pixel, channel, offset)[1])\
        *self.get_didv_raw(line, pixel, channel, offset)[1]
    
    # def get_didv_normalized (self, line, pixel, channel='LI Demod 1 X (A)', factor=0.2, offset='none', delete_zero_bias=True):
    #     """
    #     Returns
    #     -------
    #     tuple
    #         (Bias (V), normalized dIdV)
    #     """
    #     # dIdV, V = self.get_didv_scaled(line, pixel, channel)[1], self.signals['sweep_signal']
    #     V, dIdV = self.get_didv_scaled(line, pixel, channel, offset = 'none')
    #     I_cal = cumtrapz(dIdV, V, initial=0)
    # 
    #     zero = np.argwhere ( abs(V) == np.min(abs(V)) )[0, 0] # The index where V = 0 or nearest to 0.
    #     popt, pcov = curve_fit (lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
    #     I_cal -= popt[1]
    # 
    #     # get total conductance I/V
    #     with np.errstate(divide='ignore'): # Ignore the warning of 'division by zero'.
    #         IV_cal = I_cal/V
    # 
    #     # I_cal/V = 0/0으로 계산되는 경우
    #     # nan으로 처리됨. 이 값 제외를 위해 nanmedian 사용.
    #     delta = factor*np.nanmedian(IV_cal)
    #     Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
    #     if delete_zero_bias == False:
    #         return V, Normalized_dIdV
    #     else:
    #         return np.delete(V, zero), np.delete(Normalized_dIdV, zero)
    # 
    # def get_didv_normalized_rev(self, line, pixel, channel='LI Demod 1 X', factor=0.2, offset='none', delete_zero_bias=True):
    #     """
    #     Returns
    #     -------
    #     tuple
    #         (Bias (V), normalized dIdV)
    #     """
    #     V, dIdV = self.get_didv_scaled(line, pixel, channel, offset='none')
    #     I_cal = cumtrapz(dIdV, V, initial=0)
    # 
    #     zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
    # 
    #     with np.errstate(divide='ignore'): # Ignore the warning of zero division.
    #         if V[zero] == 0: # The case V has 0 as an element.
    #             I_cal -= I_cal[zero]  # Offset for I(V=0) = 0
    #             IV_cal = I_cal/V
    #             
    #             # linear interpolation for I/V at 0 V: y = mx + b
    #             m = (IV_cal[zero+1] - IV_cal[zero-1]) / (V[zero+1] - V[zero-1])
    #             b = IV_cal[zero+1] - m * V[zero+1]
    #             IV_cal[zero] = b
    #         else:
    #             popt, _ = curve_fit(lambda x, a, b: a*x + b, V[zero-1:zero+2], I_cal[zero-1:zero+2])
    #             I_cal -= popt[1]
    #             IV_cal = I_cal/V
    #     
    #     delta = factor*np.nanmedian(IV_cal)
    #     Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
    #     
    #     if delete_zero_bias:
    #         return np.delete(V, zero), np.delete(Normalized_dIdV, zero)
    #     else:
    #         return V, Normalized_dIdV

    def get_didv_normalized(self, line, pixel, channel='LI Demod 1 X (A)', factor=0.2, offset='none', delete_zero_bias=True):
        """
        Returns
        -------
        tuple
            (Bias (V), normalized dIdV)
        """
        V, dIdV = self.get_didv_scaled(line, pixel, channel, offset=offset)
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

    def get_didv_numerical (self, line, pixel):
        '''
        Returns
        -------
        tuple
            (Bias (V), numerical dIdV (S))
        '''
        step = self.signals['sweep_signal'][1] - self.signals['sweep_signal'][0]
        didv = np.gradient(self.signals['Current (A)'][line, pixel], step, edge_order=2) # I-V curve를 직접 미분.
        return self.signals['sweep_signal'], didv