# nanonispy dependency removed — parsing handled by base.py via _vendor
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import lstsq
try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz


# ═══════════════════════════════════════════════════════════════════
# Topography (2D image processing)
# ═══════════════════════════════════════════════════════════════════

class topography:
        
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def get_z (self, processing = 'raw'):
        """Deprecated: call processing methods directly, e.g. ``d.topo.subtract_linear_fit()``."""
        warnings.warn(
            "topography.get_z() is deprecated. Call processing methods directly: "
            "d.topo.raw(), d.topo.subtract_linear_fit(), etc.",
            DeprecationWarning, stacklevel=2
        )
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

    def subtract_parabolic_fit(self):        
        from . import image_processing as ip
        return ip.subtract_parabolic_fit(self.raw())
    
    def differentiate(self):
        from . import image_processing as ip
        xrange = round(self.header['size_xy'][0] * 1e9) * 1e-9
        pixels = int(self.header['dim_px'][0])
        dx = xrange / pixels
        return ip.differentiate(self.raw(), dx)


# ═══════════════════════════════════════════════════════════════════
# STS (3D spectroscopy: dI/dV vs Bias)
# ═══════════════════════════════════════════════════════════════════

class sts:
    """Unified STS grid spectroscopy.
    
    All processing methods return full 3D arrays (lines, pixels, bias_points).
    Use numpy slicing for point / map / line extraction:
    
        V, didv = d.sts.scaled()
        didv[line, pixel]         # point spectrum
        didv[:, :, sweep_idx]     # 2D map at specific bias
        didv[line]                # line spectrum (pixels × bias)
    """

    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals

    @property
    def sweep_signal(self):
        """1D bias voltage array."""
        return self.signals['sweep_signal']

    # ── Channel resolution (single implementation) ──────────────

    def _resolve_channel(self, base_channel, sweep_direction='fwd', sweep_index=None):
        """Resolve channel name (delegates to spectral_analysis.resolve_channel)."""
        from . import spectral_analysis as sa
        return sa.resolve_channel(self.signals, base_channel, sweep_direction, sweep_index)

    def _get_3d(self, base_channel, sweep_direction='fwd'):
        """Get full 3D signal array for a channel."""
        resolved = self._resolve_channel(base_channel, sweep_direction)
        return self.signals[resolved]

    # ── Processing methods (return full 3D) ─────────────────────

    def raw(self, channel='LI Demod 1 X (A)', offset='none', sweep_direction='fwd'):
        """
        Raw dI/dV signal from lock-in amplifier.
        
        Returns
        -------
        tuple
            (V, dIdV_3d) where dIdV_3d has shape (lines, pixels, bias_points)
        """
        data = np.copy(self._get_3d(channel, sweep_direction))
        if isinstance(offset, np.ndarray):
            data = data - offset
        return self.sweep_signal, data

    def scaled(self, channel='LI Demod 1 X', offset='none', sweep_direction='fwd'):
        """
        Scaled dI/dV (lock-in signal scaled by numerical derivative ratio).
        
        Returns
        -------
        tuple
            (V, dIdV_scaled_3d) where shape is (lines, pixels, bias_points)
        """
        ch = channel + ' (A)' if '(A)' not in channel else channel
        _, didv_raw = self.raw(ch, offset, sweep_direction)
        
        step = self.sweep_signal[1] - self.sweep_signal[0]
        current = self._get_3d('Current (A)', sweep_direction)
        didv_numerical = np.gradient(current, step, axis=2, edge_order=2)
        
        scale_factors = np.nanmedian(didv_numerical / didv_raw, axis=2)[..., np.newaxis]
        return self.sweep_signal, scale_factors * didv_raw

    def normalized(self, channel='LI Demod 1 X', factor=0.2, offset='none',
                   delete_zero_bias=False, sweep_direction='fwd'):
        """
        Normalized dI/dV (vectorized 3D volume).
        
        Returns
        -------
        tuple
            (V, normalized_dIdV_3d)
            Note: if delete_zero_bias=True, bias dimension is reduced by 1.
        """
        V, dIdV = self.scaled(channel, offset, sweep_direction)
        
        I_cal = cumtrapz(dIdV, V, initial=0, axis=2)
            
        zero = np.argwhere(abs(V) == np.min(abs(V)))[0, 0]
        
        # Vectorized linear offset correction at V≈0 (3 points)
        x = V[zero-1:zero+2]
        y = I_cal[:, :, zero-1:zero+2]
        
        lines, pixels, sweeps = I_cal.shape
        y_reshaped = y.reshape(lines * pixels, 3).T
        
        # NaN defence (scan interrupted → some pixels have NaN)
        valid_cols = ~np.isnan(y_reshaped).any(axis=0)
        b_offset = np.full(lines * pixels, np.nan)
        a_slope = np.full(lines * pixels, np.nan)
        
        if np.any(valid_cols):
            popt = np.polyfit(x, y_reshaped[:, valid_cols], 1)
            a_slope[valid_cols] = popt[0]   # slope (for L'Hôpital)
            b_offset[valid_cols] = popt[1]  # intercept
        
        b_offset = b_offset.reshape(lines, pixels, 1)
        I_cal = I_cal - b_offset
        
        with np.errstate(divide='ignore', invalid='ignore'):
            IV_cal = I_cal / V
            # L'Hôpital: V→0 → I/V = dI/dV ≈ slope from linear fit
            if abs(V[zero]) < 1e-5:
                IV_cal[:, :, zero] = a_slope.reshape(lines, pixels)
            
        delta = factor * np.nanmedian(IV_cal, axis=2)[..., np.newaxis]
        Normalized_dIdV = dIdV / np.sqrt(np.square(delta) + np.square(IV_cal))
        
        if delete_zero_bias:
            Normalized_dIdV = np.delete(Normalized_dIdV, zero, axis=2)
            V = np.delete(V, zero)
            
        return V, Normalized_dIdV

    def numerical(self, sweep_direction='fwd'):
        """
        Numerical dI/dV from current differentiation.
        
        Returns
        -------
        tuple
            (V, dIdV_numerical_3d) where shape is (lines, pixels, bias_points)
        """
        step = self.sweep_signal[1] - self.sweep_signal[0]
        current = self._get_3d('Current (A)', sweep_direction)
        didv = np.gradient(current, step, axis=2, edge_order=2)
        return self.sweep_signal, didv

    def iv(self, sweep_direction='fwd'):
        """
        Raw I-V curves.
        
        Returns
        -------
        tuple
            (V, current_3d) where shape is (lines, pixels, bias_points)
        """
        data = self._get_3d('Current (A)', sweep_direction)
        return self.sweep_signal, data

    def currentmap(self, sweep_idx, sweep_direction='fwd'):
        """Current map at a specific bias index.
        
        Returns
        -------
        np.ndarray
            2D array (lines, pixels)
        """
        current = self._get_3d('Current (A)', sweep_direction)
        return current[:, :, sweep_idx]


# ═══════════════════════════════════════════════════════════════════
# I-z (3D spectroscopy: Current vs Z)
# ═══════════════════════════════════════════════════════════════════

class iz:
    """Unified I-z grid spectroscopy.
    
    All methods return full 3D arrays unless noted.
    Use numpy slicing for point / map extraction:
    
        Z, current = d.iz.raw()
        current[line, pixel]       # point I-z curve
        current[:, :, sweep_idx]   # 2D current map at specific Z
    """

    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals

    @property
    def sweep_signal(self):
        """1D Z array."""
        return self.signals['sweep_signal']

    def _resolve_channel(self, base_channel, sweep_direction='fwd', sweep_index=None):
        """Resolve channel name (delegates to spectral_analysis.resolve_channel)."""
        from . import spectral_analysis as sa
        return sa.resolve_channel(self.signals, base_channel, sweep_direction, sweep_index)

    def _get_3d(self, base_channel, sweep_direction='fwd'):
        """Get full 3D signal array for a channel."""
        resolved = self._resolve_channel(base_channel, sweep_direction)
        return self.signals[resolved]

    # ── Processing methods ──────────────────────────────────────

    def raw(self, sweep_direction='fwd'):
        """
        Raw I-z curves.
        
        Returns
        -------
        tuple
            (Z, current_3d) where shape is (lines, pixels, z_points)
        """
        data = self._get_3d('Current (A)', sweep_direction)
        return self.sweep_signal, data

    # Physical constants for barrier height calculation
    _HBAR_C = 6.582119569e-16 * 2.99792458e+8  # ℏc (eV·m)
    _ME = 0.51099895e+6  # electron mass (eV/c²)

    def barrier_height(self, fitting_current_range=(1e-12, 10e-12), sweep_direction='fwd'):
        """
        Apparent barrier height map (2D).
        
        Uses polyfit on ln(I) vs Z within the specified current range.
        Pre-computes log(|I|) for the entire 3D volume, then loops per-pixel
        because each pixel has a different fitting range.
        
        Returns
        -------
        np.ndarray
            2D array (lines, pixels) of barrier height in eV.
        """
        lines, pixels = self.header['dim_px'][1], self.header['dim_px'][0]
        z = self.sweep_signal
        
        # Pre-compute log(|I|) for the entire 3D volume at once
        if sweep_direction == 'AVG':
            fwd_ch = self._resolve_channel('Current (A)', sweep_direction='fwd')
            bwd_ch = self._resolve_channel('Current (A)', sweep_direction='bwd')
            I_3d = np.abs(np.nanmean([self.signals[fwd_ch], self.signals[bwd_ch]], axis=0))
        else:
            resolved = self._resolve_channel('Current (A)', sweep_direction=sweep_direction)
            I_3d = np.abs(self.signals[resolved])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            log_I_3d = np.log(I_3d)
        
        arr = np.zeros((lines, pixels))
        for i in range(lines):
            for j in range(pixels):
                try:
                    I_abs = I_3d[i, j]
                    log_I = log_I_3d[i, j]
                    idx = np.where((fitting_current_range[0] <= I_abs) & (I_abs <= fitting_current_range[1]))
                    slope, intercept = np.polyfit(z[idx], log_I[idx], 1)
                    arr[i, j] = ((slope * self._HBAR_C / (-2))**2) / (2 * self._ME)
                except Exception as e:
                    print(f'Estimation error at: {i, j}. {str(e)}')
                    arr[i, j] = np.nan
        return arr

    def barrier_height_at(self, line, pixel, fitting_current_range=(1e-12, 10e-12), sweep_direction='fwd'):
        """
        Apparent barrier height at a single pixel.
        
        Returns
        -------
        float
            Barrier height in eV.
        """
        z = self.sweep_signal
        
        if sweep_direction == 'AVG':
            fwd_ch = self._resolve_channel('Current (A)', sweep_direction='fwd')
            bwd_ch = self._resolve_channel('Current (A)', sweep_direction='bwd')
            I_abs = np.abs(np.nanmean([self.signals[fwd_ch][line, pixel], 
                                        self.signals[bwd_ch][line, pixel]], axis=0))
        else:
            resolved = self._resolve_channel('Current (A)', sweep_direction=sweep_direction)
            I_abs = np.abs(self.signals[resolved][line, pixel])
        
        log_I = np.log(I_abs)
        idx = np.where((fitting_current_range[0] <= I_abs) & (I_abs <= fitting_current_range[1]))
        slope, intercept = np.polyfit(z[idx], log_I[idx], 1)
        return ((slope * self._HBAR_C / (-2))**2) / (2 * self._ME)


# ═══════════════════════════════════════════════════════════════════
# Deprecated wrappers (backward compatibility)
# ═══════════════════════════════════════════════════════════════════

class point_didv:
    """Deprecated: use ``sts`` instead.
    
    Examples
    --------
    Old: ``d.point.get_didv_scaled(line, pixel)``
    New: ``V, didv = d.sts.scaled(); didv[line, pixel]``
    """
    def __init__(self, instance):
        warnings.warn(
            "point_didv is deprecated. Use sts instead: "
            "V, data = d.sts.scaled(); data[line, pixel]",
            DeprecationWarning, stacklevel=2
        )
        self._sts = sts(instance)

    def _resolve_channel(self, *args, **kwargs):
        return self._sts._resolve_channel(*args, **kwargs)

    def _get_data(self, base_channel, line, pixel, sweep_direction=None, sweep_index=None):
        resolved = self._sts._resolve_channel(base_channel, sweep_direction, sweep_index)
        if isinstance(resolved, list):
            return np.stack([self._sts.signals[ch][line, pixel] for ch in resolved])
        else:
            return self._sts.signals[resolved][line, pixel]

    def _normalize_single(self, V, dIdV, factor=0.2, delete_zero_bias=True):
        from . import spectral_analysis as sa
        return sa.normalize_didv(V, dIdV, factor, delete_zero_bias)

    def get_didv_raw(self, line, pixel, channel='LI Demod 1 X (A)', offset='none', sweep_direction=None, sweep_index=None):
        V, data = self._sts.raw(channel, offset, sweep_direction)
        if sweep_index == 'all':
            resolved = self._sts._resolve_channel(channel, sweep_direction, sweep_index='all')
            return V, np.stack([self._sts.signals[ch][line, pixel] for ch in resolved])
        elif sweep_index is not None:
            ch = self._sts._resolve_channel(channel, sweep_direction, sweep_index)
            return V, self._sts.signals[ch][line, pixel]
        return V, data[line, pixel]

    def get_didv_scaled(self, line, pixel, channel='LI Demod 1 X', offset='none', sweep_direction=None, sweep_index=None):
        if sweep_index is not None:
            # For individual sweep index, fall back to per-pixel computation
            ch = channel + ' (A)' if '(A)' not in channel else channel
            V, raw = self.get_didv_raw(line, pixel, ch, offset, sweep_direction, sweep_index)
            step = self._sts.sweep_signal[1] - self._sts.sweep_signal[0]
            current = self._get_data('Current (A)', line, pixel, sweep_direction, sweep_index)
            if current.ndim == 2:
                didv_num = np.gradient(current, step, axis=1, edge_order=2)
                scale = np.median(didv_num / raw, axis=1, keepdims=True)
            else:
                didv_num = np.gradient(current, step, edge_order=2)
                scale = np.median(didv_num / raw)
            return V, scale * raw
        V, data = self._sts.scaled(channel, offset, sweep_direction)
        return V, data[line, pixel]

    def get_didv_normalized(self, line, pixel, channel='LI Demod 1 X', factor=0.2, offset='none',
                            delete_zero_bias=True, sweep_direction=None, sweep_index=None):
        from . import spectral_analysis as sa
        V, dIdV = self.get_didv_scaled(line, pixel, channel, offset, sweep_direction, sweep_index)
        if dIdV.ndim == 2:
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

    def get_didv_numerical(self, line, pixel, sweep_direction=None, sweep_index=None):
        if sweep_index is not None:
            step = self._sts.sweep_signal[1] - self._sts.sweep_signal[0]
            current = self._get_data('Current (A)', line, pixel, sweep_direction, sweep_index)
            if current.ndim == 2:
                didv = np.gradient(current, step, axis=1, edge_order=2)
            else:
                didv = np.gradient(current, step, edge_order=2)
            return self._sts.sweep_signal, didv
        V, data = self._sts.numerical(sweep_direction)
        return V, data[line, pixel]

    def get_iv_raw(self, line, pixel, sweep_direction=None, sweep_index=None):
        if sweep_index is not None:
            data = self._get_data('Current (A)', line, pixel, sweep_direction, sweep_index)
            return self._sts.sweep_signal, data
        V, data = self._sts.iv(sweep_direction)
        return V, data[line, pixel]


class didvmap(point_didv):
    """Deprecated: use ``sts`` instead.
    
    Old: ``d.didv.scaled(sweep_idx)``
    New: ``V, data = d.sts.scaled(); data[:, :, sweep_idx]``
    """
    def __init__(self, instance):
        # Suppress the parent deprecation warning to avoid double warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            super().__init__(instance)
        warnings.warn(
            "didvmap is deprecated. Use sts instead: "
            "V, data = d.sts.scaled(); data[:, :, sweep_idx]",
            DeprecationWarning, stacklevel=2
        )

    def get_didvmap(self, sweep_idx, processing='raw', channel='LI Demod 1 X (A)', sweep_direction=None, **kwargs):
        if processing == 'raw':
            V, data = self._sts.raw(channel, sweep_direction=sweep_direction)
            return data[:, :, sweep_idx]
        elif processing == 'scaled':
            ch = channel.replace(' (A)', '') if '(A)' in channel else channel
            V, data = self._sts.scaled(ch, sweep_direction=sweep_direction, **kwargs)
            return data[:, :, sweep_idx]
        elif processing == 'normalized':
            ch = channel.replace(' (A)', '') if '(A)' in channel else channel
            V, data = self._sts.normalized(ch, sweep_direction=sweep_direction, **kwargs)
            return data[:, :, sweep_idx]
        else:
            raise ValueError(f"Unknown processing method: {processing}")

    def raw(self, sweep_idx, channel='LI Demod 1 X (A)', sweep_direction=None):
        V, data = self._sts.raw(channel, sweep_direction=sweep_direction)
        return data[:, :, sweep_idx]

    def scaled(self, sweep_idx, channel='LI Demod 1 X', offset='none', sweep_direction=None):
        V, data = self._sts.scaled(channel, offset, sweep_direction)
        return data[:, :, sweep_idx]

    def normalized(self, sweep_idx, channel='LI Demod 1 X', factor=0.2, offset='none',
                   delete_zero_bias=True, sweep_direction=None):
        V, data = self._sts.normalized(channel, factor, offset, delete_zero_bias, sweep_direction)
        return data[:, :, sweep_idx]

    def get_currentmap(self, sweep_idx, sweep_direction=None):
        return self._sts.currentmap(sweep_idx, sweep_direction)

    def get_sweepsignal(self, sweep_idx):
        return self._sts.sweep_signal[sweep_idx]


class point_iz:
    """Deprecated: use ``iz`` instead.
    
    Old: ``d.point_iz.get_iz_raw(line, pixel)``
    New: ``Z, data = d.iz.raw(); data[line, pixel]``
    """
    def __init__(self, instance):
        warnings.warn(
            "point_iz is deprecated. Use iz instead: "
            "Z, data = d.iz.raw(); data[line, pixel]",
            DeprecationWarning, stacklevel=2
        )
        self._iz = iz(instance)

    def _resolve_channel(self, *args, **kwargs):
        return self._iz._resolve_channel(*args, **kwargs)

    def _get_data(self, base_channel, line, pixel, sweep_direction=None, sweep_index=None):
        resolved = self._iz._resolve_channel(base_channel, sweep_direction, sweep_index)
        if isinstance(resolved, list):
            return np.stack([self._iz.signals[ch][line, pixel] for ch in resolved])
        else:
            return self._iz.signals[resolved][line, pixel]

    def get_iz_raw(self, line, pixel, sweep_direction=None, sweep_index=None):
        if sweep_index is not None:
            data = self._get_data('Current (A)', line, pixel, sweep_direction, sweep_index)
            return self._iz.sweep_signal, data
        Z, data = self._iz.raw(sweep_direction)
        return Z, data[line, pixel]

    def get_apparent_barrier_height(self, line, pixel, fitting_current_range=(1e-12, 10e-12), 
                                     sweep_direction=None, _log_I_cache=None):
        return self._iz.barrier_height_at(line, pixel, fitting_current_range, sweep_direction)


class izmap(point_iz):
    """Deprecated: use ``iz`` instead.
    
    Old: ``d.iz.get_izmap(sweep_idx)``
    New: ``Z, data = d.iz.raw(); data[:, :, sweep_idx]``
    """
    def __init__(self, instance):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            super().__init__(instance)
        warnings.warn(
            "izmap is deprecated. Use iz instead: "
            "Z, data = d.iz.raw(); data[:, :, sweep_idx]",
            DeprecationWarning, stacklevel=2
        )

    def get_izmap(self, sweep_idx, sweep_direction=None):
        sd = sweep_direction if sweep_direction is not None else self._iz.sweep_dir
        if sd == 'AVG':
            fwd_ch = self._iz._resolve_channel('Current (A)', sweep_direction='fwd')
            bwd_ch = self._iz._resolve_channel('Current (A)', sweep_direction='bwd')
            fwd_current = self._iz.signals[fwd_ch][:, :, sweep_idx]
            bwd_current = self._iz.signals[bwd_ch][:, :, sweep_idx]
            return np.nanmean([fwd_current, bwd_current], axis=0)
        else:
            resolved = self._iz._resolve_channel('Current (A)', sweep_direction=sd)
            return self._iz.signals[resolved][:, :, sweep_idx]

    def get_apparent_barrier_height_map(self, fitting_current_range=(1e-12, 10e-12), sweep_direction=None):
        return self._iz.barrier_height(fitting_current_range, sweep_direction)

    def get_sweepsignal(self, sweep_idx):
        return self._iz.sweep_signal[sweep_idx]


class line_spectrum:
    """Deprecated: use ``sts`` instead.
    
    Old: ``d.linespec.get(line, processing='scaled')``
    New: ``V, data = d.sts.scaled(); data[line]``
    """
    def __init__(self, instance):
        warnings.warn(
            "line_spectrum is deprecated. Use sts instead: "
            "V, data = d.sts.scaled(); data[line]",
            DeprecationWarning, stacklevel=2
        )
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        self._sts = sts(instance)

    def get(self, line, processing='scaled', **kwargs):
        if processing == 'scaled':
            V, data = self._sts.scaled(**kwargs)
        elif processing == 'raw':
            V, data = self._sts.raw(**kwargs)
        elif processing == 'numerical':
            V, data = self._sts.numerical(**kwargs)
        elif processing == 'normalized':
            V, data = self._sts.normalized(**kwargs)
        else:
            raise ValueError(f"Unknown processing: {processing}")
        return data[line].T  # (bias, pixels) → matches legacy shape

    def get_didv_raw(self, line, pixel, channel='none', offset='none'):
        if channel == 'none':
            if 'LI Demod 2 X (A)' in self.signals.keys():
                channel = 'LI Demod 2 X (A)'
            elif 'LI Demod 1 X (A)' in self.signals.keys():
                channel = 'LI Demod 1 X (A)'
        V, data = self._sts.raw(channel, offset)
        return V, data[line, pixel]

    def get_didv_scaled(self, line, pixel, channel='LI Demod 2 X (A)', offset='none'):
        ch = channel.replace(' (A)', '') if '(A)' in channel else channel
        V, data = self._sts.scaled(ch, offset)
        return V, data[line, pixel]

    def get_didv_normalized(self, line, pixel, channel='LI Demod 1 X (A)', factor=0.2,
                            offset='none', delete_zero_bias=True):
        from . import spectral_analysis as sa
        V, dIdV = self.get_didv_scaled(line, pixel, channel, offset)
        return sa.normalize_didv(V, dIdV, factor, delete_zero_bias)

    def get_didv_numerical(self, line, pixel):
        V, data = self._sts.numerical()
        return V, data[line, pixel]