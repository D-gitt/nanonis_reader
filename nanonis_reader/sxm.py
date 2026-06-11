# nanonispy dependency removed — parsing handled by base.py via _vendor
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import lstsq


# ═══════════════════════════════════════════════════════════════════
#  Topography (.sxm)
# ═══════════════════════════════════════════════════════════════════

class topography:
    
    '''
    Topography processor for .sxm files.
    
    Args:
        instance : base.load
            Loaded Nanonis data file.
    
    Methods:
        get_z(processing, scan_direction)
        raw(scan_direction)
        subtract_average(scan_direction)
        subtract_linear_fit(scan_direction)
        subtract_linear_fit_xy(scan_direction)
        subtract_parabolic_fit(scan_direction)
        subtract_plane_fit(scan_direction)
        differentiate(scan_direction)
    '''
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def _get_channel(self, channel, scan_direction='fwd'):
        """Get 2D image data for a channel with scan direction handling."""
        if scan_direction == 'fwd':
            return self.signals[channel]['forward']
        elif scan_direction == 'bwd':
            return np.flip(self.signals[channel]['backward'], axis=1)
        else:
            raise ValueError(f"scan_direction must be 'fwd' or 'bwd', got '{scan_direction}'")

    def get_z(self, processing='raw', scan_direction='fwd'):
        """Deprecated: call processing methods directly, e.g. ``d.topo.subtract_linear_fit()``."""
        warnings.warn(
            "topography.get_z() is deprecated. Call processing methods directly: "
            "d.topo.raw(), d.topo.subtract_linear_fit(), etc.",
            DeprecationWarning, stacklevel=2
        )
        if processing == 'raw':
            return self.raw(scan_direction)
        elif processing == 'subtract average':
            return self.subtract_average(scan_direction)
        elif processing == 'subtract linear fit':
            return self.subtract_linear_fit(scan_direction)
        elif processing == 'subtract linear fit xy':
            return self.subtract_linear_fit_xy(scan_direction)
        elif processing == 'subtract parabolic fit':
            return self.subtract_parabolic_fit(scan_direction)
        elif processing == 'subtract plane fit':
            return self.subtract_plane_fit(scan_direction)
        elif processing == 'differentiate':
            return self.differentiate(scan_direction)
        else:
            raise ValueError(f"Unknown processing: '{processing}'")
        
    def raw(self, scan_direction='fwd'):
        return self._get_channel('Z', scan_direction)
    
    def subtract_average(self, scan_direction='fwd'):
        from . import image_processing as ip
        return ip.subtract_average(self.raw(scan_direction))

    def subtract_linear_fit(self, scan_direction='fwd'):        
        from . import image_processing as ip
        return ip.subtract_linear_fit(self.raw(scan_direction))

    def subtract_linear_fit_xy(self, scan_direction='fwd'):
        from . import image_processing as ip
        return ip.subtract_linear_fit_xy(self.raw(scan_direction))

    def subtract_parabolic_fit(self, scan_direction='fwd'):        
        from . import image_processing as ip
        return ip.subtract_parabolic_fit(self.raw(scan_direction))

    def subtract_plane_fit(self, scan_direction='fwd'):      
        from . import image_processing as ip
        return ip.subtract_plane_fit(self.raw(scan_direction))

    def differentiate(self, scan_direction='fwd'):
        from . import image_processing as ip
        xrange = round(self.header['scan_range'][0] * 1e9) * 1e-9
        pixels = int(self.header['scan>pixels/line'])
        dx = xrange / pixels
        return ip.differentiate(self.raw(scan_direction), dx)


# ═══════════════════════════════════════════════════════════════════
#  dI/dV map (.sxm)
# ═══════════════════════════════════════════════════════════════════

class didvmap:
    
    '''
    dI/dV map processor for .sxm files.
    
    Args:
        instance : base.load
            Loaded Nanonis data file.
    
    Methods:
        get_map(processing, scan_direction, channel)
        raw(scan_direction, channel)
        subtract_linear_fit(scan_direction, channel)
        subtract_linear_fit_xy(scan_direction, channel)
    '''
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals

    def _get_channel(self, channel, scan_direction='fwd'):
        """Get 2D image data for a channel with scan direction handling."""
        if scan_direction == 'fwd':
            return self.signals[channel]['forward']
        elif scan_direction == 'bwd':
            return np.flip(self.signals[channel]['backward'], axis=1)
        else:
            raise ValueError(f"scan_direction must be 'fwd' or 'bwd', got '{scan_direction}'")

    def get_map(self, processing='raw', scan_direction='fwd', channel='LI_Demod_1_X'):
        """Deprecated: call processing methods directly, e.g. ``d.didv.raw()``."""
        warnings.warn(
            "didvmap.get_map() is deprecated. Call processing methods directly: "
            "d.didv.raw(), d.didv.subtract_linear_fit(), etc.",
            DeprecationWarning, stacklevel=2
        )
        if processing == 'raw':
            return self.raw(scan_direction, channel)
        elif processing == 'subtract linear fit':
            return self.subtract_linear_fit(scan_direction, channel)
        elif processing == 'subtract linear fit xy':
            return self.subtract_linear_fit_xy(scan_direction, channel)
        else:
            raise ValueError(f"Unknown processing: '{processing}'")
        
    def raw(self, scan_direction='fwd', channel='LI_Demod_1_X'):
        return self._get_channel(channel, scan_direction)
    
    def subtract_linear_fit(self, scan_direction='fwd', channel='LI_Demod_1_X'):        
        from . import image_processing as ip
        return ip.subtract_linear_fit(self.raw(scan_direction, channel))
    
    def subtract_linear_fit_xy(self, scan_direction='fwd', channel='LI_Demod_1_X'):
        from . import image_processing as ip
        return ip.subtract_linear_fit_xy(self.raw(scan_direction, channel))


# ═══════════════════════════════════════════════════════════════════
#  Current map (.sxm)
# ═══════════════════════════════════════════════════════════════════

class currentmap:
    
    '''
    Current map processor for .sxm files.
    
    Args:
        instance : base.load
            Loaded Nanonis data file.
    
    Methods:
        get_map(scan_direction) → 2D current array
    '''
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals

    def _get_channel(self, channel, scan_direction='fwd'):
        """Get 2D image data for a channel with scan direction handling."""
        if scan_direction == 'fwd':
            return self.signals[channel]['forward']
        elif scan_direction == 'bwd':
            return np.flip(self.signals[channel]['backward'], axis=1)
        else:
            raise ValueError(f"scan_direction must be 'fwd' or 'bwd', got '{scan_direction}'")

    def raw(self, scan_direction='fwd'):
        """Raw current map."""
        return self._get_channel('Current', scan_direction)

    def get_map(self, scan_direction='fwd'):
        """Deprecated: use ``d.current.raw()`` instead."""
        warnings.warn(
            "currentmap.get_map() is deprecated. Use d.current.raw() instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.raw(scan_direction)


# ═══════════════════════════════════════════════════════════════════
#  FFT utilities (.sxm)
# ═══════════════════════════════════════════════════════════════════

class fft:
    
    '''
    FFT processor for .sxm files.
    
    Methods:
        two_d_FFT_sqrt(image) → sqrt(|FFT|)
        two_d_FFT_log(image)  → log(|FFT|)
        two_d_FFT_lin(image)  → |FFT|
    '''

    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        
    def two_d_FFT_sqrt(self, image):
        '''
        Parameters
        ----------
        image : 2D numpy input
        Calculate FFT

        Returns
        -------
        image_fit : 2D numpy output
        Check the scale bar
        '''
        fft = np.fft.fft2(image) # FFT only
        fft_shift = np.fft.fftshift(fft) #2D FFT 를 위한 image shift
        image_fft = np.sqrt(np.abs(fft_shift))
        return image_fft
    
    def two_d_FFT_log(self, image):
        '''
        Parameters
        ----------
        image : 2D numpy input
        Calculate FFT

        Returns
        -------
        image_fit : 2D numpy output
        Check the scale bar
        '''
        fft = np.fft.fft2(image) # FFT only
        fft_shift = np.fft.fftshift(fft) #2D FFT 를 위한 image shift
        image_fft = np.log(np.abs(fft_shift))
        return image_fft
    
    def two_d_FFT_lin(self, image):
        '''
        Parameters
        ----------
        image : 2D numpy input
        Calculate FFT

        Returns
        -------
        image_fit : 2D numpy output
        Check the scale bar
        '''
        fft = np.fft.fft2(image) # FFT only
        fft_shift = np.fft.fftshift(fft) #2D FFT 를 위한 image shift
        image_fft = np.abs(fft_shift)
        return image_fft