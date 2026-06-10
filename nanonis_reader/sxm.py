# nanonispy dependency removed — parsing handled by base.py via _vendor
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import lstsq


        

class topography:
    
    '''
    Args:
        filepath : str
            Name of the Nanonis spectrum file to be loaded.
    
    Attributes (name : type):
        fname : str
            The name of the file excluding its containing directory.
        header : dict
            Header information of spectrum data.
        signals : dict
            Measured values in spectrum data.

    Methods:
        get_z(self, processing = 'raw', scan_direction = 'fwd')
            Parameters:
            processing : str
                The image processing.
                Possible parameters is the following: 
                    'raw', 'subtract average', 'subtract linear fit', 'subtract parabolic fit', 'differentiate'
            scan_direction : str
                The direction of scan.
                Possible parameters is the following: 
                    'fwd', 'bwd'
            
            Returns the two-dimensional array
            that represents the topographic data
            specifically processed for scanning in the direction of scan_direction.
            
            The detailed processing is carried out through the following five methods:
                raw, subtract_average, subtract_linear_fit, subtract_parabolic_fit, differentiate.
            
        raw (self, scan_direction)
            Returns the two-dimensional array
            containing the raw z data.
        
        subtract_average (self, scan_direction)
            Returns the two-dimensional array
            containing the z data processed through subtract average.
        
        subtract_linear_fit (self, scan_direction)
            Returns the two-dimensional array
            containing the z data processed through subtract linear fit.
        
        subtract_parabolic_fit (self, scan_direction)
            Returns the two-dimensional array
            containing the z data processed through subtract parabolic fit.
        
        differentiate (self, scan_direction)
            Returns the two-dimensional array
            containing the dz/dx data, in which x represents the fast scan axis.
    '''
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def get_z(self, processing = 'raw', scan_direction = 'fwd'): # 'fwd' or 'bwd'
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
        elif processing == 'differentiate':
            return self.differentiate(scan_direction)
        elif processing == 'subtract plane fit':
            return self.subtract_plane_fit(scan_direction)
        
    def raw (self, scan_direction):
        if scan_direction == 'fwd':
            z = self.signals['Z']['forward']
        elif scan_direction == 'bwd':
            import numpy as np
            z = np.flip (self.signals['Z']['backward'], axis = 1)
        return z
    
    def subtract_average (self, scan_direction):
        from . import image_processing as ip
        return ip.subtract_average(self.raw(scan_direction))


    def subtract_linear_fit(self, scan_direction):        
        from . import image_processing as ip
        return ip.subtract_linear_fit(self.raw(scan_direction))
    
    

    def subtract_linear_fit_xy(self, scan_direction):
        from . import image_processing as ip
        return ip.subtract_linear_fit_xy(self.raw(scan_direction))


    def subtract_parabolic_fit(self, scan_direction):        
        from . import image_processing as ip
        return ip.subtract_parabolic_fit(self.raw(scan_direction))
    

    def differentiate(self, scan_direction):
        from . import image_processing as ip
        xrange = round(self.header['scan_range'][0] * 1e9) * 1e-9
        pixels = int(self.header['scan>pixels/line'])
        dx = xrange / pixels
        return ip.differentiate(self.raw(scan_direction), dx)
    
    def subtract_plane_fit (self, scan_direction):      
        from . import image_processing as ip
        return ip.subtract_plane_fit(self.raw(scan_direction))

class didvmap:
    
    '''
    Args:
        filepath : str
            Name of the Nanonis spectrum file to be loaded.
    
    Attributes (name : type):
        fname : str
            The name of the file excluding its containing directory.
        header : dict
            Header information of spectrum data.
        signals : dict
            Measured values in spectrum data.
    
    Methods:
        get_map(self, scan_direction = 'fwd', channel = 'LI_Demod_1_X')
            Parameters:
            scan_direction : str
                The direction of scan.
                Possible parameters is the following: 
                    'fwd', 'bwd'
            channel : str
                The channel to be returned.
                'LI_Demod_1_X' channel is returned by default.
                Other channels can also be returned if the input file contains. e.g. 'LI_Demod_1_Y', 'LI_Demod_2_X', ...
            
            Returns the two-dimensional array
            that represents the dI/dV map data scanned in the direction of scan_direction.
    '''
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    

    def get_map(self, processing = 'raw', scan_direction = 'fwd', channel = 'LI_Demod_1_X'):
        if processing == 'raw':
            return self.raw(scan_direction, channel)
        elif processing == 'subtract linear fit':
            return self.subtract_linear_fit(scan_direction, channel)
        elif processing == 'subtract linear fit xy':
            return self.subtract_linear_fit_xy(scan_direction, channel)
        
    def raw(self, scan_direction, channel):
        if scan_direction == 'fwd':
            didv = self.signals[channel]['forward']
        elif scan_direction == 'bwd':
            didv = np.flip(self.signals[channel]['backward'], axis = 1)
        return didv
    
    def subtract_linear_fit(self, scan_direction, channel):        
        z = self.raw(scan_direction, channel)
        lines, pixels = np.shape(z)
        x = np.arange(pixels)
        
        # nan이 있는 행 찾기
        nan_rows = np.isnan(z).any(axis=1)
        
        # 결과 배열 초기화 (nan으로)
        z_sublf = np.full_like(z, np.nan)
        
        # nan이 없는 행들에 대해 처리
        valid_rows = ~nan_rows
        valid_z = z[valid_rows]
        
        # 한번에 모든 유효한 행에 대해 선형 피팅
        coeffs = np.polyfit(x, valid_z.T, 1)
        fitted = (coeffs[0].reshape(-1,1) * x + coeffs[1].reshape(-1,1))
        
        # 결과 저장
        z_sublf[valid_rows] = valid_z - fitted
        
        return z_sublf
    
    def subtract_linear_fit_xy(self, scan_direction, channel):
        # X 방향 linear fit 제거
        z = self.subtract_linear_fit(scan_direction, channel)
        lines, pixels = np.shape(z)
        
        # y 방향으로의 linear fit을 위한 x 좌표 (실제 물리적 거리 사용)
        yrange = round(self.header['scan_range'][1] * 1e9)*1e-9
        y = np.linspace(0, yrange, lines)
        
        # nan이 있는 열 찾기
        nan_cols = np.isnan(z).any(axis=0)
        
        # 결과 배열 초기화 (nan으로)
        z_sublf = np.full_like(z, np.nan)
        
        # nan이 없는 열들에 대해 처리
        valid_cols = ~nan_cols
        valid_z = z[:, valid_cols]
        
        # 한번에 모든 유효한 열에 대해 선형 피팅
        coeffs = np.polyfit(y, valid_z, 1)
        fitted = (coeffs[0] * y.reshape(-1,1) + coeffs[1])
        
        # 결과 저장
        z_sublf[:, valid_cols] = valid_z - fitted
        
        return z_sublf

class currentmap:
    
    '''
    Args:
        filepath : str
            Name of the Nanonis spectrum file to be loaded.
    
    Attributes (name : type):
        fname : str
            The name of the file excluding its containing directory.
        header : dict
            Header information of spectrum data.
        signals : dict
            Measured values in spectrum data.
    
    Methods:
        get_map(self, scan_direction = 'fwd')
            Parameters:
            scan_direction : str
                The direction of scan.
                Possible parameters is the following: 
                    'fwd', 'bwd'
            
            Returns the two-dimensional array
            that represents the current map data scanned in the direction of scan_direction.
    '''
    
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
    
    def get_map(self, scan_direction = 'fwd'):
        if scan_direction == 'fwd':
            current = self.signals['Current']['forward']
        elif scan_direction == 'bwd':
            current = np.flip(self.signals['Current']['backward'], axis = 1)
        return current

class fft:
    
    '''
    Args:
        filepath : str
            Name of the Nanonis spectrum file to be loaded.
    
    Attributes (name : type):
        fname : str
            The name of the file excluding its containing directory.
        header : dict
            Header information of spectrum data.
        signals : dict
            Measured values in spectrum data.
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