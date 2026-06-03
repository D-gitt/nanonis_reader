# nanonispy dependency removed — parsing handled by base.py via _vendor
import os
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import lstsq


# class load:
#     def __init__(self, filepath):
#         self.fname = os.path.basename(filepath)
#         self.header = nap.read.Scan(filepath).header
#         self.signals = nap.read.Scan(filepath).signals
        

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
        z = self.raw(scan_direction)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            z_subav = z - np.nanmean(z, axis=1, keepdims=True)
        return z_subav

    # def subtract_linear_fit (self, scan_direction):
    #     import numpy as np
    #     from scipy.optimize import curve_fit
    #     def f_lin(x, a, b): return a*x + b
    #     xrange = round(self.header['scan_range'][0] * 1e9)*1e-9
    #     z = self.raw(scan_direction)
    #     z_sublf = np.zeros(np.shape(z))
    #     lines, pixels = np.shape(z)
    #     for i in range(lines):
    #         if np.shape(np.where(np.isnan(z))[0])[0] != 0: # image에 nan값이 포함되어 있을 경우 (== scan을 도중에 멈추었을 경우)
    #             # if i < np.min(np.where(np.isnan(z))[0]): # nan이 등장하는 line 이전에 대해서 linear fit. 이후는 모두 nan으로 밀기.
    #             if i not in set(np.where(np.isnan(z))[0]): # nan이 등장하는 line 만 선택. 방법: nan인 모든 point의 line을 다 불러들인후 list(set()). set은 중복 원소 제거 위함.
    #                 x = np.linspace(0, xrange, pixels)
    #                 popt, pcov = curve_fit(f_lin, x, z[i])
    #                 z_sublf[i] = z[i] - f_lin(x, *popt)
    #             else:
    #                 z_sublf[i] = np.nan
    #         else:
    #             x = np.linspace(0, xrange, pixels)
    #             popt, pcov = curve_fit(f_lin, x, z[i]) # x - ith line: linear fitting
    #             z_sublf[i] = z[i] - f_lin(x, *popt)

    #     return z_sublf

    def subtract_linear_fit(self, scan_direction):        
        z = self.raw(scan_direction)
        lines, pixels = np.shape(z)
        x = np.arange(pixels)
        
        nan_rows = np.isnan(z).all(axis=1) # 전부 NaN인 행
        partial_rows = np.isnan(z).any(axis=1) & ~nan_rows # 일부만 NaN인 행 (스캔 도중 멈춘 라인)
        valid_rows = ~np.isnan(z).any(axis=1) # 완벽한 행
        
        z_sublf = np.full_like(z, np.nan)
        
        if np.any(valid_rows):
            valid_z = z[valid_rows]
            coeffs = np.polyfit(x, valid_z.T, 1)
            fitted = (coeffs[0].reshape(-1,1) * x + coeffs[1].reshape(-1,1))
            z_sublf[valid_rows] = valid_z - fitted
            
        # 스캔 도중 멈춘 행들에 대한 부분 피팅
        for i in np.where(partial_rows)[0]:
            valid_idx = ~np.isnan(z[i])
            if np.sum(valid_idx) > 1: # 최소 2점 필요
                popt = np.polyfit(x[valid_idx], z[i][valid_idx], 1)
                fitted = popt[0]*x + popt[1]
                z_sublf[i] = z[i] - fitted
                
        return z_sublf
    
    
    # def subtract_linear_fit_xy (self, scan_direction):       
    #     def f_lin(x, a, b): return a*x + b
    #     xrange = round(self.header['scan_range'][0] * 1e9)*1e-9
    #     z = self.subtract_linear_fit(scan_direction)
    #     z_sublf = np.zeros(np.shape(z))
    #     lines, pixels = np.shape(z)
    #     for i in range(lines):
    #         if np.shape(np.where(np.isnan(z))[0])[0] != 0: # image에 nan값이 포함되어 있을 경우 (== scan을 도중에 멈추었을 경우)
    #             if i < np.min(np.where(np.isnan(z))[0]):
    #                 x = np.linspace(0, xrange, pixels)
    #                 popt, pcov = curve_fit(f_lin, x, z.T[i])
    #                 z_sublf[i] = z.T[i] - f_lin(x, *popt)
    #             else:
    #                 z_sublf[i] = np.nan
    #         else:
    #             x = np.linspace(0, xrange, pixels)
    #             popt, pcov = curve_fit(f_lin, x, z.T[i]) # x - ith line: linear fitting
    #             z_sublf[i] = z.T[i] - f_lin(x, *popt)

    #     return z_sublf.T

    def subtract_linear_fit_xy(self, scan_direction):
        # X 방향 linear fit 제거된 데이터
        z = self.subtract_linear_fit(scan_direction)
        lines, pixels = np.shape(z)
        
        yrange = round(self.header['scan_range'][1] * 1e9)*1e-9
        y = np.linspace(0, yrange, lines)
        
        nan_cols = np.isnan(z).all(axis=0)
        partial_cols = np.isnan(z).any(axis=0) & ~nan_cols
        valid_cols = ~np.isnan(z).any(axis=0)
        
        z_sublf = np.full_like(z, np.nan)
        
        if np.any(valid_cols):
            valid_z = z[:, valid_cols]
            coeffs = np.polyfit(y, valid_z, 1)
            fitted = (coeffs[0] * y.reshape(-1,1) + coeffs[1])
            z_sublf[:, valid_cols] = valid_z - fitted
            
        for j in np.where(partial_cols)[0]:
            valid_idx = ~np.isnan(z[:, j])
            if np.sum(valid_idx) > 1:
                popt = np.polyfit(y[valid_idx], z[valid_idx, j], 1)
                fitted = popt[0]*y + popt[1]
                z_sublf[:, j] = z[:, j] - fitted
                
        return z_sublf

    # def subtract_parabolic_fit (self, scan_direction):
    #     def f_parab(x, a, b, c): return a*(x**2) + b*x + c
    #     xrange = round(self.header['scan_range'][0] * 1e9)*1e-9
    #     z = self.raw(scan_direction)
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

    def subtract_parabolic_fit(self, scan_direction):        
        z = self.raw(scan_direction)
        lines, pixels = np.shape(z)
        x = np.arange(pixels)
        
        nan_rows = np.isnan(z).all(axis=1) # 전부 NaN인 행
        partial_rows = np.isnan(z).any(axis=1) & ~nan_rows # 일부만 NaN인 바닥 행
        valid_rows = ~np.isnan(z).any(axis=1) # 온전한 행
        
        z_subpf = np.full_like(z, np.nan)
        
        if np.any(valid_rows):
            valid_z = z[valid_rows]
            coeffs = np.polyfit(x, valid_z.T, 2)
            fitted = (coeffs[0].reshape(-1,1) * (x**2) + coeffs[1].reshape(-1,1) * x + coeffs[2].reshape(-1,1))
            z_subpf[valid_rows] = valid_z - fitted
            
        for i in np.where(partial_rows)[0]:
            valid_idx = ~np.isnan(z[i])
            if np.sum(valid_idx) > 2: # 최소 3점 필요
                popt = np.polyfit(x[valid_idx], z[i][valid_idx], 2)
                fitted = popt[0]*(x**2) + popt[1]*x + popt[2]
                z_subpf[i] = z[i] - fitted
                
        return z_subpf
    
    # def differentiate (self, scan_direction):
    #     import numpy as np
    #     xrange, pixels = round(self.header['scan_range'][0] * 1e9)*1e-9, int(self.header['scan>pixels/line'])
    #     dx = xrange / pixels
    #     z = self.raw(scan_direction)
    #     z_deriv = np.zeros(np.shape(z))
    #     lines = np.shape(z)[0]
    #     for i in range(lines):
    #         z_deriv[i] = np.gradient(z[i], dx, edge_order = 2) # dI/dV curve를 직접 미분. --> d^2I/dV^2
    #     return z_deriv

    def differentiate(self, scan_direction):
        """
        스캔 데이터의 미분을 계산하는 함수
        
        Args:
            scan_direction: 스캔 방향
        
        Returns:
            numpy.ndarray: 미분된 데이터
        """        
        xrange = round(self.header['scan_range'][0] * 1e9) * 1e-9
        pixels = int(self.header['scan>pixels/line'])
        dx = xrange / pixels
        
        # 전체 배열에 대해 한 번에 gradient 계산
        z_deriv = np.gradient(self.raw(scan_direction), dx, axis=1, edge_order=2)
        
        return z_deriv
    
    def subtract_plane_fit (self, scan_direction):      
        # regular grid covering the domain of the data
        Z = self.raw(scan_direction)
        X, Y = np.meshgrid( np.arange(np.shape(Z)[1]), np.arange(np.shape(Z)[0]) )
        
        # best-fit linear plane
        A = np.c_[X.flatten(), Y.flatten(), np.ones( np.shape(Z.flatten())[0] )]
        C, _, _, _ = lstsq(A, Z.flatten())    # coefficients

        # evaluate it on grid
        plane = C[0]*X + C[1]*Y + C[2]
        return Z - plane

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
    
    # def get_map(self, scan_direction = 'fwd', channel = 'LI_Demod_1_X'):
    #     if scan_direction == 'fwd':
    #         didv = self.signals[channel]['forward']
    #     elif scan_direction == 'bwd':
    #         import numpy as np
    #         didv = np.flip(self.signals[channel]['backward'], axis = 1)
    #     return didv

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