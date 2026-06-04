import numpy as np
import re
from math import factorial


def get_channel_name(base_channel, sweep_direction='fwd', sweep_index=None):
    """
    Construct a Nanonis channel name with appropriate tags.
    
    Parameters
    ----------
    base_channel : str
        Base channel name (e.g., 'LI Demod 1 X (A)', 'Current (A)')
    sweep_direction : str
        'fwd' or 'bwd'. Default is 'fwd'.
    sweep_index : None, int, or 'all'
        None  → use [AVG] tag (caller decides whether AVG exists)
               or no tag if include_avg=False at call site.
        int   → individual sweep index (0-indexed input → [00001] 1-indexed tag)
        'all' → returns a placeholder; caller should use find_sweep_channels()
    
    Returns
    -------
    str
        Complete channel name with tags.
    
    Notes
    -----
    Tag ordering follows Nanonis convention: {channel} [AVG/NNNNN] [bwd] (unit)
    """
    # Strip unit suffix for tag insertion
    # e.g., 'LI Demod 1 X (A)' → ('LI Demod 1 X', '(A)')
    match = re.match(r'^(.*?)\s*(\([^)]*\))$', base_channel.strip())
    if match:
        channel_base = match.group(1).strip()
        unit = match.group(2)
    else:
        channel_base = base_channel
        unit = ''
    
    tags = []
    
    # Sweep index tag
    if sweep_index is not None and sweep_index != 'all':
        tags.append(f'[{sweep_index + 1:05d}]')  # 0-indexed → 1-indexed
    
    # Sweep direction tag
    if sweep_direction == 'bwd':
        tags.append('[bwd]')
    
    if tags:
        channel_name = f"{channel_base} {' '.join(tags)} {unit}".strip()
    else:
        channel_name = f"{channel_base} {unit}".strip() if unit else channel_base
    
    return channel_name


def has_averaged_data(signals):
    """
    Check if a signals dict contains averaged ([AVG]) data.
    
    Parameters
    ----------
    signals : dict
        Nanonis signals dictionary.
    
    Returns
    -------
    bool
    """
    return any('[AVG]' in key for key in signals.keys())


def find_sweep_channels(signals, base_channel, sweep_direction='fwd'):
    """
    Find all individual sweep channels matching [NNNNN] pattern, sorted by index.
    
    Parameters
    ----------
    signals : dict
        Nanonis signals dictionary.
    base_channel : str
        Base channel name (e.g., 'LI Demod 1 X (A)')
    sweep_direction : str
        'fwd' or 'bwd'
    
    Returns
    -------
    list[str]
        Sorted list of matching channel names.
    """
    match = re.match(r'^(.*?)\s*(\([^)]*\))$', base_channel.strip())
    if match:
        channel_base = re.escape(match.group(1).strip())
        unit = re.escape(match.group(2))
    else:
        channel_base = re.escape(base_channel)
        unit = ''
    
    if sweep_direction == 'bwd':
        pattern = re.compile(rf'^{channel_base}\s+\[\d{{5}}\]\s+\[bwd\]\s+{unit}$')
    else:
        # fwd: must NOT contain [bwd]
        pattern = re.compile(rf'^{channel_base}\s+\[\d{{5}}\]\s+{unit}$')
    
    matched = sorted([key for key in signals.keys() if pattern.match(key)])
    return matched


def savitzky_golay(y, window_size, order, deriv = 0, rate = 1):
    r"""
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    

    try:
        window_size = np.abs(np.int64(window_size))
        order = np.abs(np.int64(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def _1gaussian(x, A, mu, sigma):
    return np.abs(A * ( np.exp((-1.0/2.0)*(((x-mu)/sigma)**2)) ))

def _2gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    f1 = A1 * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
    f2 = A2 * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
    return np.abs(f1) + np.abs(f2)

def _3gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    f1 = A1 * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
    f2 = A2 * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
    f3 = A3 * ( np.exp((-1.0/2.0)*(((x-mu3)/sigma3)**2)) )
    return f1 + f2 + f3

def _4gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4):
    f1 = A1 * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
    f2 = A2 * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
    f3 = A3 * ( np.exp((-1.0/2.0)*(((x-mu3)/sigma3)**2)) )
    f4 = A4 * ( np.exp((-1.0/2.0)*(((x-mu4)/sigma4)**2)) )
    return f1 + f2 + f3 + f4

def _1gaussian_prob(x, A, mu, sigma): # gaussian function for probability density.
    return A * ( 1/(sigma*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu)/sigma)**2)) )
    
# def _2gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
#     import numpy as np
#     f1 = A1 * ( 1/(sigma1*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
#     f2 = A2 * ( 1/(sigma2*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
#     return f1 + f2

# def _3gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
#     import numpy as np
#     f1 = A1 * ( 1/(sigma1*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
#     f2 = A2 * ( 1/(sigma2*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
#     f3 = A3 * ( 1/(sigma3*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu3)/sigma3)**2)) )
#     return f1 + f2 + f3

# def _4gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4):
#     import numpy as np
#     f1 = A1 * ( 1/(sigma1*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
#     f2 = A2 * ( 1/(sigma2*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
#     f3 = A3 * ( 1/(sigma3*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu3)/sigma3)**2)) )
#     f4 = A4 * ( 1/(sigma4*(np.sqrt(2*np.pi))) ) * ( np.exp((-1.0/2.0)*(((x-mu4)/sigma4)**2)) )
#     return f1 + f2 + f3 + f4

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return average, np.sqrt(variance)

# linear function for kappa estimation.
def f_lin_kappa(x, kappa, b):
    return -2*kappa*x + b

# linear function for work function estimation.
def f_lin_wf(x, wf, b):
    return -2 * ( np.sqrt(2 * 0.51099895e+6 * wf) / (6.582119569e-16 * 2.99792458e+8) ) * x + b

def filter_sigma(data, n_sigma=3, axis=0, weighted=False):
    """
    Filter data within ±n_sigma of the mean along the specified axis.
    
    Parameters
    ----------
    data : array_like
        Input data.
        - 1D array: a collection of scalar values. Returns values within ±nσ.
        - 2D array: a collection of spectra (each row or column is one spectrum).
          Returns only spectra where ALL points fall within ±nσ at each position.
    n_sigma : float, optional
        Number of standard deviations for the threshold. Default is 3.
    axis : int, optional
        Axis along which spectra are stacked.
        - axis=0 (default): each row is one spectrum.
        - axis=1: each column is one spectrum.
    weighted : bool, optional
        If True, uses inverse-std weighted average and weighted standard deviation.
        Each spectrum is weighted by 1/std(spectrum), so noisier spectra
        contribute less to the reference mean and σ. Default is False.
    
    Returns
    -------
    filtered : np.ndarray
        Data containing only the entries within ±nσ.
    mask : np.ndarray of bool
        Boolean mask indicating which entries passed the filter.
        - 1D: shape (N,) — True for each value within range.
        - 2D: shape (N,) — True for each spectrum where ALL points are within range.
    
    Examples
    --------
    # 1D: filter scalar values
    >>> values = np.array([1, 2, 3, 100, 2, 3])
    >>> filtered, mask = filter_sigma(values)
    
    # 2D: filter spectra (each row = one spectrum)
    >>> spectra = np.stack([spectrum1, spectrum2, ..., spectrumN])
    >>> filtered, mask = filter_sigma(spectra, n_sigma=3, axis=0)
    
    # 2D with weighted statistics (noisy spectra contribute less)
    >>> filtered, mask = filter_sigma(spectra, n_sigma=3, weighted=True)
    """
    data = np.asarray(data, dtype=float)
    
    if data.ndim == 1:
        mean = np.nanmean(data)
        std = np.nanstd(data)
        mask = np.abs(data - mean) <= n_sigma * std
        return data[mask], mask
    
    elif data.ndim == 2:
        if axis == 1:
            data = data.T  # normalize to axis=0
        
        if weighted:
            # weight = 1 / std of each spectrum (noisy → lower weight)
            per_spectrum_std = np.nanstd(data, axis=1)  # shape: (N,)
            # avoid division by zero for constant spectra
            per_spectrum_std[per_spectrum_std == 0] = np.inf
            weights = 1.0 / per_spectrum_std               # shape: (N,)
            
            mean = np.average(data, axis=0, weights=weights)  # shape: (M,)
            std = np.sqrt(
                np.average((data - mean) ** 2, axis=0, weights=weights)
            )  # shape: (M,)
        else:
            mean = np.nanmean(data, axis=0)  # shape: (M,)
            std = np.nanstd(data, axis=0)    # shape: (M,)
        
        # check each spectrum: all points must be within ±nσ
        within = np.abs(data - mean) <= n_sigma * std  # shape: (N, M)
        mask = np.all(within, axis=1)                   # shape: (N,)
        
        filtered = data[mask]
        if axis == 1:
            filtered = filtered.T
        
        return filtered, mask
    
    else:
        raise ValueError(f"filter_sigma supports 1D and 2D arrays, got {data.ndim}D")