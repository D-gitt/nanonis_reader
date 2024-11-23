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
    
    import numpy as np
    from math import factorial

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
    import numpy as np
    return np.abs(A * ( np.exp((-1.0/2.0)*(((x-mu)/sigma)**2)) ))

def _2gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    import numpy as np
    f1 = A1 * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
    f2 = A2 * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
    return np.abs(f1) + np.abs(f2)

def _3gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3):
    import numpy as np
    f1 = A1 * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
    f2 = A2 * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
    f3 = A3 * ( np.exp((-1.0/2.0)*(((x-mu3)/sigma3)**2)) )
    return f1 + f2 + f3

def _4gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4):
    import numpy as np
    f1 = A1 * ( np.exp((-1.0/2.0)*(((x-mu1)/sigma1)**2)) )
    f2 = A2 * ( np.exp((-1.0/2.0)*(((x-mu2)/sigma2)**2)) )
    f3 = A3 * ( np.exp((-1.0/2.0)*(((x-mu3)/sigma3)**2)) )
    f4 = A4 * ( np.exp((-1.0/2.0)*(((x-mu4)/sigma4)**2)) )
    return f1 + f2 + f3 + f4

def _1gaussian_prob(x, A, mu, sigma): # gaussian function for probability density.
    import numpy as np
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
    import numpy as np
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
    import numpy as np
    return -2 * ( np.sqrt(2 * 0.51099895e+6 * wf) / (6.582119569e-16 * 2.99792458e+8) ) * x + b