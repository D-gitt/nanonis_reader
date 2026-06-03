"""
Independent image processing functions for 2D arrays.
Can be used on any 2D numpy array (STM topography, dI/dV maps, etc.).
"""
import numpy as np
import warnings
from numpy.linalg import lstsq


def subtract_average(z):
    """
    Subtract the row-wise average from a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array.
    
    Returns
    -------
    np.ndarray
        Array with row averages subtracted. NaN-safe.
    """
    z = np.asarray(z, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return z - np.nanmean(z, axis=1, keepdims=True)


def subtract_linear_fit(z):
    """
    Subtract a row-wise linear (1st order polynomial) fit from a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array. Rows with partial NaN are fitted using valid points only.
        Rows that are entirely NaN remain NaN.
    
    Returns
    -------
    np.ndarray
        Array with row-wise linear trends removed.
    """
    z = np.asarray(z, dtype=float)
    lines, pixels = z.shape
    x = np.arange(pixels)
    
    nan_rows = np.isnan(z).all(axis=1)
    partial_rows = np.isnan(z).any(axis=1) & ~nan_rows
    valid_rows = ~np.isnan(z).any(axis=1)
    
    result = np.full_like(z, np.nan)
    
    if np.any(valid_rows):
        valid_z = z[valid_rows]
        coeffs = np.polyfit(x, valid_z.T, 1)
        fitted = coeffs[0].reshape(-1, 1) * x + coeffs[1].reshape(-1, 1)
        result[valid_rows] = valid_z - fitted
    
    for i in np.where(partial_rows)[0]:
        valid_idx = ~np.isnan(z[i])
        if np.sum(valid_idx) > 1:
            popt = np.polyfit(x[valid_idx], z[i][valid_idx], 1)
            fitted = popt[0] * x + popt[1]
            result[i] = z[i] - fitted
    
    return result


def subtract_linear_fit_xy(z):
    """
    Subtract linear fits in both X (row-wise) and Y (column-wise) directions.
    
    First removes row-wise linear trends (X direction),
    then removes column-wise linear trends (Y direction) from the result.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array.
    
    Returns
    -------
    np.ndarray
        Array with linear trends removed in both directions.
    """
    z = np.asarray(z, dtype=float)
    
    # X direction
    z_x = subtract_linear_fit(z)
    
    # Y direction
    lines, pixels = z_x.shape
    y = np.arange(lines)
    
    nan_cols = np.isnan(z_x).all(axis=0)
    partial_cols = np.isnan(z_x).any(axis=0) & ~nan_cols
    valid_cols = ~np.isnan(z_x).any(axis=0)
    
    result = np.full_like(z_x, np.nan)
    
    if np.any(valid_cols):
        valid_z = z_x[:, valid_cols]
        coeffs = np.polyfit(y, valid_z, 1)
        fitted = coeffs[0] * y.reshape(-1, 1) + coeffs[1]
        result[:, valid_cols] = valid_z - fitted
    
    for j in np.where(partial_cols)[0]:
        valid_idx = ~np.isnan(z_x[:, j])
        if np.sum(valid_idx) > 1:
            popt = np.polyfit(y[valid_idx], z_x[valid_idx, j], 1)
            fitted = popt[0] * y + popt[1]
            result[:, j] = z_x[:, j] - fitted
    
    return result


def subtract_parabolic_fit(z):
    """
    Subtract a row-wise parabolic (2nd order polynomial) fit from a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array. Rows with partial NaN are fitted using valid points only.
        Rows that are entirely NaN remain NaN.
    
    Returns
    -------
    np.ndarray
        Array with row-wise parabolic trends removed.
    """
    z = np.asarray(z, dtype=float)
    lines, pixels = z.shape
    x = np.arange(pixels)
    
    nan_rows = np.isnan(z).all(axis=1)
    partial_rows = np.isnan(z).any(axis=1) & ~nan_rows
    valid_rows = ~np.isnan(z).any(axis=1)
    
    result = np.full_like(z, np.nan)
    
    if np.any(valid_rows):
        valid_z = z[valid_rows]
        coeffs = np.polyfit(x, valid_z.T, 2)
        fitted = (coeffs[0].reshape(-1, 1) * (x ** 2) +
                  coeffs[1].reshape(-1, 1) * x +
                  coeffs[2].reshape(-1, 1))
        result[valid_rows] = valid_z - fitted
    
    for i in np.where(partial_rows)[0]:
        valid_idx = ~np.isnan(z[i])
        if np.sum(valid_idx) > 2:
            popt = np.polyfit(x[valid_idx], z[i][valid_idx], 2)
            fitted = popt[0] * (x ** 2) + popt[1] * x + popt[2]
            result[i] = z[i] - fitted
    
    return result


def subtract_plane_fit(z):
    """
    Subtract a best-fit plane from a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array.
    
    Returns
    -------
    np.ndarray
        Array with the planar background removed.
    """
    z = np.asarray(z, dtype=float)
    X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    
    A = np.c_[X.flatten(), Y.flatten(), np.ones(z.size)]
    C, _, _, _ = lstsq(A, z.flatten(), rcond=None)
    
    plane = C[0] * X + C[1] * Y + C[2]
    return z - plane


def differentiate(z, dx=1.0):
    """
    Compute the row-wise numerical derivative of a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array.
    dx : float, optional
        Spacing between pixels. Default is 1.0 (derivative in pixel units).
    
    Returns
    -------
    np.ndarray
        Row-wise gradient of the input array.
    """
    z = np.asarray(z, dtype=float)
    return np.gradient(z, dx, axis=1, edge_order=2)
