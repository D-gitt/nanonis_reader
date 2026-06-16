"""
Independent image processing functions for 2D arrays.
Can be used on any 2D numpy array (STM topography, dI/dV maps, etc.).
"""
import numpy as np
import warnings
from numpy.linalg import lstsq


# ── RANSAC helper ──────────────────────────────────────────────────

def _validate_method(method):
    if method not in ('polyfit', 'RANSAC'):
        raise ValueError(f"method must be 'polyfit' or 'RANSAC', got '{method}'")

def _ransac_fit_1d(x_fit, z_fit, x_predict, degree, **ransac_kwargs):
    """
    RANSAC polynomial fit on (x_fit, z_fit), predict on x_predict.
    
    Parameters
    ----------
    x_fit, z_fit : array_like
        Training data.
    x_predict : array_like
        Points at which to predict.
    degree : int
        Polynomial degree (1=linear, 2=parabolic, ...).
    **ransac_kwargs
        Passed directly to sklearn.linear_model.RANSACRegressor.
        Common options: residual_threshold, max_trials, stop_probability,
        min_samples, random_state, loss.
    
    Returns
    -------
    np.ndarray
        Predicted values at x_predict positions.
    """
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    
    X_fit = x_fit.reshape(-1, 1)
    X_pred = x_predict.reshape(-1, 1)
    
    if degree > 1:
        estimator = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )
        ransac_kwargs.setdefault('min_samples', degree + 1)
    else:
        estimator = LinearRegression()
    
    ransac = RANSACRegressor(estimator=estimator, **ransac_kwargs)
    ransac.fit(X_fit, z_fit)
    return ransac.predict(X_pred)


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


def subtract_linear_fit(z, method='polyfit', **ransac_kwargs):
    """
    Subtract a row-wise linear (1st order polynomial) fit from a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array. Rows with partial NaN are fitted using valid points only.
        Rows that are entirely NaN remain NaN.
    method : str, optional
        'polyfit' (default) or 'RANSAC'.
    **ransac_kwargs
        Passed to RANSACRegressor when method='RANSAC'.
        Common: residual_threshold, max_trials, random_state.
    
    Returns
    -------
    np.ndarray
        Array with row-wise linear trends removed.
    """
    _validate_method(method)
    z = np.asarray(z, dtype=float)
    lines, pixels = z.shape
    x = np.arange(pixels)
    
    nan_rows = np.isnan(z).all(axis=1)
    partial_rows = np.isnan(z).any(axis=1) & ~nan_rows
    valid_rows = ~np.isnan(z).any(axis=1)
    
    result = np.full_like(z, np.nan)
    
    if method == 'polyfit':
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
    
    elif method == 'RANSAC':
        for i in np.where(~nan_rows)[0]:
            valid_idx = ~np.isnan(z[i])
            if np.sum(valid_idx) > 1:
                fitted = _ransac_fit_1d(
                    x[valid_idx], z[i][valid_idx], x, 1, **ransac_kwargs
                )
                result[i] = z[i] - fitted
    
    return result


def subtract_linear_fit_xy(z, method='polyfit', **ransac_kwargs):
    """
    Subtract linear fits in both X (row-wise) and Y (column-wise) directions.
    
    First removes row-wise linear trends (X direction),
    then removes column-wise linear trends (Y direction) from the result.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array.
    method : str, optional
        'polyfit' (default) or 'RANSAC'.
    **ransac_kwargs
        Passed to RANSACRegressor when method='RANSAC'.
    
    Returns
    -------
    np.ndarray
        Array with linear trends removed in both directions.
    """
    _validate_method(method)
    z = np.asarray(z, dtype=float)
    
    # X direction
    z_x = subtract_linear_fit(z, method, **ransac_kwargs)
    
    # Y direction: transpose → row-wise fit → transpose back
    z_xy = subtract_linear_fit(z_x.T, method, **ransac_kwargs).T
    
    return z_xy


def subtract_parabolic_fit(z, method='polyfit', **ransac_kwargs):
    """
    Subtract a row-wise parabolic (2nd order polynomial) fit from a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array. Rows with partial NaN are fitted using valid points only.
        Rows that are entirely NaN remain NaN.
    method : str, optional
        'polyfit' (default) or 'RANSAC'.
    **ransac_kwargs
        Passed to RANSACRegressor when method='RANSAC'.
    
    Returns
    -------
    np.ndarray
        Array with row-wise parabolic trends removed.
    """
    _validate_method(method)
    z = np.asarray(z, dtype=float)
    lines, pixels = z.shape
    x = np.arange(pixels)
    
    nan_rows = np.isnan(z).all(axis=1)
    partial_rows = np.isnan(z).any(axis=1) & ~nan_rows
    valid_rows = ~np.isnan(z).any(axis=1)
    
    result = np.full_like(z, np.nan)
    
    if method == 'polyfit':
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
    
    elif method == 'RANSAC':
        for i in np.where(~nan_rows)[0]:
            valid_idx = ~np.isnan(z[i])
            if np.sum(valid_idx) > 2:
                fitted = _ransac_fit_1d(
                    x[valid_idx], z[i][valid_idx], x, 2, **ransac_kwargs
                )
                result[i] = z[i] - fitted
    
    return result


def subtract_plane_fit(z, method='polyfit', **ransac_kwargs):
    """
    Subtract a best-fit plane from a 2D array.
    
    Parameters
    ----------
    z : array_like, shape (lines, pixels)
        2D input array.
    method : str, optional
        'polyfit' (default) or 'RANSAC'.
    **ransac_kwargs
        Passed to RANSACRegressor when method='RANSAC'.
    
    Returns
    -------
    np.ndarray
        Array with the planar background removed.
    """
    _validate_method(method)
    z = np.asarray(z, dtype=float)
    X, Y = np.meshgrid(np.arange(z.shape[1]), np.arange(z.shape[0]))
    
    if method == 'polyfit':
        A = np.c_[X.flatten(), Y.flatten(), np.ones(z.size)]
        C, _, _, _ = lstsq(A, z.flatten(), rcond=None)
        plane = C[0] * X + C[1] * Y + C[2]
    
    elif method == 'RANSAC':
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        
        A = np.c_[X.flatten(), Y.flatten()]
        ransac = RANSACRegressor(estimator=LinearRegression(), **ransac_kwargs)
        ransac.fit(A, z.flatten())
        plane = ransac.predict(A).reshape(z.shape)
    
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
