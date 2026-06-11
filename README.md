# nanonis-reader [![PyPI version](https://badge.fury.io/py/nanonis-reader.svg)](https://pypi.org/project/nanonis-reader/)

A Python package for reading and processing Nanonis STM data files (`.sxm`, `.dat`, `.3ds`, `.nsp`).

## Installation

```bash
pip install nanonis-reader
```

### Dependencies

- `numpy`, `scipy`, `matplotlib`, `python-pptx`

---

## Quick Start

```python
import nanonis_reader as nr

d = nr.load("path/to/file.sxm")   # supports .sxm / .dat / .3ds / .nsp
```

---

## .sxm — Topography & Maps

```python
d = nr.load("file.sxm")

# Topography
z = d.topo.raw()                           # forward scan (default)
z = d.topo.raw(scan_direction='bwd')       # backward scan
z = d.topo.subtract_linear_fit()           # line-by-line linear fit subtraction
z = d.topo.subtract_linear_fit_xy()        # both x and y directions
z = d.topo.subtract_parabolic_fit()        # parabolic fit subtraction
z = d.topo.subtract_plane_fit()            # plane fit subtraction
z = d.topo.subtract_average()              # row average subtraction
z = d.topo.differentiate()                 # dz/dx

# dI/dV map
didv = d.didv.raw()                        # raw lock-in signal
didv = d.didv.subtract_linear_fit()        # with linear fit subtraction
didv = d.didv.subtract_linear_fit_xy()     # x and y

# Current map
I = d.current.raw()

# FFT
fft_img = d.fft.two_d_FFT_sqrt(z)         # sqrt(|FFT|)
fft_img = d.fft.two_d_FFT_log(z)          # log(|FFT|)
fft_img = d.fft.two_d_FFT_lin(z)          # |FFT|
```

---

## .dat — Point Spectroscopy

### STS (dI/dV vs Bias)

```python
d = nr.load("file.dat")

V, didv = d.sts.raw()                      # raw lock-in signal
V, didv = d.sts.scaled()                   # scaled by numerical derivative
V, didv = d.sts.numerical()                # numerical dI/dV from current
V, didv = d.sts.normalized(factor=0.2)     # normalized dI/dV
V, I    = d.sts.iv()                       # I-V curve
V, dzdv = d.sts.dzdv_numerical()           # numerical dZ/dV (nm/V)

# Sweep direction / index
V, didv = d.sts.scaled(sweep_direction='bwd')   # backward sweep
V, didv = d.sts.scaled(sweep_index=0)            # individual sweep
V, didv = d.sts.scaled(sweep_index='all')        # all sweeps (2D array)
```

### I-z Spectroscopy

```python
d = nr.load("iz_file.dat")

z, I = d.iz.raw()                          # I-z curve
phi, err, slope = d.iz.barrier_height()    # apparent barrier height (eV)
phi, err, slope = d.iz.barrier_height(fitting_current_range=(1e-12, 10e-12))
```

### Noise / History / Long-term

```python
freq, psd    = d.noise.get_noise()               # noise spectrum
time, data   = d.history.get_history('Z (m)')     # history data
time, z      = d.longterm.get_z_longterm_chart()  # long-term Z chart
```

---

## .3ds — Grid Spectroscopy

All methods return 3D arrays with shape `(lines, pixels, sweep_points)`.
Use numpy slicing to extract points, maps, or lines.

### STS Grid

```python
d = nr.load("file.3ds")

# Topography
topo = d.topo.subtract_linear_fit()

# 3D dI/dV
V, didv = d.sts.scaled()                  # scaled dI/dV
V, didv = d.sts.normalized(factor=0.2)    # normalized dI/dV
V, didv = d.sts.numerical()               # numerical dI/dV
V, I    = d.sts.iv()                       # I-V curves
V, didv = d.sts.raw()                      # raw lock-in signal

# Numpy slicing
spectrum = didv[line, pixel]               # single point spectrum
didv_map = didv[:, :, bias_idx]            # 2D map at specific bias
line_data = didv[line]                     # all spectra along a line
```

### I-z Grid

```python
d = nr.load("iz_grid.3ds")

Z, current  = d.iz.raw()                  # 3D I-z curves
barrier_map = d.iz.barrier_height()        # 2D apparent barrier height map (eV)
```

---

## Standalone Utilities

### Image Processing

```python
from nanonis_reader import image_processing as ip

# Apply to any 2D array
processed = ip.subtract_linear_fit(array_2d)
processed = ip.subtract_linear_fit_xy(array_2d)
processed = ip.subtract_average(array_2d)
processed = ip.subtract_parabolic_fit(array_2d)
processed = ip.subtract_plane_fit(array_2d)
processed = ip.differentiate(array_2d, dx=1.0)
```

### Spectral Analysis

```python
# Sigma filtering (remove outlier spectra)
filtered = nr.filter_sigma(spectra_2d, n_sigma=3)

# Standalone dI/dV normalization
V_n, norm = nr.normalize_didv(V, dIdV, factor=0.2)
```

### Custom Colormap

```python
from nanonis_reader.cmap_custom import bwr
plt.imshow(data, cmap=bwr())
```

---

## PPT Auto-generation

```python
ppt = nr.util.DataToPPT(
    base_path='your_folder_path',
    keyword='your_file_keyword',
    output_filename='output.pptx'
)
ppt.generate_ppt()
```
