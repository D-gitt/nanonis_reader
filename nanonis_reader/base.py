import os
from . import _vendor as nap  # vendored nanonispy (MIT License)
from . import nsp

class load:
    """Base class for handling all Nanonis data files."""
    
    EXTENSION_READERS = {
        'sxm': nap.read.Scan,
        'dat': nap.read.Spec,
        '3ds': nap.read.Grid,
        'nsp': nsp.Nsp
    }
    
    def __init__(self, filepath):
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        self.fname = os.path.basename(filepath)
        self.extension = self.fname.split('.')[-1].lower()
        
        if self.extension not in self.EXTENSION_READERS:
            raise ValueError(f"Unsupported file extension: {self.extension}")
        
        reader_class = self.EXTENSION_READERS[self.extension]
        reader = reader_class(filepath)
        
        self.header = reader.header
        self.signals = reader.signals

    # ── Lazy properties ─────────────────────────────────────────────
    # These allow  d.topo.get_z(...)  instead of  nr.sxm.topography(d).get_z(...)

    def _require_ext(self, name, *exts):
        if self.extension not in exts:
            raise AttributeError(
                f".{name} is not available for .{self.extension} files "
                f"(requires {', '.join('.' + e for e in exts)})"
            )

    # --- shared across .sxm / .3ds ---

    @property
    def topo(self):
        """Topography processor (available for .sxm and .3ds)."""
        if self.extension == 'sxm':
            from . import sxm
            return sxm.topography(self)
        if self.extension == '3ds':
            from . import grid
            return grid.topography(self)
        self._require_ext('topo', 'sxm', '3ds')

    @property
    def didv(self):
        """dI/dV map processor (available for .sxm and .3ds)."""
        if self.extension == 'sxm':
            from . import sxm
            return sxm.didvmap(self)
        if self.extension == '3ds':
            from . import grid
            return grid.didvmap(self)
        self._require_ext('didv', 'sxm', '3ds')

    # --- .sxm only ---

    @property
    def current(self):
        """Current map processor (available for .sxm)."""
        self._require_ext('current', 'sxm')
        from . import sxm
        return sxm.currentmap(self)

    @property
    def fft(self):
        """FFT processor (available for .sxm)."""
        self._require_ext('fft', 'sxm')
        from . import sxm
        return sxm.fft(self)

    # --- .dat only ---

    @property
    def spec(self):
        """Point spectrum processor (available for .dat)."""
        self._require_ext('spec', 'dat')
        from . import dat
        return dat.spectrum(self)

    @property
    def z_spec(self):
        """Z-spectroscopy processor (available for .dat)."""
        self._require_ext('z_spec', 'dat')
        from . import dat
        return dat.z_spectrum(self)

    @property
    def noise(self):
        """Noise spectrum processor (available for .dat)."""
        self._require_ext('noise', 'dat')
        from . import dat
        return dat.noise_spectrum(self)

    @property
    def history(self):
        """History data processor (available for .dat)."""
        self._require_ext('history', 'dat')
        from . import dat
        return dat.history_data(self)

    @property
    def longterm(self):
        """Long-term data processor (available for .dat)."""
        self._require_ext('longterm', 'dat')
        from . import dat
        return dat.longterm_data(self)

    # --- .3ds: new unified API ---

    @property
    def sts(self):
        """STS grid spectroscopy (available for .3ds).
        
        Returns full 3D arrays. Use numpy slicing for point/map/line.
        """
        self._require_ext('sts', '3ds')
        from . import grid
        return grid.sts(self)

    @property
    def iz(self):
        """I-z grid spectroscopy (available for .3ds).
        
        Returns full 3D arrays. Use numpy slicing for point/map.
        """
        self._require_ext('iz', '3ds')
        from . import grid
        return grid.iz(self)

    # --- .3ds: deprecated wrappers (backward compatibility) ---

    @property
    def point(self):
        """[Deprecated] Use d.sts instead. Point dI/dV processor (.3ds)."""
        self._require_ext('point', '3ds')
        from . import grid
        return grid.point_didv(self)

    @property
    def point_iz(self):
        """[Deprecated] Use d.iz instead. Point I-z processor (.3ds)."""
        self._require_ext('point_iz', '3ds')
        from . import grid
        return grid.point_iz(self)

    @property
    def linespec(self):
        """[Deprecated] Use d.sts instead. Line spectrum processor (.3ds)."""
        self._require_ext('linespec', '3ds')
        from . import grid
        return grid.line_spectrum(self)

    # --- .nsp only ---

    @property
    def ltspec(self):
        """Long-term spectrum processor (available for .nsp)."""
        self._require_ext('ltspec', 'nsp')
        from . import nsp as nsp_mod
        return nsp_mod.ltspec(self)