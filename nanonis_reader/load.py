import os
import nanonispy as nap

class load:
    """Base class for handling all Nanonis data files."""
    
    EXTENSION_READERS = {
        'sxm': nap.read.Scan,
        'dat': nap.read.Spec,
        '3ds': nap.read.Grid
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