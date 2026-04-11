import numpy as np

def per_sqrt_hz(name):
    """
    Extract unit and convert to include /sqrt(Hz).
    
    Parameters
    ----------
    name : str
        Signal name with unit, e.g., 'Current (A)'
    
    Returns
    -------
    str
        Unit string only, e.g., '(A/sqrt(Hz))'
    """
    start = name.rfind('(')
    end = name.rfind(')')
    
    if start != -1 and end != -1:
        unit = name[start+1:end]
        return f'({unit}/sqrt(Hz))'
    return name


class Nsp:
    """
    Nanonis Noise Spectrum (.nsp) file parser.
    Mimics the interface of nanonispy (nap) classes like nap.read.Scan
    so that it can be seamlessly integrated into nanonis_reader's logic.
    """
    def __init__(self, fname):
        self.fname = fname
        self.header = dict()
        self.signals = dict()
        self._parse()

    def _parse(self):
        """
        Main method to find header bounds, read the raw strings,
        parse into header dict, and finally extract the binary float data.
        """
        byte_offset = self._start_byte()
        header_raw = self._read_raw_header(byte_offset)
        self.header = self._parse_nsp_header(header_raw)
        self.signals = self._load_data(byte_offset)

    def _start_byte(self, tag=':HEADER_END:'):
        """
        Finds the byte offset marking the end of the ASCII header section.
        """
        with open(self.fname, 'rb') as f:
            byte_offset = -1
            
            for line in f:
                # Convert from bytes to str
                try:
                    entry = line.strip().decode()
                except UnicodeDecodeError:
                    entry = line.strip().decode('utf-8', errors='replace')
                if tag in entry:
                    byte_offset = f.tell()
                    break
            
            if byte_offset == -1:
                return 0
        
        return byte_offset

    def _read_raw_header(self, byte_offset):
        """
        Extract the literal string contents of the header section.
        """
        with open(self.fname, 'rb') as f:
            header_raw = f.read(byte_offset)
        return header_raw.decode('utf-8', errors='replace')

    def _parse_nsp_header(self, header_raw):
        """
        Parse header lines and transform them into a dictionary.
        """
        header_entries = header_raw.split('\n')
        # Ensure we drop at least the trailing bits properly
        if len(header_entries) >= 2:
            header_entries = header_entries[:-2] 
        
        header_dict = dict()
        for i, entry in enumerate(header_entries):
            if entry.startswith(':') and entry.endswith(':'):
                # We found a key, the value must be the next line
                key = entry.strip(':')
                if i + 1 < len(header_entries):
                    val = header_entries[i + 1].strip()
                    header_dict[key] = val
        
        return header_dict

    def _load_data(self, byte_offset):
        """
        Locate the binary spectrum data block, read as Float32 (>f4),
        and convert to a 2D numpy array storing it in a dictionary
        indexed by the SIGNAL key found in the header.
        """
        data_type = self.header.get('DATA_TYPE', 'FLOAT MSBFIRST')
        signal_name = self.header.get('SIGNAL', 'UnknownSignal')
        nrows = int(self.header.get('DATASIZEROWS', 1))
        ncols = int(self.header.get('DATASIZECOLS', 1))
        
        if 'FLOAT' in data_type and 'MSBFIRST' in data_type:
            data_format = '>f4'
        else:
            # Fallback for now if something unexpected occurs
            data_format = '>f4'
            
        data_dict = dict()
        
        # Nanonis binary block traditionally begins with a 2-byte separator code.
        with open(self.fname, 'rb') as f:
            f.seek(byte_offset + 2)
            specdata = np.fromfile(f, dtype=data_format)
            
        expected_size = nrows * ncols
        if len(specdata) == expected_size:
            specdata = specdata.reshape(nrows, ncols)
        elif len(specdata) > expected_size:
            specdata = specdata[:expected_size].reshape((nrows, ncols))
            
        data_dict[signal_name] = specdata
        return data_dict

class ltspec:
    """
    Data extractor for nsp files.
    """
    def __init__(self, instance):
        self.fname = instance.fname
        self.header = instance.header
        self.signals = instance.signals
        
    def get(self):
        """
        Returns the data array transposed, using the first available signal.
        """
        if not self.signals:
            raise ValueError(f"No signals found in {self.fname}")
        first_key = list(self.signals.keys())[0]
        return self.signals[first_key].T
