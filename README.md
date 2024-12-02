# nanonis-reader package [![PyPI version](https://badge.fury.io/py/nanonis-reader.svg)](https://pypi.org/project/nanonis-reader/)

**nanonis-reader** is a Python library designed to help you analyze and visualize data with ease.  

---

## Installation

### From PyPI

You can install the latest stable release from PyPI:

```
pip install nanonis-reader
```

## Usage

### Example

When you input Nanonis file numbers, corresponding files will be automatically generated as a pptx file.

```python
import nanonis-reader as nr
path = 'your_folder_path'
ppt_maker = nr.util.DataToPPT(base_path=path, keyword='your_file_keyword', output_filename='your_file_name.pptx')
ppt_maker.generate_ppt()
```

Then, you will get below:

```terminal
Maximum file number in directory: XXXX
Enter start number (or 'q' to quit): 
```

Enter the starting file number here.

```terminal
Enter end number (or 'q' to quit): 
```

Enter the ending file number here.

```terminal
Generate PPT for files 1 to 1? (y/n): 
```

Enter 'y' to generate the PowerPoint file.
