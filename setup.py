from setuptools import setup, find_packages

# Load the contents of the README.md file.
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='nanonis_reader',
    version='0.1.4',
    description='A Python package for reading STM experimental data files obtained from Nanonis, based on nanonispy',
    long_description=long_description,  # Set long_description
    long_description_content_type='text/markdown',  # Specify Markdown format
    author='Dowook Kim',
    author_email='dw.kim@postech.ac.kr',
    url='https://github.com/D-gitt/nanonis_reader',
    install_requires=[
        'matplotlib',
        'nanonispy',
        'numpy',
        'python-pptx',
        'scipy'
    ],
    packages=find_packages(exclude=[]),
    keywords=['nanonis', 'reader', 'nanonispy', 'STM data', 'scientific data analysis'],
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)