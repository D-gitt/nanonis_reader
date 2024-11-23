from setuptools import setup, find_packages

setup(
    name='nanonis_reader',
    version='0.0.1',
    description='A Python package for reading STM experimental data files obtained from Nanonis, based on nanonispy',
    author='Dowook Kim',
    author_email='dw.kim@postech.ac.kr',
    url='https://github.com/D-gitt/nanonis_reader',
    install_requires=[
        'numpy',
        'matplotlib',  # 시각화를 위한 라이브러리
        'nanonispy',    # 필수 의존성 패키지
        'scipy',
        'warning',
        'math'
    ],
    packages=find_packages(exclude=[]),
    keywords=['nanonis', 'reader', 'nanonispy', 'STM data', 'scientific data analysis'],
    python_requires='>=3.6',
    package_data={},
    zip_safe=False,
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
