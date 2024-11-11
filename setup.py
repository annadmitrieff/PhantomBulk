#!/usr/bin/env python3
# setup.py

from setuptools import setup, find_packages

setup(
    name='PhantomBulk',
    version='1.0.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=2.2.3',
        'plotly>=5.24.1',
        'scipy>=1.14.1',
        'jinja2>=3.0.0',
        'pyyaml>=5.4.0',
        'dataclasses==0.6',
        'configparser==7.1.0',
        'packaging==24.1',
        'PyMySQL==1.1.1',
        'shutils==0.1.0',
        'six==1.16.0',
        'tenacity==9.0.0',
        'typing==3.7.4.3',
        'tzdata==2024.2'
    ],
    entry_points={
        'console_scripts': [
            'phantombulk=PhantomBulk.main:main',  
        ],
    },
    author='Anna Dmitrieff',
    author_email='annadmitrieff@uga.edu',
    description='A tool to generate protoplanetary disk simulations with hands-off parameter sweeps.',
    url='https://github.com/annadmitrieff/PhantomBulk',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Unix',
    ],
)
