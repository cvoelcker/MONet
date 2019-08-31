#!/usr/bin/env python

from setuptools import setup

setup(name='Spatial MONet',
      version='0.1',
      description='Implementation of MONet with spatial transformers',
      author='Claas Voelcker',
      author_email='claas@voelcker.net',
      py_modules=['spatial_monet', 'model'],
      install_requires=['numpy', 'torch', 'torchvision']
     )
