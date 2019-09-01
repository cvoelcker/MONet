#!/usr/bin/env python

from setuptools import setup

setup(name='Spatial MONet',
      version='0.1',
      description='Implementation of MONet with spatial transformers',
      author='Claas Voelcker',
      author_email='claas@voelcker.net',
      py_modules=['spatial_monet.spatial_monet', 'spatial_monet.model', 'spatial_monet.experiment_config'],
      install_requires=['numpy', 'torch', 'torchvision']
     )
