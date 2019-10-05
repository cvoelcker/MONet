#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='Spatial MONet',
      version='0.2',
      description='Implementation of MONet with spatial transformers',
      author='Claas Voelcker',
      author_email='claas@voelcker.net',
      packages=find_packages(),
      install_requires=['numpy', 'torch', 'torchvision']
     )
