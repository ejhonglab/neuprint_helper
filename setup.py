#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    # TODO legal to use dirname (or last part if need be) here?
    # why not default to that?
    name='neuprint_helper',
    version='0.0.1',
    install_requires=['numpy', 'pandas', 'neuprint-python'],
    packages=find_packages(),
    scripts=['scripts/write_neuprint_csvs.py'],
)
