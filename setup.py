# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='HANG',
    version='0.1.0',
    description='Hierarchical Attention based Neural Generator for Explainable Recommendation',
    long_description=readme,
    author='You-Xiang Chen',
    author_email='60747018s@gapps.ntnu.edu.tw',
    url='https://github.com/n60512/HANG',
    license=license,
    packages=find_packages(exclude=('tests', 'HANG'))
)