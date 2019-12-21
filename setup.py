import setuptools
from setuptools import setup

setup(
    name='probecon',
    version='',
    packages=setuptools.find_packages(),
    install_requires=['torch', 'gym', 'cvxopt', 'sympy', 'numpy', 'pickle', 'symbtools'],
    requires=['sympy_to_c (>=0.1.2)', 'ffmpeg'],
    package_data={'probecon.system_models.symbtoolmodels': ['*.p']},
    url='https//github.com/mpritzkoleit/...',
    author='Max Pritzkoleit',
    author_email='Max.Pritzkoleit@tu-dresden.de',
    description = 'Python library for control and reinforcement learning',
    long_description=''
)
