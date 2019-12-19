from setuptools import setup

setup(
    name='Probabilistic Ennsembles for Control',
    version='0.15',
    packages=['probecon', 'probecon.system_models','probecon.helpers','probecon.system_models.symbtoolmodels'],#packages=['pygent', 'pygent/algorithms', 'pygent/modeling_scripts','pygent/modeling_scripts/c_files'],
    install_requires=['torch', 'gym', 'cvxopt'],
    requires=['sympy_to_c (>=0.1.2)', 'ffmpeg'],
    package_data={'probecon.system_models.symbtoolmodels': ['*.p']},
    url='https//github.com/mpritzkoleit/...',
    author='Max Pritzkoleit',
    author_email='Max.Pritzkoleit@tu-dresden.de',
    description = 'Python library for control and reinforcement learning',
    long_description=''
)
