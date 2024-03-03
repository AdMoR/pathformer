from setuptools import find_packages, setup

__version__ = "0.1"

setup(
    name='pathformer',
    packages=find_packages(exclude=['tests', 'tests.*']),
    setup_requires=[],
    version=__version__,
    description='Transformer based on vector based image',
    author='admor'
)