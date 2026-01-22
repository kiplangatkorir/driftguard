from setuptools import setup, find_packages

setup(
    name='driftguard',
    version='0.1.6',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
    ],
)