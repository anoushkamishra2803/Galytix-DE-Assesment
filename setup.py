from setuptools import setup, find_packages

setup(
    name='Galytix',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'gensim',
        'scipy',
        'pandas',
    ],
)
