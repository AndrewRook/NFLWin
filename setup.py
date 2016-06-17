import os
import tarfile
import warnings
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

###################################################################
#Boilerplate I modified from the internet

NAME = "nflwin"
PACKAGES = find_packages(where=".")
META_PATH = os.path.join(NAME, "__init__.py")
KEYWORDS = ['NFL','WP','Win Probability']
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Football Analytics Enthusiasts",
    "Natural Language :: English",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 2",
    "Programming Language :: Python :: 2.7",
]
INSTALL_REQUIRES = ['numpy',
                    'scipy',
                    'pandas',
                    'scikit-learn']

EXTRAS_REQUIRES = {
    "plotting": ["matplotlib"],
    "nfldb": ["nfldb", "sqlalchemy"],
    "dev": ["matplotlib", "nfldb", "sqlalchemy", "pytest", "pytest-cov"]
    }

HERE = os.path.abspath(os.path.dirname(__file__))
README = None
with open(os.path.join(HERE, 'README.rst'),'r') as f:
    README = f.read()
    
###################################################################

if __name__ == "__main__":
    setup(
        name=NAME,
        description='A Python implementation of NFL Win Probability (WP)',
        license='MIT',
        url='https://github.com/AndrewRook/NFLWin',
        version='0.1.0',
        author='Andrew Schechtman-Rook',
        author_email='footballastronomer@gmail.com',
        maintainer='Andrew Schechtman-Rook',
        maintainer_email='footballastronomer@gmail.com',
        keywords=KEYWORDS,
        long_description=README,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES
    )
