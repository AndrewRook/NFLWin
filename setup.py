import os
import re
import tarfile
import warnings
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

###################################################################
#Boilerplate I modified from the internet

VERSION_FILE = "nflwin/_version.py"
version_string = open(VERSION_FILE, "r").read()
version_re = r"^__version__ = [u]{0,1}['\"]([^'\"]*)['\"]"
version_match = re.search(version_re, version_string, re.M)
if version_match:
    VERSION = version_match.group(1)
else:
    raise RuntimeError("Unable to find version string in {0}".format(VERSION_FILE))

NAME = "nflwin"
PACKAGES = find_packages(where=".")
META_PATH = os.path.join(NAME, "__init__.py")
KEYWORDS = ['NFL','WP','Win Probability']
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
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

EXTRAS_REQUIRE = {
    "plotting": ["matplotlib"],
    "nfldb": ["nfldb", "sqlalchemy"],
    "dev": ["matplotlib", "nfldb", "sqlalchemy", "pytest", "pytest-cov", "sphinx", "numpydoc"]
    }

PACKAGE_DATA = {"nflwin": ["models/default_model.nflwin*"]}

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
        version=VERSION,
        author='Andrew Schechtman-Rook',
        author_email='footballastronomer@gmail.com',
        maintainer='Andrew Schechtman-Rook',
        maintainer_email='footballastronomer@gmail.com',
        keywords=KEYWORDS,
        long_description=README,
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRE
    )
