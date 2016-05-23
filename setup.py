import os
import tarfile
import warnings
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

###################################################################
#Boilerplate I modified from the internet

NAME = "PyWPA"
PACKAGES = find_packages(where=".")
META_PATH = os.path.join(NAME, "__init__.py")
KEYWORDS = ['NFL','WPA','Win Probability Added']
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
INSTALL_REQUIRES = ['nfldb',
                    'numpy',
                    'scipy',
                    'statsmodels',
                    'pandas',
                    'scikit-learn']

HERE = os.path.abspath(os.path.dirname(__file__))
README = None
with open(os.path.join(HERE, 'README.rst'),'r') as f:
    README = f.read()
    
###################################################################

class install(_install):
    def run(self):
        #Run the regular install:
        _install.run(self)

        #Get the installation directory:
        INSTALL_DIR = os.path.join(self.install_lib,NAME)
        #Unzip files:

        #Data tarball:
        data_tarball = os.path.join(INSTALL_DIR,'data','nfldb_processed_data.tar.gz')
        if os.path.exists(data_tarball):
            #Only unzip the files if the path exists:
            with tarfile.open(data_tarball) as tarball:
                tarball.extractall(path=os.path.dirname(data_tarball))
        else:
            warnings.warn("Could not find data tarball!")

        #Saved model tarball:
        model_tarball = os.path.join(INSTALL_DIR,'models','PyWPA_model.tar.gz')
        if os.path.exists(model_tarball):
            #Only unzip the files if the path exists:
            with tarfile.open(model_tarball) as tarball:
                tarball.extractall(path=os.path.dirname(model_tarball))
        else:
            warnings.warn("Could not find model tarball!")

if __name__ == "__main__":
    setup(
        name=NAME,
        description='A Python implementation of NFL Win Probability Added (WPA)',
        license='MIT',
        url='https://github.com/AndrewRook/PyWPA',
        version='0.1.0',
        author='Andrew Schechtman-Rook',
        author_email='footballastronomer@gmail.com',
        maintainer='Andrew Schechtman-Rook',
        maintainer_email='footballastronomer@gmail.com',
        keywords=KEYWORDS,
        long_description=README,
        packages=PACKAGES,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        include_package_data=True,
        cmdclass={'install': install},
    )
