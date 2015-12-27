import os
from setuptools import setup, find_packages

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
                    'pandas',
                    'scikit-learn']

HERE = os.path.abspath(os.path.dirname(__file__))
README = None
with open(os.path.join(HERE, 'README.rst'),'r') as f:
    README = f.read()
    
###################################################################
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
        #package_dir={"": "src"},
        #zip_safe=False,
        classifiers=CLASSIFIERS,
        install_requires=INSTALL_REQUIRES,
        # package_data = {
        #     'data': ['data/'],
        #     'model': ['model/'],
        #     },
        include_package_data=True,
    )
