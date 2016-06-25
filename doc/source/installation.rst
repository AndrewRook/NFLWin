Installation
===============
NFLWin only supports Python 2, as nfldb is currently incompatible
with Python 3. The bulk of NFLWin should work natively with Python 3,
however that is currently untested. Pull requests ensuring this
compatibility would be welcome.


Releases
----------------------
Stable releases of NFLWin are available on PyPI::

  $ pip install nflwin

The default install provides exactly the tools necessary to make
predictions using the standard WP model as well as make new
models. However it does not include the dependencies necessary for
:ref:`using nfldb <nfldb-install>`, producing diagnostic plots, or contributing to the
package.

Installing NFLWin with those extra dependencies is accomplished by
adding a parameter in square brackets::

  $ pip install nflwin[plotting] #Adds matplotlib for plotting
  $ pip install nflwin[nfldb] #Dependencies for using nfldb
  $ pip install nflwin[dev] #Everything you need to develop on NFLWin 

.. note::
   NFLWin depends on the scipy library, which is notoriously difficult
   to install via pip or from source. One option if you're having
   difficulty getting scipy installed is to use the `Conda
   <http://conda.pydata.org/docs/>`_ package manager. After installing
   Conda, you can create a new environment and install dependencies
   manually before pip installing NFLWin::

     $ conda create -n nflwin-env python=2.7 numpy scipy scikit-learn pandas

Bleeding Edge
---------------------------
If you want the most recent stable version you can install directly
from GitHub::

  $ pip install git+https://github.com/AndrewRook/NFLWin.git@master#egg=nflwin

You can append the arguments for the extra dependencies in the same
way as for the installation from PyPI.

.. note::
   GitHub installs **do not** come with the default model. If you want
   to use a GitHub install with the default model, you'll need to
   install NFLWin from PyPI somewhere else and then copy the model
   into the model directory from your GitHub install. If you need to
   figure out where that directory is, print
   ``model.WPModel.model_directory``. 
