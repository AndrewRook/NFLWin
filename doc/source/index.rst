==================================
NFLWin
==================================

NFLWin is designed from the ground up to provide two things:

* A simple-to-use interface for users to compute Win Probabilities
  (WP) for NFL plays based on a built-in WP model.
* A robust framework for improving estimates of WP.

NFLWin builds on `scikit-learn's <http://scikit-learn.org/stable/>`_
``fit``-``transform`` idiom, allowing for pipelines that take in raw
box score data and return estimated WPs - all data
preprocessing takes place behind the scenes. Additionally,
these preprocessing steps can be easily reordered, replaced, and/or
extended, allowing for rapid iteration and prototyping of potential
improvements to the WP model.

NFLWin also has built-in support for efficiently querying data from
`nfldb <https://github.com/BurntSushi/nfldb>`_ directly into a format
useable by the built-in WP model, although the model is fully
data-source-agnostic as long as the data is formatted properly for the
model to parse.

Quickstart
---------------

NFLWin is ``pip``-installable::

  $ pip install nflwin

.. note:: NFLWin depends on `SciPy <https://www.scipy.org/>`_, which
	  is notoriously difficult to install properly via
	  ``pip``. You may wish to use the `Conda
	  <http://conda.pydata.org/docs/>`_ package manager to install
	  Scipy before installing NFLWin.

When installed via ``pip``, NFLWin comes with a working Win Probability model out-of-the-box:

.. code-block:: python

  >>> from nflwin.model import WPModel
  >>> standard_model = WPModel.load_model()

The default model can be inspected to learn what data it requires:

.. code-block:: python

  >>> standard_model.column_descriptions
  {'home_team': 'Abbreviation for the home team', 'yardline': "The yardline, given by (yards from own goalline - 50). -49 is your own 1 while 49 is the opponent's 1.", 'seconds_elapsed': 'Seconds elapsed in the quarter', 'down': 'The current down', 'curr_away_score': 'Abbreviation for the visiting team', 'offense_team': 'Abbreviation for the offensive team', 'yards_to_go': 'Yards to a first down (or the endzone)', 'quarter': 'The quarter'}

  

NFLWin operates on `Pandas <http://pandas.pydata.org/>`_ DataFrames:
  
.. code-block:: python

  >>> import pandas as pd 
  >>> plays = pd.DataFrame({
  ... "quarter": ["Q1", "Q2", "Q4"],
  ... "seconds_elapsed": [0, 0, 600],
  ... "offense_team": ["NYJ", "NYJ", "NE"],
  ... "yardline": [-20, 20, 35],
  ... "down": [1, 3, 3],
  ... "yards_to_go": [10, 2, 10],
  ... "home_team": ["NYJ", "NYJ", "NYJ"],
  ... "away_team": ["NE", "NE", "NE"],
  ... "curr_home_score": [0, 0, 21],
  ... "curr_away_score": [0, 0, 10]
  ... })

Once data is loaded, using the model to predict WP is easy:

.. code-block:: python

  >>> standard_model.predict_wp(plays)
  array([ 0.58300397,  0.64321796,  0.18195466])

Why NFLWin?
--------------
Put simply, there are no other options: while WP models have been
widely used in NFL analytics for years, the analytics community has
almost totally dropped the ball in making these models available for the
general public or even explaining their algorithms at all.

For a (much) longer explanation, see the `PhD Football blog
<http://phdfootball.blogspot.com/>`_.


Resources
--------

Default Model metrics


.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Links

   installation.rst
   model.rst
   Using nfldb <nfldb.rst>
   Developer Documentation <dev.rst>
   Full API Documentation <modules.rst>

* :ref:`Full API Documentation <modindex>`
* :ref:`Search NFLWin's Documentation <search>`

