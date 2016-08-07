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
NFLWin is far from the first effort to compute Win Probabilities for
NFL plays. Brian Burke at Advanced NFL Analytics was one of the first
to popularize WP in recent years, `writing about the theory behind it
<http://www.advancedfootballanalytics.com/index.php/home/stats/stats-explained/win-probability-and-wpa>`_
as well as providing `real-time WP charts for games
<http://www.advancedfootballanalytics.com/index.php/home/tools>`_. Others
have picked up on this technique: Pro Football Reference (PFR) has their
`own model
<http://www.pro-football-reference.com/about/win_prob.htm>`_ as well
as an `interactive WP calculator
<http://www.pro-football-reference.com/play-index/win_prob.cgi?>`_,
and the technique is offered by `multiple
<https://www.numberfire.com/live>`_  analytics `startups
<http://massey-peabody.com/win-probabilities/>`_. It's one of the most
important tools in the NFL analytics toolchest.

So why create NFLWin? Well, to put it bluntly, while there are many
other analysts using WP, they're not publishing their methodologies
and algorithms or quantifying the quality of their results. This
information is critical in order to allow others both to use WP
themselves but also `to validate the correctness of the models <https://en.wikipedia.org/wiki/Peer_review>`_. Brian Burke has never discussed any of the details of
his WP model (and now that `he's at ESPN <http://www.advancedfootballanalytics.com/index.php/29-site-news/249-all-good-things>`_, that situation is unlikely to
improve any time soon), and analytics startups are (unsurprisingly)
treating their models as trade secrets. PFR goes
into more detail about their model, but it relies on an Estimated
Points model that is `not explained in sufficient detail to reproduce
it
<http://www.sports-reference.com/blog/2012/03/features-expected-points/>`_.

Possibly the best description of a WP model comes from Dennis Lock and
Dan Nettleton,
`who wrote an academic paper outline his approach
<http://nebula.wsimg.com/d376d7bbfed4109a6fbdf8c09e442161?AccessKeyId=6D2A085DACA3DAA9E4A3&disposition=0&alloworigin=1>`_. Lock
and Nettleton's
paper provides information regarding the data source used to train the
model, the type of model used, the software used to build the model,
and some statistics indicating the quality of the model. It even
includes a qualitative comparison with Brian Burke's WP estimates. This is far
and away the most complete, transparent accounting of the guts of a WP
model and is laudable. However, as often happens in academia, none of
the code used to build and test their WP model is available for others
to use; while in principle it would be possible for anyone to recreate
their model to build on or validate their work, this would require
building their entire pipeline from scratch based off of prose
descriptions. 



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

