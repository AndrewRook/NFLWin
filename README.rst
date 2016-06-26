NFLWin
===============

|Build Status| |Doc Status|


Estimate Win Probability (WP) for plays in NFL games:

.. code-block:: python

  >>> import pandas as pd
  >>> from nflwin.model import WPModel
  >>> standard_model = WPModel.load_model()
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
  >>> standard_model.predict_wp(plays)
  array([ 0.58300397,  0.64321796,  0.18195466])

For full documentation, including information about methods and accuracy, click `here <http://nflwin.readthedocs.io/>`_.

License
---------------
MIT. See `license file <LICENSE>`_.

.. |Build Status| image:: https://travis-ci.org/AndrewRook/NFLWin.svg?branch=master
   :target: https://travis-ci.org/AndrewRook/NFLWin
   :alt: Build Status
.. |Doc Status| image:: https://readthedocs.org/projects/nflwin/badge/?version=latest
   :target: http://nflwin.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
