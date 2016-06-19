.. NFLWin documentation master file, created by
   sphinx-quickstart on Thu Jun 16 22:35:58 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NFLWin: Open Source NFL Win Probabilities
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
`nfldb <https://github.com/BurntSushi/nfldb`_ directly into a format
useable by the built-in WP model, although the model is fully
data-source-agnostic as long as the data is formatted properly for the
model to parse.

.. toctree::
   :maxdepth: 2



Resources
==================

* :ref:`genindex`
* :ref:`Full API Documentation <modindex>`
* :ref:`Search NFLWin's Documentation <search>`

