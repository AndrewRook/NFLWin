Creating a New WP Model
==============================
While NFLWin ships with a fairly robust default model, there is always
room for improvement. Maybe there's a new dataset you want to use to
train the model, a new feature you want to add, or a new machine
learning model you want to evaluate.

Good news! NFLWin makes it easy to train a new model, whether you just
want to refresh the data or to do an entire refit from scratch. We'll
start with the simplest case:

Default Model, New Data
-----------------------
Refreshing the data with NFLWin is a snap. If you want to change the
data used by the default model but keep the source as nfldb, all you
have to do is override the default keyword arguments when calling the
:meth:`~nflwin.model.WPModel.train_model` and :meth:`~nflwin.model.WPModel.validate_model`
methods. For instance, if for some insane reason you wanted to train on the 2009 and 2010 regular
seasons and validate on the 2011 and 2012 playoffs, you would do the following:
  
.. code-block:: python

  >>> from nflwin.model import WPModel
  >>> new_data_model = WPModel()
  >>> new_data_model.train_model(training_seasons=[2009, 2010], training_season_types=["Regular"])
  >>> new_data_model.validate_model(validation_seasons=[2011, 2012], validation_season_types=["Postseason"])
  0.14963235412213988

If you want to supply your own data, that's easy too - simply set the
`source_data` kwarg of :meth:`~nflwin.model.WPModel.train_model` and
:meth:`~nflwin.model.WPModel.validate_model` to be a Pandas DataFrame of your training and validation data (respectively):

.. code-block:: python

  >>> from nflwin.model import WPModel
  >>> new_data_model = WPModel()
  >>> training_data.head()
        gsis_id  drive_id  play_id offense_team  yardline  down  yards_to_go  \
  0  2012090500         1       35          DAL     -15.0     0            0   
  1  2012090500         1       57          NYG     -34.0     1           10   
  2  2012090500         1       79          NYG     -34.0     2           10   
  3  2012090500         1      103          NYG     -29.0     3            5   
  4  2012090500         1      125          NYG     -29.0     4            5   
  
    home_team away_team offense_won quarter  seconds_elapsed  curr_home_score  \
  0       NYG       DAL        True      Q1              0.0                0   
  1       NYG       DAL       False      Q1              4.0                0   
  2       NYG       DAL       False      Q1             11.0                0   
  3       NYG       DAL       False      Q1             55.0                0   
  4       NYG       DAL       False      Q1             62.0                0   
  
     curr_away_score  
  0                0  
  1                0  
  2                0  
  3                0  
  4                0 
  
