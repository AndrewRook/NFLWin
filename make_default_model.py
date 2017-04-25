"""A simple script to create, train, validate, and save the default model"""
from __future__ import division, print_function

import datetime as dt
import time
import os

from sklearn.pipeline import Pipeline

from nflwin import data
from nflwin import model
from nflwin import preprocessing

def main():
    start = time.time()
    
    training_seasons = [2009, 2010, 2011, 2012, 2013, 2014, 2015]
    validation_seasons = [2016]
    season_types = ["Regular", "Postseason"]
    engine = data.connect_nfldb()
    #raw_data = data.query_nfldb(engine, training_seasons + validation_seasons, season_types)
    import pandas as pd
    raw_data = pd.read_csv("test_data.csv")

    model_columns = ["home_team", "away_team",
                     "home_wins", "home_losses",
                     "away_wins", "away_losses",
                     "pos_team", "yardline",
                     "down", "yards_to_go",
                     "quarter", "seconds_elapsed",
                     "current_home_score",
                     "current_away_score"]

    training_data = raw_data[raw_data["season_year"].isin(training_seasons)][model_columns]
    validation_data = raw_data[raw_data["season_year"].isin(validation_seasons)][model_columns]
    print(training_data.head())

    steps = []
    steps.append(("compute_offense_home",
                  preprocessing.ComputeIfOffenseIsHome("pos_team", "home_team",
                                                     offense_home_team_colname="is_offense_home")
                ))
    
    #TODO (AndrewRook): Add DropColumns class to drop columns by name

    #TODO (AndrewRook): Add ConvertToOffenseDefense class to take pairs of (home, away) fields
    #and convert to (offense, defense).

    #TODO (AndrewRook) Add ConvertWinningTeamToOffense class to convert the winning team
    #to a boolean did home team win yes/no field
    
    pipe = Pipeline(steps)
    pipe.fit_transform(training_data)
    
    #win_probability_model = model.WPModel()

if __name__ == "__main__":
    main()
