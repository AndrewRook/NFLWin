"""A simple script to create, train, validate, and save the default model"""
from __future__ import division, print_function

import datetime as dt
import time
import os

from nflwin import model, data

def main():
    start = time.time()
    
    training_seasons = [2009, 2010, 2011, 2012, 2013, 2014, 2015]
    validation_seasons = [2016]
    season_types = ["Regular", "Postseason"]
    engine = data.connect_nfldb()
    #raw_data = data.query_nfldb(engine, training_seasons + validation_seasons, season_types)
    import pandas as pd

    model_columns = ["home_team", "away_team",
                     "home_wins", "home_losses",
                     "away_wins", "away_losses",
                     "pos_team", "agg_home_team_points",
                     "agg_away_team_points"]

    raw_data = pd.read_csv("test_data.csv")
    training_data = raw_data[raw_data["season_year"].isin(training_seasons)][model_columns]
    validation_data = raw_data[raw_data["season_year"].isin(validation_seasons)][model_columns]
    print(training_data.head())
    
    #win_probability_model = model.WPModel()

if __name__ == "__main__":
    main()
