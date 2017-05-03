"""A simple script to create, train, validate, and save the default model"""
from __future__ import division, print_function

import datetime as dt
import numpy as np
import time
import os

#from sklearn.base import BaseEstimator
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
    training_target = raw_data[raw_data["season_year"].isin(training_seasons)]["winning_team"]
    validation_target = raw_data[raw_data["season_year"].isin(validation_seasons)]["winning_team"]


    training_features = raw_data[raw_data["season_year"].isin(training_seasons)][model_columns]
    validation_features = raw_data[raw_data["season_year"].isin(validation_seasons)][model_columns]
    
    #Convert target to a boolean based on whether the offense won:
    training_target = (training_target == training_features["pos_team"])
    validation_target = (validation_target == validation_features["pos_team"])


    steps = []
    steps.append(("compute_offense_home",
                  preprocessing.ComputeIfOffenseIsHome("pos_team", "home_team",
                                                     offense_home_team_colname="is_offense_home")
                ))
    steps.append(("compute_offense_defense_wins",
                  preprocessing.ConvertToOffenseDefense("home_wins", "away_wins",
                                                        "is_offense_home", "offense_wins",
                                                        "defense_wins")))
    steps.append(("compute_offense_defense_losses",
                  preprocessing.ConvertToOffenseDefense("home_losses", "away_losses",
                                                        "is_offense_home", "offense_losses",
                                                        "defense_losses")))
    steps.append(("compute_offense_defense_score",
                  preprocessing.ConvertToOffenseDefense("current_home_score", "current_away_score",
                                                        "is_offense_home", "offense_score",
                                                        "defense_score")))
    steps.append(("map_quarters_to_ints",
                  preprocessing.MapToInt("quarter")))
    steps.append(("encode_quarters",
                  preprocessing.OneHotEncoderFromDataFrame(categorical_feature_names=["quarter"])))
    
    #TODO (AndrewRook): Add DropColumns class to drop columns by name
    
    pipe = Pipeline(steps)
    transformed_training_features = pipe.fit_transform(training_features)
    transformed_training_features.drop(["home_team", "away_team", "pos_team"], axis=1, inplace=True)
    transformed_training_features = transformed_training_features.sort_index(axis=1).values.astype(np.float)
    print(transformed_training_features.shape)
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    model = Sequential()
    model.add(Dense(32, activation="relu", input_dim=transformed_training_features.shape[1]))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer="sgd",
                  loss="mean_squared_error",
                  metrics=["accuracy"])
    model.fit(transformed_training_features, training_target, epochs=10)
    print(model.evaluate(transformed_training_features, training_target, batch_size=128))
    #win_probability_model = model.WPModel()

if __name__ == "__main__":
    main()
