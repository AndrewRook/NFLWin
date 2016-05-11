"""Tools for creating and running the model."""
from __future__ import print_function, division

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

import preprocessing
import utilities

def create_pipeline(home_score_colname="curr_home_score",
                    away_score_colname="curr_away_score",
                    quarter_colname="quarter",
                    down_colname="down",
                    copy=True):
    
    steps = []

    steps.append(("check_columns", preprocessing.CheckColumnNames()))
    steps.append(("create_score_differential", preprocessing.CreateScoreDifferential(
        home_score_colname,
        away_score_colname,
        copy=copy)))
    steps.append(("map_downs_to_int", preprocessing.MapToInt(down_colname, copy=copy)))
    steps.append(("map_quarter_to_int", preprocessing.MapToInt(quarter_colname, copy=copy)))
    steps.append(("encode_categorical_columns", preprocessing.OneHotEncoderFromDataFrame(
        categorical_feature_names=[quarter_colname, down_colname],
        copy=copy)))

    pipe = Pipeline(steps)
    return pipe


if __name__ == "__main__":
    play_df = utilities.get_nfldb_play_data(season_years=[2015], season_types=["Postseason"])
    play_df.drop(["gsis_id", "drive_id", "play_id"], axis=1, inplace=True)
    pipe = create_pipeline()
    print(play_df.head())
    pipe.fit(play_df)
    play_df = pipe.transform(play_df)
    print(play_df.head())
