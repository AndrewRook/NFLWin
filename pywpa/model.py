"""Tools for creating and running the model."""
from __future__ import print_function, division

from sklearn.pipeline import Pipeline

import preprocessing
import utilities

def create_pipeline(home_score_colname="home_score",
                    away_score_colname="away_score",
                    quarter_colname="quarter",
                    copy=True):
    
    steps = []

    steps.append(("check_columns", preprocessing.CheckColumnNames()))
    steps.append(("create_score_differential", preprocessing.CreateScoreDifferential(
        home_score_colname,
        away_score_colname,
        copy=copy)))

    pipe = Pipeline(steps)
    return pipe


if __name__ == "__main__":
    play_df = utilities.get_nfldb_play_data(season_years=[2015], season_types=["Postseason"])
    print(create_pipeline())
