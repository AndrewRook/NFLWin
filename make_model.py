import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split

from nflwin import preprocessing


def load_data():
    data_path = "~/nflstats_pregit/armchairanalysis_data"
    games = pd.read_csv(os.path.join(data_path, "GAMES.csv"))
    core = pd.read_csv(os.path.join(data_path, "CORE.csv"))
    merged_data = core.merge(games, on="GID", how="inner")
    return merged_data

def _generate_target(home_points, away_points, off_team, home_team):
    home_won = home_points > away_points
    offense_won = (
            (home_won & (off_team == home_team)) |
            ((home_won == False) & (off_team != home_team))
    ).astype(np.int)
    return offense_won

def preprocess_data(raw_data):
    non_tied_game_data = raw_data[(
        (raw_data["PTSH"] != raw_data["PTSV"])
    )]
    target = _generate_target(
        non_tied_game_data["PTSH"],
        non_tied_game_data["PTSV"],
        non_tied_game_data["OFF"],
        non_tied_game_data["H"]
    )
    return non_tied_game_data, target


def create_model():
    elapsed_time = preprocessing.ComputeElapsedTime(
        "quarter",
        
    )


def validate_model(model, features, target):
    pass


def export_model(model):
    pass


def make_default_model():
    np.random.seed(0)
    raw_data= load_data()
    features, target = preprocess_data(raw_data)
    train_features, train_target, test_features, test_target = train_test_split(
        features, target, test_size=0.2
    )
    model = create_model()
    model.fit(train_features, train_target)
    validate_model(model, test_features, test_target)
    export_model(model)


if __name__ == "__main__":
    make_default_model()