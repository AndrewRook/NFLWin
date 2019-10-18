import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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
    columns_to_keep = [
        "DSEQ",
        "LEN",
        "QTR",
        "MIN",
        "SEC",
        "PTSO",
        "PTSD",
        "TIMO",
        "TIMD",
        "DWN",
        "YTG",
        "YFOG"
    ]
    return (
        non_tied_game_data[columns_to_keep]
    , target)


def create_model():
    elapsed_time_calculator = preprocessing.CalculateDerivedVariable(
        "game_fraction_elapsed", (
            "((QTR - 1) * 15 * 60 + "
            "(15 * 60 - (MIN * 60 + SEC))) / (4 * 15 * 60)"
        )
    )
    point_differential_calculator = preprocessing.CalculateDerivedVariable(
        "point_differential", "PTSO - PTSD"
    )
    numpy_converter = preprocessing.DataFrameToNumpy(
        np.float
    )
    model = LogisticRegression()
    pipe = Pipeline([
        ("create_elapsed_time", elapsed_time_calculator),
        #("create_point_differential", point_differential_calculator),
        ("numpy_shim", numpy_converter),
        ("model", model)
    ])
    return pipe


def _plot_roc_auc(target, model_scores, filename):
    auc = metrics.roc_auc_score(target, model_scores)
    print("  AUC: {0:.5f}".format(auc))
    false_positive_rate, true_positive_rate, _ = metrics.roc_curve(
        target, model_scores
    )
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], color="black", lw=2, ls="--")
    ax.plot(
        false_positive_rate, true_positive_rate,
        color="green", lw=2, ls="-",
        label="Model, AUC={0:.3f}".format(auc)
    )
    ax.legend()
    fig.savefig(filename)


def validate_model(model, features, target):
    #import pdb; pdb.set_trace()
    print(
        "y = " +
        " + ".join([
            "{0:.3f}*{1}".format(coef, column)
            for coef, column
            in zip(
                model.steps[-1][1].coef_[0,:],
                model.steps[-2][1].columns_
            )
        ])
    )
    model_scores = model.predict_proba(features)[:,1]
    _plot_roc_auc(target, model_scores, "roc_auc.png")



def export_model(model):
    pass


def make_default_model():
    np.random.seed(0)
    raw_data= load_data()
    features, target = preprocess_data(raw_data)
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2
    )
    model = create_model()
    model.fit(train_features, train_target)
    validate_model(model, test_features, test_target)
    export_model(model)
    return model, test_features, test_target


if __name__ == "__main__":
    make_default_model()