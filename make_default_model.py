"""A simple script to create, train, validate, and save the default model"""
from __future__ import division, print_function

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

from scipy import integrate
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from numba import jit

from nflwin import data
from nflwin import model
from nflwin import preprocessing

@jit(nopython=True)
def _smooth_data(sorted_x, sorted_y, sample_x, sigma, truncation_limit=3):
    min_index = 0
    smoothed_y = np.zeros(len(sample_x))
    for i in range(len(sample_x)):
        while  min_index < len(sorted_x) and sorted_x[min_index] < sample_x[i] - sigma * truncation_limit:
            min_index += 1
        curr_index = min_index
        sum_weights = 0.
        while (curr_index < len(sorted_x)) and (sorted_x[curr_index] <= sample_x[i] + sigma * truncation_limit):
            delta_x = sample_x[i] - sorted_x[curr_index]
            weight = np.exp(-delta_x**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
            sum_weights += weight
            smoothed_y[i] += sorted_y[curr_index] * weight
            curr_index += 1
        if sum_weights < 0.000001:
            smoothed_y[i] = np.nan
        else:
            smoothed_y[i] /= sum_weights
    return smoothed_y

def smooth_probabilities(y_true, predicted_probabilities, sigma=0.005, sample_probabilities=np.linspace(0, 1, 1001)):
    probability_order = np.argsort(predicted_probabilities)
    sorted_predictions = predicted_probabilities[probability_order]
    sorted_truth = y_true[probability_order]
    
    smoothed_truth = _smooth_data(sorted_predictions, sorted_truth, sample_probabilities, sigma)
    smoothed_ideal = _smooth_data(sorted_predictions, sorted_predictions, sample_probabilities, sigma)

    return smoothed_truth, smoothed_ideal


def plot_loss_function(estimator, X, y, ax=None, n_samples=1001, sigma=0.005, **kwargs):
    func = create_loss_function(n_samples=n_samples, sigma=sigma, return_vals="eval")
    func_return = func(estimator, X, y)
    if ax is None:
        ax = plt.figure().add_subplot(111)
    kwargs["label"] = "{0}, max={1:.3f}, area={2:.3f}".format(
        kwargs.get("label", "Data"), func_return["max"], func_return["area"])
    ax.plot(func_return["samples"], func_return["smoothed_data"], **kwargs)
    ax.plot(func_return["samples"], func_return["smoothed_ideal"], ls="--", lw=2, color="black", label="Ideal")
    ax.legend(loc="upper left", fontsize=10)
    return ax
    

def create_loss_function(n_samples=1001, sigma=0.005, return_vals="area"):
    return_vals = return_vals.lower()
    if return_vals not in ("area", "max", "both", "eval"):
        raise ValueError('return_vals must be one of "area", "max", "both", or "eval"')
    def loss_function(estimator, X, y):
        predicted_probabilities = estimator.predict_proba(X)[:,1]
        samples = np.linspace(0, 1, n_samples)
        smoothed_data, smoothed_ideal = smooth_probabilities(
            y, predicted_probabilities, sigma=sigma, sample_probabilities=samples
        )
        abs_difference = np.abs(smoothed_data - smoothed_ideal)
        max_distance = np.max(abs_difference)
        area_between_curves = integrate.simps(abs_difference, samples)
        if return_vals == "area":
            return area_between_curves
        if return_vals == "max":
            return max_distance
        if return_vals == "both":
            return {"area": area_between_curves, "max": max_distance}
        if return_vals == "eval":
            return {"area": area_between_curves, "max": max_distance,
                    "samples": samples, "smoothed_data": smoothed_data,
                    "smoothed_ideal": smoothed_ideal
                    }
    return loss_function
        


def main():
    start = time.time()
    
    training_seasons = [2009, 2010, 2011, 2012, 2013, 2014, 2015]
    validation_seasons = [2016]
    season_types = ["Regular", "Postseason"]
    engine = data.connect_nfldb()
    #raw_data = data.query_nfldb(engine, training_seasons + validation_seasons, season_types)
    raw_data = pd.read_csv("test_data.csv")

    raw_data.fillna(value=0, inplace=True)

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
    training_target = (training_target == training_features["pos_team"]).values
    validation_target = (validation_target == validation_features["pos_team"]).values

    #Create a test set for model selection and hyperparameter search
    training_features, test_features, training_target, test_target = train_test_split(
        training_features, training_target, test_size=0.2, random_state=4656)


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
    steps.append(("drop_unnecessary_columns",
                  preprocessing.DropColumns(["home_team", "away_team", "pos_team"])))
    steps.append(("convert_to_numpy",
                  preprocessing.ConvertToNumpy(np.float)))

    #TODO (AndrewRook): Figure out how to handle special teams plays
    
    pipe = Pipeline(steps)
    transformed_training_features = pipe.fit_transform(training_features)
    transformed_test_features = pipe.transform(test_features)
    transformed_validation_features = pipe.transform(validation_features)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from xgboost import XGBClassifier
    #from sklearn.metrics import accuracy_score

    def run_model(train_features, train_target, test_features, test_target, classifier, param_grid,
                  fit_params=None):
        scoring_func = create_loss_function()
        grid_search = GridSearchCV(classifier, param_grid, scoring=scoring_func, fit_params=fit_params)
        grid_search.fit(train_features, train_target)
        print(pd.DataFrame(grid_search.cv_results_))
        ax = plot_loss_function(grid_search, test_features, test_target)
        plt.show()

    # print("Logistic:")
    # run_model(transformed_training_features, training_target,
    #           transformed_test_features, test_target,
    #           LogisticRegression(), {"C": [0.1, 1, 10]})
    # print("Random Forest")
    # run_model(transformed_training_features, training_target,
    #           transformed_test_features, test_target,
    #           RandomForestClassifier(n_estimators=100, min_samples_split=100))
    print("XGBoost")
    run_model(
        transformed_training_features, training_target,
        transformed_test_features, test_target,
        XGBClassifier(max_depth=4, n_estimators=100),
        {"max_depth": [3, 4, 5], "n_estimators": [1000]},
        fit_params={
            "eval_set": [(transformed_test_features, test_target)],
            "eval_metric": "logloss",
            "early_stopping_rounds": 10
        }
    )
    #clf = LogisticRegression()
    # from sklearn.ensemble import RandomForestClassifier
    # clf = RandomForestClassifier(n_estimators=100, min_samples_split=100)
    # clf.fit(transformed_training_features, training_target)
    # predictions = clf.predict(transformed_test_features)
    # print("Accuracy:", accuracy_score(test_target, predictions))
    # loss_function(test_target, clf.predict_proba(transformed_test_features)[:,1])

    #NOTE: see https://stats.stackexchange.com/a/76726 for possible things to set this to.
    # print(transformed_training_features.shape)
    # from keras.models import Sequential
    # from keras.layers import Dense, Activation
    # model = Sequential()
    # model.add(Dense(32, activation="relu", input_dim=transformed_training_features.shape[1]))
    # model.add(Dense(32, activation="relu"))
    # model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer="sgd",
    #               loss="binary_crossentropy",
    #               metrics=["accuracy"])
    # model.fit(transformed_training_features, training_target, batch_size=256, epochs=15)
    # #print(dir(model))
    # # print(probabilities.shape)
    # # print(model.evaluate(transformed_test_features, test_target, batch_size=128))
    # #test = model.predict_proba(transformed_test_features)
    # #for i,probs in enumerate(test):
    # #    print(i, probs, test_target[i])
    # print("")
    # print("")
    # loss_function(test_target, model.predict_proba(transformed_test_features)[:,0])
    #win_probability_model = model.WPModel()

if __name__ == "__main__":
    main()
