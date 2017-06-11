"""A simple script to create, train, validate, and save the default model"""
from __future__ import division, print_function

import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import os

from numba import jit
from scipy import integrate
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from nflwin import data
from nflwin import model
from nflwin import preprocessing

@jit(nopython=True)
def _smooth_data(sorted_x, sorted_y, sample_x, sigma, truncation_limit=5):
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
    accuracy = accuracy_score(
        y, estimator.predict(X)
    )
    if ax is None:
        ax = plt.figure().add_subplot(111)
        ax.plot(func_return["samples"], func_return["smoothed_ideal"],
                ls="--", lw=2, color="black", label="Ideal")
        ax.set_ylabel("Actual Winning Percentage")
        ax.set_xlabel("Predicted Winning Percentage")
        ax2 = ax.twinx()
        ax2.hist(100 * estimator.predict_proba(X)[:,1], bins=100, alpha=0.25, rwidth=1, color="black")
        ax2.set_ylabel("Number of Plays")
    kwargs["label"] = "{0}, accuracy={1:.3f}, max={2:.2f}%, area={3:.3f}".format(
        kwargs.get("label", "Data"), accuracy, func_return["max"], func_return["area"])
    ax.plot(func_return["samples"], func_return["smoothed_data"], **kwargs)
    ax.legend(loc="upper left", fontsize=9)
    return ax
    

def create_loss_function(n_samples=1001, sigma=0.005, return_vals="area", inverse=False):
    return_vals = return_vals.lower()
    if return_vals not in ("area", "max", "both", "eval"):
        raise ValueError('return_vals must be one of "area", "max", "both", or "eval"')
    def loss_function(estimator, X, y):
        predicted_probabilities = estimator.predict_proba(X)[:,1]
        samples = np.linspace(0, 1, n_samples)
        smoothed_data, smoothed_ideal = smooth_probabilities(
            y, predicted_probabilities, sigma=sigma, sample_probabilities=samples
        )
        samples *= 100
        smoothed_data *= 100
        smoothed_ideal *= 100
        abs_difference = np.abs(smoothed_data - smoothed_ideal)
        max_distance = np.max(abs_difference)
        area_between_curves = integrate.simps(abs_difference, samples)
        if inverse:
            max_distance = 1. / max_distance
            area_between_curves = 1. / area_between_curves
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
        

class BinaryCalibrator(BaseEstimator):

    def __init__(self, base_estimator, calibration_order=5):
        self.base_estimator = base_estimator
        self.calibration_order = calibration_order

    def fit(self, X, y):
        probabilities = self.base_estimator.predict_proba(X)[:, 1]
        self.estimator_ = LogisticRegression(C=100)
        self.estimator_.fit(self._make_features(probabilities), y)
        return self

    def predict_proba(self, X, y=None):
        base_probabilities= self.base_estimator.predict_proba(X)[:, 1]
        return self.estimator_.predict_proba(self._make_features(base_probabilities))

    def predict(self, X, y=None):
        base_probabilities= self.base_estimator.predict_proba(X)[:, 1]
        return self.estimator_.predict(self._make_features(base_probabilities))
        

    def _make_features(self, probabilities):
        if self.calibration_order < 1:
            raise ValueError("calibration_order must be >= 1")
        if self.calibration_order == 1:
            return probabilities[:, np.newaxis]
        else:
            features = []
            for i in range(self.calibration_order):
                features.append(probabilities ** i)
            return np.vstack(features).T

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
    #transformed_validation_features = pipe.transform(validation_features)


            
    
    def run_model(train_features, train_target, test_features, test_target, classifier, param_grid,
                  fit_params=None):
        scoring_func = create_loss_function(inverse=True)
        grid_search = GridSearchCV(classifier, param_grid,
                                   fit_params=fit_params,
                                   cv=2, verbose=1)
        grid_search.fit(train_features, train_target)
        calibrated_classifier = BinaryCalibrator(grid_search.best_estimator_)
        calibrated_classifier.fit(test_features, test_target)
        return calibrated_classifier, grid_search.best_params_

    model_list = []
    # print("Random Forest:")
    # best_rforest_model, best_rforest_params = run_model(
    #     transformed_training_features, training_target,
    #     transformed_test_features, test_target,
    #     RandomForestClassifier(),
    #     {"n_estimators": [100], "min_samples_split": [100]}
    # )
    # best_rforest_accuracy = accuracy_score(
    #     test_target, best_rforest_model.predict(transformed_test_features)
    # )
    # model_list.append(
    #     {
    #         "model_name": "Random Forest",
    #         "model": best_rforest_model,
    #         "params": best_rforest_params,
    #         "accuracy": best_rforest_accuracy
    #     }
    # )
    # print("  Best model: {0}, accuracy={1:.3f}".format(best_rforest_params, best_rforest_accuracy))
    
    print("XGBoost")
    best_xgboost_model, best_xgboost_params = run_model(
        transformed_training_features, training_target,
        transformed_test_features, test_target,
        XGBClassifier(objective="binary:logistic", reg_lambda=0),
        {"max_depth": [3], "n_estimators": [100], "learning_rate": [0.1]},
        fit_params={
            "eval_set": [(transformed_test_features, test_target)],
            "eval_metric": "logloss",
            "early_stopping_rounds": 10,
            "verbose": False,
        }
    )
    best_xgboost_accuracy = accuracy_score(
        test_target, best_xgboost_model.predict(transformed_test_features)
    )
    model_list.append(
        {
            "model_name": "XGBoost",
            "model": best_xgboost_model,
            "params": best_xgboost_params,
            "accuracy": best_xgboost_accuracy
        }
    )
    print("  Best model: {0}, accuracy={1:.3f}".format(best_xgboost_params, best_xgboost_accuracy))
    

    best_model_info = model_list[0]
    for model_info in model_list:
        if model_info["accuracy"] > best_model_info["accuracy"]:
            best_model_info = model_info
    print("Best model overall: type: {0}, parameters {1}, accuracy={2:3f}".format(
        best_model_info["model_name"], best_model_info["params"], best_model_info["accuracy"]
    ))
    ax = plot_loss_function(
        best_model_info["model"], transformed_test_features, test_target,
        color="blue",
        label=best_model_info["model_name"]
    )
    ax.set_xlim(0, 100)
    plt.tight_layout()
    plt.show()

    #TODO (ASR):
    #1 connect best model to transforms
    #2 ensure model scoring works properly for entire pipeline
    #3 save plot to disk
    #4 save model to disk
    #5 use validation data instead of test data to make plot

if __name__ == "__main__":
    main()
