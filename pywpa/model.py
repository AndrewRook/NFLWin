"""Tools for creating and running the model."""
from __future__ import print_function, division

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import brier_score_loss
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import NotFittedError

import preprocessing
import utilities

class WPModel(object):
    """The object that computes win probabilities.


    Parameters
    ----------
    home_score_colname : string (default="curr_home_score")
        The name of the column containing the current home score at the start of a play.
    away_score_colname : string (default="curr_away_score")
        The name of the column containing the current away score at the start of a play.
    quarter_colname : string (default="quarter")
        The name of the column containing the quarter the play took place in.
    time_colname : string (default="seconds_elapsed")
        The name of the column containing the time elapsed (in seconds) from the start
        of the quarter when the play began.
    down_colname : string (default="down")
        The name of the column containing the current down number, with zeros for plays like
        kickoffs and extra points.
    yards_to_go_colname : string (default="yards_to_go")
        The name of the column containing the number of yards to go in order to get a first down.
    offense_team_colname : string (default="offense_team")
        The name of the column containing the abbreviation for the team currently on offense.
    home_team_colname : string (default="home_team")
        The name of the column containing the abbreviation for the home team.
    copy_data : boolean (default=``True``)
        Whether or not to copy data when fitting and applying the model. Running the model
        in-place (``copy_data=False``) will be faster and have a smaller memory footprint,
        but if not done carefully can lead to data integrity issues.

    Attributes
    ----------
    model : A Scikit-learn pipeline (or equivalent)
        The actual model used to compute WP. Upon initialization it will be set to
        a default model, but can be overridden by the user.

    """

    def __init__(self,
                 home_score_colname="curr_home_score",
                 away_score_colname="curr_away_score",
                 quarter_colname="quarter",
                 time_colname = "seconds_elapsed",
                 down_colname="down",
                 yards_to_go_colname="yards_to_go",
                 yardline_colname="yardline",
                 offense_team_colname="offense_team",
                 home_team_colname="home_team",
                 copy_data=True
                ):
        self.home_score_colname = home_score_colname
        self.away_score_colname = away_score_colname
        self.quarter_colname = quarter_colname
        self.time_colname = time_colname
        self.down_colname = down_colname
        self.yards_to_go_colname = yards_to_go_colname
        self.yardline_colname = yardline_colname
        self.offense_team_colname = offense_team_colname
        self.home_team_colname = home_team_colname
        self.copy_data = copy_data

        self.model = None


    def create_default_pipeline(self):
        """Create the default win probability estimation pipeline.

        This sets the ``model`` attribute of the class, so be aware that
        if you run it it will overwrite any model you have stored already.
        """

        steps = []

        is_offense_home = preprocessing.ComputeIfOffenseIsHome(self.offense_team_colname,
                                                               self.home_team_colname,
                                                               copy=self.copy_data)
        steps.append(("compute_offense_home", is_offense_home))
        score_differential = preprocessing.CreateScoreDifferential(self.home_score_colname,
                                                                   self.away_score_colname,
                                                                   is_offense_home.offense_home_team_colname,
                                                                   copy=self.copy_data)
        steps.append(("create_score_differential", score_differential))
        steps.append(("map_downs_to_int", preprocessing.MapToInt(self.down_colname, copy=self.copy_data)))
        total_time_elapsed = preprocessing.ComputeElapsedTime(self.quarter_colname, self.time_colname, copy=self.copy_data)
        steps.append(("compute_total_time_elapsed", total_time_elapsed))
        steps.append(("remove_unnecessary_columns", preprocessing.CheckColumnNames(
            column_names=[is_offense_home.offense_home_team_colname,
                          score_differential.score_differential_colname,
                          total_time_elapsed.total_time_colname,
                          self.yardline_colname,
                          self.yards_to_go_colname,
                          self.down_colname],
            copy=self.copy_data)))
        steps.append(("encode_categorical_columns", preprocessing.OneHotEncoderFromDataFrame(
            categorical_feature_names=[self.down_colname],
            copy=self.copy_data)))

        model = self.model_class(**self.model_kwargs)
        model = CalibratedClassifierCV(model, cv=2, method="isotonic")
        if self.parameter_search_grid is not None:
            model = GridSearchCV(model, self.parameter_search_grid,
                                 scoring=self._brier_loss_scorer)
        else:
            model = model
        steps.append(("compute_model", model))

        pipe = Pipeline(steps)
        self.model = pipe

    @staticmethod
    def _brier_loss_scorer(estimator, X, y):
        """Use the Brier loss to estimate model score.

        For use in GridSearchCV, instead of accuracy.
        """
        predicted_positive_probabilities = estimator.predict_proba(X)[:, 1]
        return 1. - brier_score_loss(y, predicted_positive_probabilities)


if __name__ == "__main__":
    import time
    start = time.time()
    play_df = utilities.get_nfldb_play_data(season_years=[2009, 2010, 2011, 2012, 2013, 2014])
    target_col = play_df["offense_won"]
    play_df.drop("offense_won", axis=1, inplace=True)
    play_df_train, play_df_test, target_col_train, target_col_test = (
       train_test_split(play_df, target_col, test_size=0.2, random_state=891))
    
    print("Took {0:.2f}s to query data".format(time.time() - start))
    start = time.time()
    # win_probability_model = WPModel()
    win_probability_model = WPModel(model_class=LogisticRegression,
                                    parameter_search_grid={'base_estimator__penalty': ['l1', 'l2'],
                                                           'base_estimator__C': [0.01, 0.1, 1, 10, 100],
                                                           'method': ['isotonic', 'sigmoid']})
    # win_probability_model = WPModel(model_class=RandomForestClassifier,
    #                                 parameter_search_grid={'base_estimator__n_estimators': [10, 50, 100, 150, 200],
    #                                                        'method': ['isotonic']})
    pipe = win_probability_model.model
    print("Took {0:.2f}s to create pipeline".format(time.time() - start))

    start = time.time()
    win_probability_model.fit(play_df_train, target_col_train)
    print("Took {0:.2f}s to fit pipeline".format(time.time() - start))
    print(win_probability_model.get_model_params())

    predicted_win_probabilities = pipe.predict_proba(play_df_test)[:,1]


    kde_offense_won = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(
        (predicted_win_probabilities[(target_col_test.values == 1)])[:, np.newaxis])
    kde_total = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(
        predicted_win_probabilities[:, np.newaxis])
    sample_probabilities = np.linspace(0, 1, 101)[:, np.newaxis]
    number_density_offense_won = np.exp(kde_offense_won.score_samples(sample_probabilities)) * np.sum((target_col_test))
    number_density_total = np.exp(kde_total.score_samples(sample_probabilities)) * len(target_col_test)
    number_offense_won = number_density_offense_won * np.sum(target_col_test) / np.sum(number_density_offense_won)
    number_total = number_density_total * len(target_col_test) / np.sum(number_density_total)
    predicted_win_percents = number_offense_won / number_total
    from statsmodels.stats.proportion import proportion_confint
    win_pct_errors = np.array([proportion_confint(sample_probabilities[i,0]*number_total[i], number_total[i], method="jeffrey", alpha=0.333) for i in range(len(number_total))])
    max_deviation = np.max(np.abs(predicted_win_percents - sample_probabilities[:, 0]))
    print("Max deviation: {0:.2f}%".format(max_deviation * 100))
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    ax = plt.figure().add_subplot(111)
    ax.plot([0, 1], [0, 1], ls="--", lw=2, color="black")
    ax.fill_between(sample_probabilities[:, 0],
                    win_pct_errors[:,0],
                    win_pct_errors[:,1],
                    facecolor="blue", alpha=0.25)
    ax.plot(sample_probabilities[:, 0], predicted_win_percents, ls="-", color="blue",
            label="Max Deviation = {0:.2f}%".format(max_deviation * 100))
    ax.set_xlabel("Predicted WP")
    ax.set_ylabel("Actual WP")
    ax.legend(loc="lower right")
    ax2 = ax.twinx()
    ax2.fill_between(sample_probabilities[:, 0], number_total,
                     facecolor="gray", alpha=0.25, interpolate=True)
    ax2.set_ylabel("Number of Plays in Test Set")
    # ax.fill_between(sample_probabilities[:,0], number_total,
    #                 facecolor="blue", alpha=0.25, interpolate=True, label="all_probabilities")
    # ax.fill_between(sample_probabilities[:,0], number_offense_won,
    #                 facecolor="red", alpha=0.25, interpolate=True, label="offense_won")
    # ax.legend(loc="upper right")
    ax.figure.savefig("test.png")

    
    # for i in range(len(sample_probabilities)):
    #     print(sample_probabilities[i], predicted_win_percents[i])

    
    #pipe.transform(play_df[nrows:nrows+20])
    # start = time.time()
    # predicted_probabilities = pipe.predict_proba(play_df_test)[:, 0]
    # print("Took {0:.2f}s to make predictions".format(time.time() - start))
    # print(len(predicted_probabilities), predicted_probabilities.max())

    # pipe_sigmoid = CalibratedClassifierCV(pipe, cv=3, method='sigmoid')
    # pipe_sigmoid.fit(play_df_train, target_col_train)
    # predicted_probabilities_sigmoid = pipe_sigmoid.predict_proba(play_df_test)[:, 0]
    # print(len(predicted_probabilities), predicted_probabilities.max())
    # # for i in range(len(play_df)):
    #     print(play_df.ix[i], predictions[i])
