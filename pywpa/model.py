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
    copy : boolean (default=``True``)
        Whether or not to copy data at each step in the modeling pipeline. Setting this to ``False``
        can speed up execution time but will also overwrite your input data.
    model_class : A valid Scikit-Learn classifier (default=``sklearn.linear_model.LogisticRegression``)
        The type of classifier to be used to make the predictions.
    model_kwargs : dict (default=``{}``)
        Keyword arguments for ``model_class``. 
    parameter_search_grid : ``None`` or dict (default=``None``)
        If not ``None``, then ``sklearn.grid_search.GridSearchCV`` will be
        used and this parameter should contain a dictionary with the ``param_grid``
        positional argument for that estimator. See note below about naming conventions,
        which are slightly nonstandard.


    Notes
    -----
    When using the ``parameter_search_grid`` parameter, in order to adjust the hyperparameters
    of the selected ``model_class`` you'll need to prepend them with ``base_estimator``. For
    instance, in the case of ``sklearn.linear_model.LogisticRegression`` you could have
    ``{'base_estimator__penalty': ['l1', 'l2'], 'base_estimator__C': [0.1, 1, 10]}``. This is
    necessary because ``WPModel`` wraps the ``model_class`` with an instances of
    ``sklearn.calibration.CalibratedClassifierCV`` to attempt to provide a better fit to the
    probabilities, rather than just optimizing on predictive accuracy. If you want to run
    cross-validation on the ``CalibratedClassifierCV`` hyperparameters you can: those you **don't**
    need to preface with ``base_estimator__``. 
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
                 home_team_colname = "home_team",
                 copy=True,
                 model_class=LogisticRegression,
                 model_kwargs={},
                 parameter_search_grid=None):
        self.home_score_colname = home_score_colname
        self.away_score_colname = away_score_colname
        self.quarter_colname = quarter_colname
        self.time_colname = time_colname
        self.down_colname = down_colname
        self.yards_to_go_colname = yards_to_go_colname
        self.yardline_colname = yardline_colname
        self.offense_team_colname = offense_team_colname
        self.home_team_colname = home_team_colname
        self.copy = copy
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.parameter_search_grid = parameter_search_grid

        self.model = self.create_pipeline()

        self._fit = False


    def create_pipeline(self):
        """Create the win probability estimation pipeline.


        Returns
        -------
        sklearn.pipeline.Pipeline
            A pipeline object containing all the steps in the model.
        """

        steps = []

        is_offense_home = preprocessing.ComputeIfOffenseIsHome(self.offense_team_colname,
                                                               self.home_team_colname,
                                                               copy=self.copy)
        steps.append(("compute_offense_home", is_offense_home))
        score_differential = preprocessing.CreateScoreDifferential(self.home_score_colname,
                                                                   self.away_score_colname,
                                                                   is_offense_home.offense_home_team_colname,
                                                                   copy=self.copy)
        steps.append(("create_score_differential", score_differential))
        steps.append(("map_downs_to_int", preprocessing.MapToInt(self.down_colname, copy=self.copy)))
        total_time_elapsed = preprocessing.ComputeElapsedTime(self.quarter_colname, self.time_colname, copy=self.copy)
        steps.append(("compute_total_time_elapsed", total_time_elapsed))
        steps.append(("remove_unnecessary_columns", preprocessing.CheckColumnNames(
            column_names=[is_offense_home.offense_home_team_colname,
                          score_differential.score_differential_colname,
                          total_time_elapsed.total_time_colname,
                          self.yardline_colname,
                          self.yards_to_go_colname,
                          self.down_colname],
            copy=self.copy)))
        steps.append(("encode_categorical_columns", preprocessing.OneHotEncoderFromDataFrame(
            categorical_feature_names=[self.down_colname],
            copy=self.copy)))

        model = self.model_class(**self.model_kwargs)
        model = CalibratedClassifierCV(model, cv=2, method="isotonic")
        if self.parameter_search_grid is not None:
            model = GridSearchCV(model, self.parameter_search_grid,
                                 scoring=self._brier_loss_scorer)
        else:
            model = model
        steps.append(("compute_model", model))

        pipe = Pipeline(steps)
        return pipe

    @staticmethod
    def _brier_loss_scorer(estimator, X, y):
        """Use the Brier loss to estimate model score.

        For use in GridSearchCV, instead of accuracy.
        """
        predicted_positive_probabilities = estimator.predict_proba(X)[:, 1]
        return 1. - brier_score_loss(y, predicted_positive_probabilities)

    def fit(self, X, y):
        """Fit the model to data.

        Parameters
        ----------
        X : Pandas DataFrame, of shape (n_plays, n_features)
            The plays to be used to fit the model.
        y : numpy array or Pandas Series, of length n_plays
            The result of the game (offense win/lose) for each play.

        Returns
        -------
        self
            Preserving the Scikit-learn idiom.
        """
        
        self.model.fit(X, y)
        self._fit = True
        return self

    def get_model_params(self):
        """Return the hyperparameters used in the fitted model.

        This wrapper is necessary because the API changes if you
        use ``GridSearchCV`` instead of a bare classifier.

        Returns
        -------
        dict
            The parameter names and their values.

        Raises
        ------
        NotFittedError
            If ``fit`` has not been called yet.

        Notes
        -----
        Because the classifier is wrapped to force better probability estimates,
        the parameters of the classifier will be prepended with "best_estimator__".
        """
        if not self._fit:
            raise NotFittedError("WPModel.get_model_params: Must run 'fit' first!")

        last_step = self.model.steps[-1][1]

        try:
            best_hyperparams_dict = last_step.best_params_
            params_dict = last_step.estimator.get_params()
            for param in best_hyperparams_dict:
                params_dict[param] = best_hyperparams_dict[param]
            return params_dict
        except AttributeError:
            return last_step.get_params()

if __name__ == "__main__":
    import time
    start = time.time()
    play_df = utilities.get_nfldb_play_data(season_years=[2013, 2014])
    target_col = play_df["offense_won"]
    play_df.drop("offense_won", axis=1, inplace=True)
    play_df_train, play_df_test, target_col_train, target_col_test = (
       train_test_split(play_df, target_col, test_size=0.2, random_state=891))
    
    print("Took {0:.2f}s to query data".format(time.time() - start))
    start = time.time()
    #win_probability_model = WPModel()
    # win_probability_model = WPModel(model_class=LogisticRegression,
    #                                 parameter_search_grid={'base_estimator__penalty': ['l1', 'l2'],
    #                                                        'base_estimator__C': [0.1, 1, 10],
    #                                                        'method': ['isotonic']})
    win_probability_model = WPModel(model_class=RandomForestClassifier,
                                    parameter_search_grid={'base_estimator__n_estimators': [5, 40],
                                                           'method': ['isotonic']})
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
    number_offense_won = np.exp(kde_offense_won.score_samples(sample_probabilities)) * np.sum((target_col_test))
    number_total = np.exp(kde_total.score_samples(sample_probabilities)) * len(target_col_test) #actually number density
    predicted_win_percents = number_offense_won / number_total
    max_deviation = np.max(np.abs(predicted_win_percents - sample_probabilities[:, 0]))
    print("Max deviation: {0:.2f}%".format(max_deviation * 100))
    print("DEBUG: ", len(target_col_test))
    
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(111)
    ax.plot([0, 1], [0, 1], ls="--", lw=2, color="black")
    ax.plot(sample_probabilities[:, 0], predicted_win_percents, ls="-", lw=3, color="green",
            label="Max Deviation = {0:.2f}%".format(max_deviation * 100))
    ax.set_xlabel("Predicted WP")
    ax.set_ylabel("Actual WP")
    ax.legend(loc="lower right")
    ax2 = ax.twinx()
    ax2.fill_between(sample_probabilities[:, 0], number_total,
                     facecolor="gray", alpha=0.25, interpolate=True)
    ax2.set_ylabel("Number of Plays")
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
