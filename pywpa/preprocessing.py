"""Tools to get raw data ready for modeling."""
from __future__ import print_function, division

from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError

class CreateScoreDifferential(BaseEstimator):
    """Convert home and away scores into a differential (home - away).

    Parameters
    ----------
    home_score_colname : string
        The name of the column containing the score of the home team.
    away_score_colname : string
        The name of the column containing the score of the away team.
    score_differential_colname : string (default=``"score_differential"``)
        The name of column containing the score differential. Must not already
        exist in the DataFrame.
    copy : boolean (default = ``True``)
        If ``False``, add the score differential in place.
    """
    def __init__(self, home_score_colname,
                 away_score_colname,
                 score_differential_colname="score_differential",
                 copy=True):
        self.home_score_colname = home_score_colname
        self.away_score_colname = away_score_colname
        self.score_differential_colname = score_differential_colname
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Create the score differential column.

        Parameters
        ----------
        X : Pandas DataFrame, of shape(number of plays, number of features)
            NFL play data.
        y : Numpy array, with length = number of plays, or None
            1 if the home team won, 0 if not.
            (Used as part of Scikit-learn's ``Pipeline``)

        Returns
        -------
        X : Pandas DataFrame, of shape(number of plays, number of features + 1)
            The input DataFrame, with the score differential column added.
        """
        try:
            score_differential = X[self.home_score_colname] - X[self.away_score_colname]
        except KeyError:
            raise KeyError("CreateScoreDifferential: data missing required column. Must "
                           "include columns named {0} and {1}".format(self.home_score_colname,
                                                                      self.away_score_colname))
        if self.score_differential_colname in X.columns:
            raise KeyError("CreateScoreDifferential: column {0} already in DataFrame, and can't "
                           "be used for the score differential".format(self.score_differential_colname))
        if self.copy:
            X = X.copy()

        X[self.score_differential_colname] = score_differential

        return X
        


class CheckColumnNames(BaseEstimator):
    """Make sure user has the right column names, in the right order.

    This is a useful first step to make sure that nothing
    is going to break downstream.

    Attributes
    ----------
    column_names : list of strings
        A list of column names that need to be present in the scoring
        data. All other columns will be stripped out. The order of the
        columns will be applied to any scoring
        data as well, in order to handle the fact that pandas lets
        you play fast and loose with column order.
    """
    def __init__(self):
        self.column_names = None
        self._fit = False

    def fit(self, X, y=None):
        """Grab the column names from a Pandas DataFrame.

        Parameters
        ----------
        X : Pandas DataFrame, of shape(number of plays, number of features)
            NFL play data.
        y : Numpy array, with length = number of plays, or None
            1 if the home team won, 0 if not.
            (Used as part of Scikit-learn's ``Pipeline``)

        Returns
        -------
        self : For compatibility with Scikit-learn's ``Pipeline``. 
        """
        self.column_names = X.columns
        self._fit = True

        return self

    def transform(self, X, y=None):
        """Apply the column ordering to the data.

        Parameters
        ----------
        X : Pandas DataFrame, of shape(number of plays, number of features)
            NFL play data.
        y : Numpy array, with length = number of plays, or None
            1 if the home team won, 0 if not.
            (Used as part of Scikit-learn's ``Pipeline``)

        Returns
        -------
        X : Pandas DataFrame, of shape(number of plays, ``len(column_names)``)
            The input DataFrame, properly ordered and with extraneous
            columns dropped

        Raises
        ------
        KeyError
            If the input data frame doesn't have all the columns specified
            by ``column_names``.
        NotFittedError
            If ``transform`` is called before ``fit``.
        """
        if not self._fit:
            raise NotFittedError("CheckColumnName: Call 'fit' before 'transform")
        try:
            return X[self.column_names]
        except KeyError:
            raise KeyError("CheckColumnName: DataFrame does not have required columns. "
                           "Must contain at least {0}".format(self.column_names))
        

class ConvertToNumpy(BaseEstimator):
    """Preserve the right column order between fitting and predicting"""
