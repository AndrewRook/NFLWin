"""Tools to get raw data ready for modeling."""
from __future__ import print_function, division

from sklearn.base import BaseEstimator
from sklearn.utils.validation import NotFittedError

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
