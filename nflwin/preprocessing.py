"""Tools to get raw data ready for modeling."""
from __future__ import print_function, division

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import NotFittedError

class ComputeElapsedTime(BaseEstimator):
    """Compute the total elapsed time from the start of the game.

    Parameters
    ----------
    quarter_colname : string
        Which column indicates what quarter it is.
    quarter_time_colname : string
        Which column indicates how much time has elapsed in the current quarter.
    quarter_to_second_mapping : dict (default=``{"Q1": 0, "Q2": 900, "Q3": 1800, "Q4": 2700,
                                                 "OT": 3600, "OT2": 4500, "OT3": 5400}``)
        What mapping to use between the string values in the quarter column and the seconds they
        correspond to. Mostly useful if your data had quarters listed as something like "Quarter 1"
        or "q1" instead of the values from ``nfldb``.
    total_time_colname : string (default="total_elapsed_time")
        What column name to store the total elapsed time under.
    copy : boolean (default=True)
        Whether to add the new column in place.
    """
    def __init__(self, quarter_colname, quarter_time_colname,
                 quarter_to_second_mapping={"Q1": 0, "Q2": 900, "Q3": 1800, "Q4": 2700,
                                            "OT": 3600, "OT2": 4500, "OT3": 5400},
                 total_time_colname="total_elapsed_time", copy=True):
        self.quarter_colname = quarter_colname
        self.quarter_time_colname = quarter_time_colname
        self.quarter_to_second_mapping = quarter_to_second_mapping
        self.total_time_colname = total_time_colname
        self.copy = copy

    def fit(self, X, y=None):
        return self

    
    def transform(self, X, y=None):
        """Create the new column.

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
            The input DataFrame, with the new column added.

        Raises
        ------
        KeyError
            If ``quarter_colname`` or ``quarter_time_colname`` don't exist, or
            if ``total_time_colname`` **does** exist.
        TypeError
            If the total time elapsed is not a numeric column, which typically indicates
            that the mapping did not apply to every row.
        """

        if self.quarter_colname not in X.columns:
            raise KeyError("ComputeElapsedTime: quarter_colname {0} does not exist in dataset."
                           .format(self.quarter_colname))
        if self.quarter_time_colname not in X.columns:
            raise KeyError("ComputeElapsedTime: quarter_time_colname {0} does not exist in dataset."
                           .format(self.quarter_time_colname))

        if self.total_time_colname in X.columns:
            raise KeyError("ComputeElapsedTime: total_time_colname {0} already exists in dataset."
                           .format(self.total_time_colname))

        if self.copy:
            X = X.copy()

        try:
            time_elapsed = X[self.quarter_colname].replace(self.quarter_to_second_mapping) + X[self.quarter_time_colname]
        except TypeError:
            raise TypeError("ComputeElapsedTime: Total time elapsed not numeric. Check your mapping from quarter name to time.")

        X[self.total_time_colname] = time_elapsed

        return X
    

class ComputeIfOffenseIsHome(BaseEstimator):
    """Determine if the team currently with possession is the home team.


    Parameters
    ----------
    offense_team_colname : string
        Which column indicates what team was on offense.
    home_team_colname : string
        Which column indicates what team was the home team.
    offense_home_team_colname : string (default="is_offense_home")
        What column to store whether or not the offense was the home team.
    copy : boolean (default=True)
        Whether to add the new column in place.
    """
    def __init__(self, offense_team_colname,
                 home_team_colname,
                 offense_home_team_colname="is_offense_home",
                 copy=True):
        self.offense_team_colname = offense_team_colname
        self.home_team_colname = home_team_colname
        self.offense_home_team_colname = offense_home_team_colname
        self.copy = copy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Create the new column.

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
            The input DataFrame, with the new column added.

        Raises
        ------
        KeyError
            If ``offense_team_colname`` or ``home_team_colname`` don't exist, or
            if ``offense_home_team_colname`` **does** exist.
        """

        if self.home_team_colname not in X.columns:
            raise KeyError("ComputeIfOffenseWon: home_team_colname {0} does not exist in dataset."
                           .format(self.home_team_colname))
        if self.offense_team_colname not in X.columns:
            raise KeyError("ComputeIfOffenseWon: offense_team_colname {0} does not exist in dataset."
                           .format(self.offense_team_colname))

        if self.offense_home_team_colname in X.columns:
            raise KeyError("ComputeIfOffenseWon: offense_home_team_colname {0} already exists in dataset."
                           .format(self.offense_home_team_colname))

        if self.copy:
            X = X.copy()

        X[self.offense_home_team_colname] = (X[self.home_team_colname] == X[self.offense_team_colname])

        return X


class MapToInt(BaseEstimator):
    """Map a column of values to integers.

    Mapping to integer is nice if you know a column
    only has a few specific values in it, but you need
    to convert it to integers before one-hot encoding it.

    Parameters
    ----------
    colname : string
        The name of the column to perform the mapping on.
    copy : boolean (default=True)
        If ``False``, apply the mapping in-place.

    Attributes
    ----------
    mapping : dict
        Keys are the unique values of the column, values are the
        integers those values will be mapped to.

    Note
    ----
    The ``transform`` method DOES NOT CHECK to see if the input
    DataFrame only contains values in ``mapping``. Any values not
    in ``mapping`` will be left alone, which can cause subtle bugs
    if you're not careful.
    """

    def __init__(self, colname, copy=True):
        self.colname = colname
        self.copy = copy
        self.mapping = None

    def fit(self, X, y=None):
        """Find all unique strings and construct the mapping.

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

        Raises
        ------
        KeyError
            If ``colname`` is not in ``X``.

        """
        if self.colname not in X.columns:
            raise KeyError("MapStringsToInt: Required column {0} "
                           "not present in data".format(self.colname))
        unique_values = X[self.colname].unique()
        
        self.mapping = {unique_values[i]: i for i in range(len(unique_values))}
        
        try:
            del self.mapping[np.nan]
        except KeyError:
            pass
        
        return self

    def transform(self, X, y=None):
        """Apply the mapping to the data.

        Parameters
        ----------
        X : Pandas DataFrame, of shape(number of plays, number of features)
            NFL play data.
        y : Numpy array, with length = number of plays, or None
            1 if the home team won, 0 if not.
            (Used as part of Scikit-learn's ``Pipeline``)

        Returns
        -------
        X : Pandas DataFrame, of shape(number of plays, number of features)
            The input DataFrame, with the mapping applied.

        Raises
        ------
        NotFittedError
            If ``transform`` is called before ``fit``.
        KeyError
            If ``colname`` is not in ``X``.
        """
        if not self.mapping:
            raise NotFittedError("MapStringsToInt: Must fit before transform.")
        
        if self.colname not in X.columns:
            raise KeyError("MapStringsToInt: Required column {0} "
                           "not present in data".format(self.colname))

        if self.copy:
            X = X.copy()

        X[self.colname].replace(self.mapping, inplace=True)

        return X
        

class OneHotEncoderFromDataFrame(BaseEstimator):
    """One-hot encode a DataFrame.

    This cleaner wraps the standard scikit-learn OneHotEncoder,
    handling the transfer between column name and column index.

    Parameters
    ----------
    categorical_feature_names : "all" or array of column names.
        Specify what features are treated as categorical.
        * "all" (default): All features are treated as categorical.
        * array of column names: Array of categorical feature names.
    dtype : number type, default=np.float.
        Desired dtype of output.
    handle_unknown : str, "error" (default) or "ignore".
        Whether to raise an error or ignore if an unknown categorical feature
        is present during transform.
    copy : boolean (default=True)
        If ``False``, apply the encoding in-place.
    """

    @property
    def dtype(self):
        return self._dtype
    @dtype.setter
    def dtype(self, dtype):
        self._dtype = dtype
        self.onehot.dtype = self._dtype

    @property
    def handle_unknown(self):
        return self._handle_unknown
    @handle_unknown.setter
    def handle_unknown(self, handle_unknown):
        self._handle_unknown = handle_unknown
        self.onehot.handle_unknown = self._handle_unknown
        
    def __init__(self,
                 categorical_feature_names="all",
                 dtype=np.float,
                 handle_unknown="error",
                 copy=True):
        self.onehot = OneHotEncoder(sparse=False, n_values="auto",
                                    categorical_features="all") #We'll subset the DF
        self.categorical_feature_names = categorical_feature_names
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.copy = copy

    def fit(self, X, y=None):
        """Convert the column names to indices, then compute the one hot encoding.

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

        if self.categorical_feature_names == "all":
            self.categorical_feature_names = X.columns

        #Get all columns that need to be encoded:
        data_to_encode = X[self.categorical_feature_names]
            

        self.onehot.fit(data_to_encode)

        return self

    def transform(self, X, y=None):
        """Apply the encoding to the data.
        
        Parameters
        ----------
        X : Pandas DataFrame, of shape(number of plays, number of features)
            NFL play data.
        y : Numpy array, with length = number of plays, or None
            1 if the home team won, 0 if not.
            (Used as part of Scikit-learn's ``Pipeline``)

        Returns
        -------
        X : Pandas DataFrame, of shape(number of plays, number of new features)
            The input DataFrame, with the encoding applied.
        """
        if self.copy:
            X = X.copy()
        
        data_to_transform = X[self.categorical_feature_names]
        transformed_data = self.onehot.transform(data_to_transform)

        #TODO (AndrewRook): Find good column names for the encoded columns.
        colnames = ["onehot_col{0}".format(i+1) for i in range(transformed_data.shape[1])]
        #Create a dataframe from the transformed columns (setting the index is critical for
        #merging with data containing non-standard indexes)
        transformed_df = pd.DataFrame(transformed_data, columns=colnames, index=X.index)
        
        X.drop(self.categorical_feature_names, axis=1, inplace=True)
        X[transformed_df.columns] = transformed_df
        
        return X
            
    

class CreateScoreDifferential(BaseEstimator):
    """Convert offense and defense scores into a differential (offense - defense).

    Parameters
    ----------
    home_score_colname : string
        The name of the column containing the score of the home team.
    away_score_colname : string
        The name of the column containing the score of the away team.
    offense_home_colname : string
        The name of the column indicating if the offense is home.
    score_differential_colname : string (default=``"score_differential"``)
        The name of column containing the score differential. Must not already
        exist in the DataFrame.
    copy : boolean (default = ``True``)
        If ``False``, add the score differential in place.
    """
    def __init__(self, home_score_colname,
                 away_score_colname,
                 offense_home_colname,
                 score_differential_colname="score_differential",
                 copy=True):
        self.home_score_colname = home_score_colname
        self.away_score_colname = away_score_colname
        self.offense_home_colname = offense_home_colname
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
            score_differential = ((X[self.home_score_colname] - X[self.away_score_colname]) *
                                  (2 * X[self.offense_home_colname] - 1))
        except KeyError:
            raise KeyError("CreateScoreDifferential: data missing required column. Must "
                           "include columns named {0}, {1}, and {2}".format(self.home_score_colname,
                                                                            self.away_score_colname,
                                                                            self.offense_home_colname))
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
    is going to break downstream, but can also be used effectively
    to drop columns that are no longer necessary.

    Parameters
    ----------
    column_names : ``None``, or list of strings
        A list of column names that need to be present in the scoring
        data. All other columns will be stripped out. The order of the
        columns will be applied to any scoring
        data as well, in order to handle the fact that pandas lets
        you play fast and loose with column order. If ``None``,
        will obtain every column in the DataFrame passed to the
        ``fit`` method.
    copy : boolean (default=``True``)
        If ``False``, add the score differential in place.
       
    """
    def __init__(self, column_names=None, copy=True):
        self.column_names = column_names
        self.copy = copy
        self._fit = True
        self.user_specified_columns = False
        if self.column_names is None:
            self._fit = False
        else:
            self.user_specified_columns = True
            

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
        if not self.user_specified_columns:
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
        
        if self.copy:
            X = X.copy()

        try:
                
            return X[self.column_names]
        except KeyError:
            raise KeyError("CheckColumnName: DataFrame does not have required columns. "
                           "Must contain at least {0}".format(self.column_names))
