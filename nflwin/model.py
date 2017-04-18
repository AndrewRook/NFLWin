"""Tools for creating and running the model."""
from __future__ import print_function, division

import os

import preprocessing
import utilities

class WPModel(object):
    """The object that computes win probabilities.

    Parameters
    ----------
    model : A Scikit-learn pipeline (or equivalent)
        The model used to compute WP. 
    

    Attributes
    ----------
    model_directory : string
        The directory where all models will be saved to or loaded from.

    """
    model_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    _default_model_filename = "default_model.nflwin"

    def __init__(self, model):
        self.model = model
                                       

    def predict_wp(self, plays, copy_data=True):
        """Estimate the win probability for a set of plays.

        Basically a simple wrapper around whatever the model uses to make predictions,
        takes in a DataFrame and then spits out an array of predicted
        win probabilities. By using a wrapper rather than the explicit prediction
        methods for a given model if we change the model to something with a different
        prediction interface we can change the code in this function without changing
        the interface an nflwin user interacts with.

        Parameters
        ----------
        plays : Pandas DataFrame
            The input data to use to make the predictions.
        copy_data : boolean (default = ``True``)
            Whether or not to estimate probability on a copy of the data.
            If you have a large dataset you may want to set this to ``False``
            to save time, but note that the model may not be idempotent so if
            you think you may want to re-use your input data in the same program
            make sure to set this to ``True``.

        Returns
        -------
        Numpy array, of length ``len(plays)``
            Predicted probability that the offensive team in each play
            will go on to win the game.

        """
        if copy_data:
            plays = plays.copy(deep=True)

        return None



    def save_model(self, filename=None):
        """Save the WPModel instance to disk.

        All models are saved to the same place, with the installed
        NFLWin library (given by ``WPModel.model_directory``). 

        Parameters
        ----------
        filename : string (default=None):
            The filename to use for the saved model. If this parameter
            is not specified, save to the default filename. Note that if a model
            already lists with this filename, it will be overwritten. Note also that
            this is a filename only, **not** a full path. If a full path is specified
            it is likely (albeit not guaranteed) to cause errors.

        Returns
        -------
        ``None``
        """

        if filename is None:
            filename = self._default_model_filename
        joblib.dump(self, os.path.join(self.model_directory, filename))

    @classmethod
    def load_model(cls, filename=None):
        """Load a saved WPModel.

        Parameters
        ----------
        Same as ``save_model``.

        Returns
        -------
        ``nflwin.WPModel`` instance.
        """
        if filename is None:
            filename = cls._default_model_filename
            
        return joblib.load(os.path.join(cls.model_directory, filename))
