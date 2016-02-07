'''
This module has functions to predict the Win Probability (WP)
and Win Probability Added (WPA).
'''
from __future__ import print_function, division
# try:
#    import cPickle as pickle
# except ImportError:
#    import pickle

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import config as cf
from model import rescale_data

def load_model():
    '''
    Load the saved model into memory.

    Arguments:
    None

    Returns:
    model_info_dict: A dictionary containing the following
        key-value pairs:
        seasons: The seasons used in making the model.
        fit_model: The best-fitting KNeighborsClassifier model.
        bootstrapped_models: A list of models fit to bootstrapped
            resamples of the data
    '''
    #Load the model in:
    model = joblib.load(cf.MODEL_FILENAME)

    return model

def compute_wp(model, play_data):
    '''
    Compute the Win Percentage for a given situation.

    Arguments:
    model: A dictionary containing model information (at least the 'fit_model'
        and 'bootstrapped_models' keys) loaded in from load_model().
    play_data: Either a dictionary or Pandas Play_dataFrame with at least the
        following keys/columns:
        quarter: The quarter the game is in (1,2,3,4, or 5 for anything in OT).
        time_remaining: Seconds counting down from the start of the quarter
            (e.g. the start of the quarter is 15*60, the end of the quarter
            is 0).
        score_diff: The score differential (home - away).
        is_offense_home: Is the offense the home team? Boolean true/false.
        down: What down is it (1,2,3, or 4).
        distance: How many yards to go for the first down.
        field_position: How many yards from your own endzone you are
            (1 is your one-yard line, 99 is 1 yard from a touchdown).

    Returns:
    prediction: A dictionary/Play_dataFrame with the following keys/columns:
        WP: The predicted win percentage for the play.
        WP_error: The predicted uncertainty in the WP estimate based on
            sampling error.
    '''
    rescaled_play_data = rescale_data(play_data)
    try:
        #Start by assuming the play_data is a dataframe:
        features = rescaled_play_data[cf.DATA_COLUMNS[:-1]].values
        prediction = pd.DataFrame(index=np.arange(features.shape[0]))
    except TypeError:
        feature_list = []
        for key in cf.DATA_COLUMNS[:-1]:
            feature_list.append(rescaled_play_data[key])
        features = np.array(feature_list).reshape(1,-1)
        prediction = {}

    win_prob = model['fit_model'].predict_proba(features)[:,1]
    win_prob_errors = np.std([bootstrapped_model.predict_proba(features)[:,1] \
                       for bootstrapped_model in model['bootstrapped_models']], axis=0, ddof=1)
    print(win_prob)
    print(win_prob_errors)
    prediction['WP'] = win_prob
    prediction['WP_error'] = win_prob_errors

    return prediction

def compute_wpa(model, before_play_data, after_play_data):
    '''
    A simple wrapper around compute_wp to compute the win probability added.

    model: see the docstring for the 'model' argument of compute_wp.
    before_play_data: see the docstring for the 'play_data' argument of compute_wp.
    after_play_data: Same rules as for before_play_data, except containing the game
        state after the play(s) end. 
    '''
    
    

if __name__ == "__main__":
    import time
    start = time.time()
    model = load_model()
    print("took {0:.2f}s".format(time.time()-start))
    play_dict = {
        "quarter": 4,
        "time_remaining": 180,
        "score_diff": -4, #Offense winning is positive
        "is_offense_home": True,
        "down": 2,
        "distance": 7,
        "field_position": 80,
        }
    
    compute_wp(model, play_dict)
