'''
This module has functions to predict the Win Probability (WP)
and Win Probability Added (WPA).
'''
from __future__ import print_function, division
try:
   import cPickle as pickle
except ImportError:
   import pickle
   
from sklearn.externals import joblib

import config as cf

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

if __name__ == "__main__":
    import time
    start = time.time()
    load_model()
    print("took {0:.2f}s".format(time.time()-start))
    
