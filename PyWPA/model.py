'''
Contains functions that control building the
win probably estimation model.
'''
from __future__ import print_function, division
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split

import config as cf


def load_data():
    '''
    Loads the data files into a pandas
    dataframe.
    '''
    #Get all data files:
    data_files = glob.glob(os.path.join(cf.DATA_DIR,cf.DATA_PREFIX)+"*")
    loaded_seasons = []

    #Loop through all the files:
    data_df = None
    year_regex = re.compile(os.path.join(cf.DATA_DIR,cf.DATA_PREFIX)+r"_(?P<year>[\d]{4})")
    for filename in data_files:
        loaded_seasons.append(year_regex.search(filename).group('year'))
        season_data = pd.read_csv(filename, names=cf.DATA_COLUMNS)
        if data_df is None:
            data_df = season_data
        else:
            data_df = pd.concat([data_df,season_data], ignore_index=True)

            
    return {'data': data_df, 'loaded_seasons': loaded_seasons}


def rescale_data(data, inplace=False):
    '''
    Rescale the data from a play or set of plays
    to be uniform among every dimension.

    Arguments:
    data: A dataframe of play data, likely from load_data(),
            -OR-
        A dictionary-like object, containing:
            quarter: The quarter the game is in (1,2,3,4, or 5 for anything in OT).
            time_remaining: Seconds counting down from the start of the quarter
                (e.g. the start of the quarter is 15*60, the end of the quarter
                is 0).
            score_diff: The score differential (home - away).
            is_offense_home: Is the offense the home team? Boolean true/false.
            down: What down is it (1,2,3, or 4).
            distance: How many yards to go for the first down.
            field_position: How many yards from your own endzone you are (1 is your one-yard
                line, 99 is 1 yard from a touchdown).
    inplace (False): if False, copy 'data' before editing it, and return the modified copy.
        if True, modify inplace.

    Returns:
    An object of the same type as 'data', with the following scalings applied:
        quarter_scaled = (quarter-1)/4.
        time_remaining_scaled = 1. - (time_remaining-1)/(15*60.)
        score_diff_scaled = 1/(1+np.exp(-0.1*score_diff))
        is_offense_home_scaled = is_offense_home*1.
        down_scaled = (down-1)/3.
        distance_scaled = 2*(1/(1+np.exp(-0.2*(distance-1)))-0.5)
        field_position_scaled = (field_position-1)/99.
    '''
    scaled_data = data
    if inplace == False:
        scaled_data = data.copy()
    scaled_data['quarter'] = (data['quarter']-1)/4.
    scaled_data['time_remaining'] = 1. - (data['time_remaining']-1)/(15*60.)
    scaled_data['score_diff'] = 1/(1+np.exp(-0.1*data['score_diff']))
    scaled_data['is_offense_home'] = data['is_offense_home']*1.
    scaled_data['down'] = (data['down']-1)/3.
    scaled_data['distance'] = 2*(1/(1+np.exp(-0.2*(data['distance']-1)))-0.5)
    scaled_data['field_position'] = (data['field_position']-1)/99.

    if inplace == False:
        return scaled_data

    
def fit_model(data_df, n_neighbors_list=[10,20,40,80,160,320,640,1280],
              test_frac=0.2, n_bootstrap=20):
    '''
    Fit a model to the data.

    Arguments:
    data_df: A dataframe from rescale_data().
    n_neighbors_list ([10,20,40,80,160,320,640,1280]): How many neighbors to try in the fit.
    test_frac (0.2): What fraction of the data to use to test.
    n_bootstrap (20): The number of iterations to use to estimate the
        error. More will be more robust but take longer.

    Returns:
    A dictionary with the following key/values:
        model: The fitted model (a KNeighborsClassifier instance that has been fit to the data).
        bootstrapped_model_list
    '''
    #Make the train/test split:
    features = data_df[cf.DATA_COLUMNS[:-1]].values
    target = data_df[cf.DATA_COLUMNS[-1]].values
    features_train, features_test, target_train, target_test = \
      train_test_split(features, target, test_size=test_frac)

    #Find the best model:
    best_model = None
    best_fitness = None
    for n in n_neighbors_list:
        #Make the fit:
        knn = KNeighborsClassifier(n_neighbors=n, algorithm='ball_tree', leaf_size=30)
        knn.fit(features_train, target_train)

        #Predict the test data:
        fit_metric = compute_goodness_of_fit(knn.predict_proba(features_test), target_test)
        print(n,fit_metric)

        #If this is better than the previous best, use it:
        if best_fitness is None or abs(fit_metric) < abs(best_fitness):
            best_model = knn
            best_fitness = fit_metric

    print("Best model has n_neighbors={0:d}".format(best_model.n_neighbors))
    
    #Bootstrap it:
    bootstrapped_models = []
    for i in range(n_bootstrap):
        indices = np.random.randint(len(features_train),size=len(features_train))
        bootstrapped_features = features_train[indices,:]
        bootstrapped_target = target_train[indices]
        knn_temp = KNeighborsClassifier(n_neighbors=best_model.n_neighbors, algorithm='ball_tree', leaf_size=30)
        knn_temp.fit(bootstrapped_features, bootstrapped_target)
        bootstrapped_models.append(knn_temp)
    return {'model':best_model, 'bootstrapped_model_list':bootstrapped_models}


def compute_goodness_of_fit(predicted_probabilities, target_test):
    '''
    Take a set of predicted probabilities and turn it into a metric
    useable to compare models against each other.

    Arguments:
    predicted_probabilities: The output of KNeighborsClassifier.predict_proba (or equivalent).
    target_test: The actual answers. 

    Returns:
    fit_quality_metric: The sum of the scaled probabilities for each target, with scaling as follows:
    1. Any probability less than 0.5 is set at probability - 1.0 (e.g. 0.2 becomes -0.8).
    2. Any probability greater than 0.5 is set at 1.0 - probability (e.g 0.8 becomes 0.2).
    3. If the probability is exactly 0.5, then set one to -0.5 and the other to 0.5.
    '''

    #Rescale the probabilities:
    scaled_probabilities = predicted_probabilities.copy()
    scaled_probabilities[scaled_probabilities < 0.5] -= 1.0
    scaled_probabilities[scaled_probabilities > 0.5] = 1.0 - scaled_probabilities[scaled_probabilities > 0.5]
    toss_ups = (abs(scaled_probabilities[:,0] - 0.5) <= 1e-4)
    scaled_probabilities[toss_ups,0] -= 1 #Can leave the other one as is.

    #Apply the scaled probabilities to the target, and sum:
    fit_metric = np.sum(scaled_probabilities[np.arange(len(target_test)),target_test])

    return fit_metric


if __name__ == "__main__":
    data_info = load_data()
    scaled_data = rescale_data(data_info['data'])

    model_info_dict = fit_model(scaled_data, n_neighbors_list=[10,20,40,80,160])

    for i in range(1000,1010):
       probability = model_info_dict['model'].predict_proba(scaled_data[cf.DATA_COLUMNS[:-1]].iloc[i].reshape(1,-1))
       error = np.std([model.predict_proba(scaled_data[cf.DATA_COLUMNS[:-1]].iloc[i].reshape(1,-1))[0,1] for model in model_info_dict['bootstrapped_model_list']], ddof=1)
       print(i,probability[0,1], error)
    #print(knn.predict_proba(scaled_data[cf.DATA_COLUMNS[:-1]].iloc[1030:1040]))
