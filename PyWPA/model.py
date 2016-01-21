'''
Contains functions that control building the
win probably estimation model.
'''
from __future__ import print_function, division
import glob
import os
import re
try:
   import cPickle as pickle
except ImportError:
   import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib

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
        regex_search_results = year_regex.search(filename)
        if regex_search_results is not None:
            loaded_seasons.append(regex_search_results.group('year'))
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

    
def fit_model(data_df, n_neighbors_list=cf.DEFAULT_N_NEIGHBORS_LIST,
              test_frac=0.2, n_bootstrap=20):
    '''
    Fit a model to the data.

    Arguments:
    data_df: A dataframe from rescale_data().
    n_neighbors_list (config.DEFAULT_N_NEIGHBORS_LIST): How many neighbors to try in the fit.
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

    leaf_size = 40

    #Find the best model:
    best_model = None
    best_fitness = None
    for n in n_neighbors_list:
        #Make the fit:
        knn = KNeighborsClassifier(n_neighbors=n, algorithm='ball_tree', leaf_size=leaf_size)
        knn.fit(features_train, target_train)

        #Predict the test data:
        fit_metric = compute_goodness_of_fit(knn.predict_proba(features_test)[:,1], target_test) #Need the [:,1] because the prediction returns probabilities for both 0 and 1
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
        knn_temp = KNeighborsClassifier(n_neighbors=best_model.n_neighbors, algorithm='ball_tree', leaf_size=leaf_size)
        knn_temp.fit(bootstrapped_features, bootstrapped_target)
        bootstrapped_models.append(knn_temp)
    return {'model':best_model, 'bootstrapped_model_list':bootstrapped_models}


def _compute_goodness_of_fit(predicted_probabilities, target_test):
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

def compute_goodness_of_fit(predicted_probabilities, target_test):
    '''
    Take a set of predicted probabilities and computes a fitting function
    as follows:
    1. Split the plays on whether or not the offense won the game.
    2. Use a kernel-density estimate of each set of play probabilities.
    3. Take the ratio of the two, and compare that to the expected frequency
        of wins at that predicted probability to compute a final
        goodness of fit statistic: integral((expected-actual)**2). 

    Arguments:
    predicted_probabilities: The output of KNeighborsClassifier.predict_proba (or equivalent).
    target_test: The actual answers. 

    Returns:
    goodness_of_fit: the integral of the squared differences between the expected and
        actual probabilities.
    '''
    num_mult = 1000000.
    integer_predicted_probabilities = (predicted_probabilities*num_mult).astype(np.int)
    is_winner = (target_test == 1)
    winner_predicted_probabilities = integer_predicted_probabilities[is_winner]
    #winner_target_test = target_test[is_winner]
    loser_predicted_probabilities = integer_predicted_probabilities[is_winner == False]
    #loser_target_test = target_test[is_winner == False]
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ax = plt.figure().add_subplot(111)
    winner_hist,bin_edges = np.histogram(winner_predicted_probabilities,bins=100)
    loser_hist,bin_edges = np.histogram(loser_predicted_probabilities,bins=bin_edges)
    bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])
    bin_width = bin_edges[1]-bin_edges[0]
    ax.bar(bin_centers/num_mult,winner_hist, align='center', width=bin_width/num_mult, color='green', alpha=0.5)
    ax.bar(bin_centers/num_mult,loser_hist, align='center', width=bin_width/num_mult, color='red', alpha=0.5)
    ax.figure.savefig('test.png')
    import sys
    sys.exit(0)
    
    # import scipy
    # print("woo", len(winner_target_test), len(loser_target_test))
    # winner_kernel = scipy.stats.gaussian_kde(winner_predicted_probabilities)
    # print("one")
    # loser_kernel = scipy.stats.gaussian_kde(loser_predicted_probabilities)
    # print("two")
    # import matplotlib.pyplot as plt
    # ax = plt.figure().add_subplot(111)
    # xvals = np.linspace(0,1,100)
    # ax.plot(xvals,winner_kernel(xvals), ls='-', color='green', lw=3)
    # ax.plot(xvals,loser_kernel(xvals), ls='-', color='red', lw=3)
    # ax.figure.savefig('test.png')
    # print(np.unique(target_test))
    return 0
    

def make_model(n_neighbors_list=cf.DEFAULT_N_NEIGHBORS_LIST, test_frac=0.2, n_bootstrap=20):
    '''
    A wrapper that loads data, scales it, then finds the best fitting model.
    See the documentation for data_info(), rescale_data(), and fit_model() for
    info about specific parameters.
    '''
    data_info_dict = load_data()
    scaled_data = rescale_data(data_info_dict['data'])
    model_info_dict = fit_model(scaled_data, n_neighbors_list=n_neighbors_list,
                                test_frac=test_frac,
                                n_bootstrap=n_bootstrap)

    result_dict = {'seasons': data_info_dict['loaded_seasons'],
                'fit_model': model_info_dict['model'],
                'bootstrapped_models': model_info_dict['bootstrapped_model_list']}

    # with open(cf.MODEL_FILENAME,'wb') as model_file:
    #     pickle.dump(result_dict,model_file)
    joblib.dump(result_dict,cf.MODEL_FILENAME,compress=0)
        


if __name__ == "__main__":
    np.random.seed(891)
    make_model(n_bootstrap=1)
    #print(knn.predict_proba(scaled_data[cf.DATA_COLUMNS[:-1]].iloc[1030:1040]))
