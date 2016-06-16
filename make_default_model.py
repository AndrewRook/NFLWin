"""A simple script to create, train, validate, and save the default model"""
from __future__ import division, print_function

import time

from nflwin import model

def main():
    start = time.time()
    win_probability_model = model.WPModel()
    win_probability_model.train_model(training_seasons=[2009, 2010, 2011, 2012, 2013, 2014])
    print("Took {0:.2f}s to build model".format(time.time() - start))
    
    start = time.time()
    combined_pvalue = win_probability_model.validate_model(validation_seasons=[2015])
    print("Took {0:.2f}s to validate model, with combined p_value of {1:.2f}".format(time.time() - start, combined_pvalue))
    
    win_probability_model.save_model()




if __name__ == "__main__":
    main()
