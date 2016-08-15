"""A simple script to create, train, validate, and save the default model"""
from __future__ import division, print_function

import datetime as dt
import time
import os

from nflwin import model

def main():
    start = time.time()
    win_probability_model = model.WPModel()
    win_probability_model.train_model(training_seasons=[2009, 2010, 2011, 2012, 2013, 2014])
    print("Took {0:.2f}s to build model".format(time.time() - start))
    
    start = time.time()
    max_deviation, residual_area = win_probability_model.validate_model(validation_seasons=[2015])
    print("Took {0:.2f}s to validate model, with a max residual of {1:.2f} and a residual area of {2:.2f}"
          .format(time.time() - start, max_deviation, residual_area))
    
    win_probability_model.save_model()

    ax = win_probability_model.plot_validation(label="max deviation={0:.2f}, \n"
                                               "residual total area={1:.2f}"
                                               "".format(max_deviation, residual_area))
    curr_datetime = dt.datetime.now()
    ax.set_title("model generated at: " + curr_datetime.strftime("%Y-%m-%d %H:%M:%S"))
    ax.legend(loc="lower right")

    this_filepath = os.path.dirname(os.path.abspath(__file__))
    save_filepath = os.path.join(this_filepath, "doc", "source", "_static", "validation_plot.png")
    ax.figure.savefig(save_filepath)


if __name__ == "__main__":
    main()
