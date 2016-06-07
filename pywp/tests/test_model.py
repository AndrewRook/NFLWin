from __future__ import print_function, division

import numpy as np
import pandas as pd
import pytest

from pywpa import model

class TestTestDistribution(object):
    """Tests the _test_distribution static method of WPModel."""

    def test_simple_case(self):
        input_probabilities = [0.1, 0.2, 0.3]
        input_predicted_win_percents = [0.1, 0.2, 0.3]
        input_num_plays_used = [10, 10, 10]

        expected_output = 1.0

        assert (expected_output -
                model.WPModel._test_distribution(input_probabilities,
                                                 input_predicted_win_percents,
                                                 input_num_plays_used)
               ) < 1e-5

    def test_more_complicated_case(self):
        input_probabilities = [0.1, 0.2, 0.4]
        input_predicted_win_percents = [0.1, 0.2, 0.3]
        input_num_plays_used = [10, 10, 100000]

        expected_output = 0.0

        assert (expected_output -
                model.WPModel._test_distribution(input_probabilities,
                                                 input_predicted_win_percents,
                                                 input_num_plays_used)
               ) < 1e-5
        
