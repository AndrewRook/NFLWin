from __future__ import print_function, division

import os
import collections

import numpy as np
import pandas as pd
import pytest

from nflwin import model

class TestDefaults(object):
    """Tests for defaults."""

    def test_column_descriptions_set(self):
        wpmodel = model.WPModel()
        assert isinstance(wpmodel.column_descriptions, collections.Mapping)
        

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
        

class TestModelIO(object):
    """Tests functions that deal with model saving and loading"""

    def teardown_method(self, method):

        try:
            os.remove(self.expected_path)
        except OSError:
            pass

    def test_model_save_default(self):
        instance = model.WPModel()
        model_name = "test_model_asljasljt.nflwin"
        instance._default_model_filename = model_name

        self.expected_path = os.path.join(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                , model_name)

        assert os.path.isfile(self.expected_path) is False

        instance.save_model()

        assert os.path.isfile(self.expected_path) is True
        
    def test_model_save_specified(self):
        instance = model.WPModel()
        model_name = "test_model_qerooiua.nflwin"

        self.expected_path = os.path.join(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                , model_name)

        assert os.path.isfile(self.expected_path) is False

        instance.save_model(filename=model_name)

        assert os.path.isfile(self.expected_path) is True

    def test_model_load_default(self):
        instance = model.WPModel()
        model_name = "test_model_asljasljt.nflwin"
        instance._default_model_filename = model_name

        self.expected_path = os.path.join(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                , model_name)

        assert os.path.isfile(self.expected_path) is False

        instance.save_model()

        WPModel_class = model.WPModel
        WPModel_class._default_model_filename = model_name

        loaded_instance = WPModel_class.load_model()

        assert isinstance(loaded_instance, model.WPModel)
        
    def test_model_load_specified(self):
        instance = model.WPModel()
        model_name = "test_model_qerooiua.nflwin"

        self.expected_path = os.path.join(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
                , model_name)

        assert os.path.isfile(self.expected_path) is False

        instance.save_model(filename=model_name)

        loaded_instance = model.WPModel.load_model(filename=model_name)

        assert isinstance(loaded_instance, model.WPModel)
        
        
