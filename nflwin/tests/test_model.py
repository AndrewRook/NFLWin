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
        
class TestModelTrain(object):
    """Tests for the train_model method."""

    def test_bad_string(self):
        wpmodel = model.WPModel()
        with pytest.raises(ValueError):
            wpmodel.train_model(source_data="this is a bad string")

    def test_dataframe_input(self):
        wpmodel = model.WPModel()
        test_data = {'offense_won': {0: True, 1: False, 2: False,
                                     3: False, 4: False, 5: True,
                                     6: True, 7: True, 8: True, 9: False},
                     'home_team': {0: 'NYG', 1: 'NYG', 2: 'NYG', 3: 'NYG',
                                   4: 'NYG', 5: 'NYG', 6: 'NYG', 7: 'NYG',
                                   8: 'NYG', 9: 'NYG'},
                     'away_team': {0: 'DAL', 1: 'DAL', 2: 'DAL', 3: 'DAL',
                                   4: 'DAL', 5: 'DAL', 6: 'DAL', 7: 'DAL',
                                   8: 'DAL', 9: 'DAL'},
                     'gsis_id': {0: '2012090500', 1: '2012090500', 2: '2012090500',
                                 3: '2012090500', 4: '2012090500', 5: '2012090500',
                                 6: '2012090500', 7: '2012090500', 8: '2012090500',
                                 9: '2012090500'},
                     'play_id': {0: 35, 1: 57, 2: 79, 3: 103, 4: 125, 5: 150,
                                 6: 171, 7: 190, 8: 212, 9: 252},
                     'seconds_elapsed': {0: 0.0, 1: 4.0, 2: 11.0, 3: 55.0, 4: 62.0,
                                         5: 76.0, 6: 113.0, 7: 153.0, 8: 159.0, 9: 171.0},
                      'down': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 1, 6: 2, 7: 3, 8: 4, 9: 1},
                      'curr_home_score': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
                      'offense_team': {0: 'DAL', 1: 'NYG', 2: 'NYG', 3: 'NYG',
                                       4: 'NYG', 5: 'DAL', 6: 'DAL', 7: 'DAL',
                                       8: 'DAL', 9: 'NYG'},
                      'curr_away_score': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0},
                      'yardline': {0: -15.0, 1: -34.0, 2: -34.0, 3: -29.0,
                                   4: -29.0, 5: -26.0, 6: -23.0, 7: -31.0, 8: -31.0, 9: -37.0},
                      'drive_id': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2, 9: 3},
                      'yards_to_go': {0: 0, 1: 10, 2: 10, 3: 5, 4: 5, 5: 10, 6: 7, 7: 15, 8: 15, 9: 10},
                      'quarter': {0: 'Q1', 1: 'Q1', 2: 'Q1', 3: 'Q1', 4: 'Q1',
                                  5: 'Q1', 6: 'Q1', 7: 'Q1', 8: 'Q1', 9: 'Q1'}
                    }
        test_df = pd.DataFrame(test_data)
        wpmodel.train_model(source_data=test_df)

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
        
        
