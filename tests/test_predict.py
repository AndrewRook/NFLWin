from __future__ import print_function, division

import pandas as pd

from PyWPA import predict

class TestComputeWP(object):

    @classmethod
    def setup_class(cls):
        cls.model = predict.load_model()

    @classmethod
    def teardown_class(cls):
        cls.model = None

    def test_wp_dict(self):
        play_dict = {
            "quarter": 4,
            "time_remaining": 180,
            "score_diff": -4, #Offense winning is positive
            "is_offense_home": True,
            "down": 2,
            "distance": 7,
            "field_position": 80,
        }

        prediction = predict.compute_wp(self.model, play_dict)
        assert prediction['WP'] >= 0 and prediction['WP'] <= 1

    def test_wp_df(self):
        plays_dict = {
            "quarter": [1,2,3,4,5],
            "time_remaining": [1000,400,0,250,800],
            "score_diff": [10,5,0,-5,10], #Offense winning is positive
            "is_offense_home": [True, False, True, False, True],
            "down": [4,3,2,1,3],
            "distance": [1,5,10,15,20],
            "field_position": [5,20,50,80,95],
        }
        plays_df = pd.DataFrame.from_dict(plays_dict)

        prediction = predict.compute_wp(self.model, plays_df)
        assert ((prediction['WP'] >= 0) & (prediction['WP'] <= 1)).all()
            
