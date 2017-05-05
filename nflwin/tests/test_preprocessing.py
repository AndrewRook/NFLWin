from __future__ import print_function, division

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import NotFittedError
from sklearn.pipeline import Pipeline

from nflwin import preprocessing

class TestPipelines(object):
    """Testing if pipelining cleaning steps works."""
    def test_map_to_int_to_onehot(self):
        fit_df = pd.DataFrame({"quarter": ["Q1", "Q1", "Q1", "Q2", "Q2"]})
        transform_df = fit_df.copy()

        mti = preprocessing.MapToInt("quarter", copy=True)
        ohe = preprocessing.OneHotEncoderFromDataFrame(categorical_feature_names=["quarter"], copy=True)
        pipe = Pipeline(steps=[("one", mti), ("two", ohe)])
        pipe.fit(fit_df)
        output_df = pipe.transform(transform_df)

        expected_df = pd.DataFrame({"onehot_col1": [1.0, 1, 1, 0, 0], "onehot_col2": [0.0, 0, 0, 1, 1]})
        pd.util.testing.assert_frame_equal(output_df, expected_df)

class TestComputeElapsedTime(object):
    """Testing if we can properly map quarters and time elapsed to a total time elapsed."""

    def test_bad_quarter_colname_produces_error(self):
        input_df = pd.DataFrame({"blahblahblah": ["Q1", "Q2", "Q3", "Q4", "OT"],
                                 "time_elapsed": [200, 0, 50, 850, 40]})
        cet = preprocessing.ComputeElapsedTime("quarter", "time_elapsed")
        cet.fit(input_df)

        with pytest.raises(KeyError):
            cet.transform(input_df)

    def test_bad_time_elapsed_colname_produces_error(self):
        input_df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "OT"],
                                 "blahblahblah": [200, 0, 50, 850, 40]})
        cet = preprocessing.ComputeElapsedTime("quarter", "time_elapsed")
        cet.fit(input_df)

        with pytest.raises(KeyError):
            cet.transform(input_df)

    def test_preexisting_output_colname_produces_error(self):
        input_df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "OT"],
                                 "time_elapsed": [200, 0, 50, 850, 40],
                                 "total_time_elapsed": [0, 0, 0, 0, 0]})
        cet = preprocessing.ComputeElapsedTime("quarter", "time_elapsed",
                                               total_time_colname="total_time_elapsed")
        cet.fit(input_df)

        with pytest.raises(KeyError):
            cet.transform(input_df)

    def test_incomplete_quarter_mapping(self):
        input_df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "OT1"],
                                 "time_elapsed": [200, 0, 50, 850, 40]})
        cet = preprocessing.ComputeElapsedTime("quarter", "time_elapsed",
                                               quarter_to_second_mapping={
                                                   "Q1": 0,
                                                   "Q2": 900,
                                                   "Q4": 2700,
                                                   "OT1":3600} )
        cet.fit(input_df)

        with pytest.raises(TypeError):
            cet.transform(input_df)

    def test_simple_working_case(self):
        input_df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "OT"],
                                 "time_elapsed": [200, 0, 50, 850, 40]})
        cet = preprocessing.ComputeElapsedTime("quarter", "time_elapsed")
        cet.fit(input_df)

        transformed_df = cet.transform(input_df)
        expected_df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "OT"],
                                    "time_elapsed": [200, 0, 50, 850, 40],
                                    "total_elapsed_time": [200, 900, 1850, 3550, 3640]})
        pd.util.testing.assert_frame_equal(transformed_df, expected_df)

    def test_inplace_transform(self):
        input_df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "OT"],
                                 "time_elapsed": [200, 0, 50, 850, 40]})
        cet = preprocessing.ComputeElapsedTime("quarter", "time_elapsed", copy=False)
        cet.fit(input_df)

        cet.transform(input_df)
        expected_df = pd.DataFrame({"quarter": ["Q1", "Q2", "Q3", "Q4", "OT"],
                                    "time_elapsed": [200, 0, 50, 850, 40],
                                    "total_elapsed_time": [200, 900, 1850, 3550, 3640]})
        pd.util.testing.assert_frame_equal(input_df, expected_df)

    def test_custom_mapping(self):
        input_df = pd.DataFrame({"quarter": ["quarter1", "Q2", "Q3", "Q4", "OT1"],
                                 "time_elapsed": [200, 0, 50, 850, 40]})
        cet = preprocessing.ComputeElapsedTime("quarter", "time_elapsed",
                                               quarter_to_second_mapping={
                                                   "quarter1": 0,
                                                   "Q2": 500,
                                                   "Q3": 1800,
                                                   "Q4": 2700,
                                                   "OT1":3600})
        cet.fit(input_df)

        transformed_df = cet.transform(input_df)
        expected_df = pd.DataFrame({"quarter": ["quarter1", "Q2", "Q3", "Q4", "OT1"],
                                    "time_elapsed": [200, 0, 50, 850, 40],
                                    "total_elapsed_time": [200, 500, 1850, 3550, 3640]})
        pd.util.testing.assert_frame_equal(transformed_df, expected_df)
        

class TestComputeIfOffenseIsHome(object):
    """Testing if we can correctly compute if the offense is the home team."""

    def test_bad_offense_colname_produces_error(self):
        input_df = pd.DataFrame({"home_team": ["a", "a", "a"],
                                 "blahblahblah": ["a", "b", "a"]})
        ciow = preprocessing.ComputeIfOffenseIsHome("offense_team", "home_team")
        ciow.fit(input_df)

        with pytest.raises(KeyError):
            ciow.transform(input_df)
            
    def test_bad_home_team_colname_produces_error(self):
        input_df = pd.DataFrame({"blahblahblah": ["a", "a", "a"],
                                 "offense_team": ["a", "b", "a"]})
        ciow = preprocessing.ComputeIfOffenseIsHome("offense_team", "home_team")
        ciow.fit(input_df)

        with pytest.raises(KeyError):
            ciow.transform(input_df)
            
    def test_existing_offense_home_team_colname_produces_error(self):
        input_df = pd.DataFrame({"home_team": ["a", "a", "a"],
                                 "offense_team": ["a", "b", "a"]})
        ciow = preprocessing.ComputeIfOffenseIsHome("offense_team", "home_team",
                                                 offense_home_team_colname="home_team")
        ciow.fit(input_df)

        with pytest.raises(KeyError):
            ciow.transform(input_df)

    def test_correct_answer_with_copy(self):
        input_df = pd.DataFrame({"home_team": ["a", "a", "a"],
                                 "offense_team": ["a", "b", "a"]})
        expected_input_df = input_df.copy()
        expected_transformed_df = pd.DataFrame({"home_team": ["a", "a", "a"],
                                 "offense_team": ["a", "b", "a"],
                                 "offense_home_team": [True, False, True]})
        ciow = preprocessing.ComputeIfOffenseIsHome("offense_team", "home_team",
                                                 offense_home_team_colname="offense_home_team",
                                                 copy=True)
        transformed_df = ciow.transform(input_df)
        pd.util.testing.assert_frame_equal(input_df.sort_index(axis=1), expected_input_df.sort_index(axis=1))
        pd.util.testing.assert_frame_equal(transformed_df.sort_index(axis=1), expected_transformed_df.sort_index(axis=1))

    def test_correct_answer_without_copy(self):
        input_df = pd.DataFrame({"home_team": ["a", "a", "a"],
                                 "offense_team": ["a", "b", "a"]})
        expected_transformed_df = pd.DataFrame({"home_team": ["a", "a", "a"],
                                 "offense_team": ["a", "b", "a"],
                                 "offense_home_team": [True, False, True]})
        ciow = preprocessing.ComputeIfOffenseIsHome("offense_team", "home_team",
                                                 offense_home_team_colname="offense_home_team",
                                                 copy=False)
        ciow.transform(input_df)
        pd.util.testing.assert_frame_equal(input_df.sort_index(axis=1), expected_transformed_df.sort_index(axis=1))


class TestConvertToOffenseDefense(object):

    @pytest.mark.parametrize("home_colname,away_colname,offense_home_colname", [
        ("one", "two", "woo"),
        ("one", "woo", "two"),
        ("woo", "one", "two")
    ])
    def test_non_existent_colnames_raise_error(self, home_colname, away_colname, offense_home_colname):
        input_data = pd.DataFrame({"one": [1, 2, 3],
                                   "two": [4, 5, 6],
                                   "three": [7, 8, 9]})
        ctod = preprocessing.ConvertToOffenseDefense(home_colname, away_colname,
                                                     offense_home_colname, "blah", "blahblah")
        with pytest.raises(KeyError):
            ctod.transform(input_data)

    @pytest.mark.parametrize("new_offense_colname,new_defense_colname", [
        ("real", "fake"),
        ("fake", "real")
    ])
    def test_existent_colnames_raise_error(self, new_offense_colname, new_defense_colname):
        input_data = pd.DataFrame({"home_colname": [1, 2, 3],
                                   "away_colname": [4, 5, 6],
                                   "offense_home_colname": [1, 0, 1],
                                   "real": [1, 2, 3]})
        ctod = preprocessing.ConvertToOffenseDefense("home_colname", "away_colname", "offense_home_colname",
                                                     new_offense_colname, new_defense_colname)
        with pytest.raises(KeyError):
            ctod.transform(input_data)

    def test_transform_works(self):
        home_colname = "home_colname"
        away_colname = "away_colname"
        new_offense_colname = "new_offense_colname"
        new_defense_colname = "new_defense_colname"
        input_data = pd.DataFrame({home_colname: [1, 2, 3, 4],
                                   away_colname: [5, 6, 7, 8],
                                   "offense_home_colname": [False, True, False, False]})
        ctod = preprocessing.ConvertToOffenseDefense(home_colname, away_colname, "offense_home_colname",
                                                     new_offense_colname, new_defense_colname)
        transformed_data = ctod.transform(input_data)
        expected_data = pd.DataFrame({new_offense_colname: [5, 2, 7, 8],
                                      new_defense_colname: [1, 6, 3, 4],
                                      "offense_home_colname": [False, True, False, False]})
        pd.util.testing.assert_frame_equal(transformed_data.sort_index(axis=1),
                                           expected_data.sort_index(axis=1))

    def test_transform_inplace(self):
        home_colname = "home_colname"
        away_colname = "away_colname"
        new_offense_colname = "new_offense_colname"
        new_defense_colname = "new_defense_colname"
        input_data = pd.DataFrame({home_colname: [1, 2, 3, 4],
                                   away_colname: [5, 6, 7, 8],
                                   "offense_home_colname": [False, True, False, False]})
        ctod = preprocessing.ConvertToOffenseDefense(home_colname, away_colname, "offense_home_colname",
                                                     new_offense_colname, new_defense_colname, copy=False)
        ctod.transform(input_data)
        expected_data = pd.DataFrame({new_offense_colname: [5, 2, 7, 8],
                                      new_defense_colname: [1, 6, 3, 4],
                                      "offense_home_colname": [False, True, False, False]})
        pd.util.testing.assert_frame_equal(input_data.sort_index(axis=1),
                                           expected_data.sort_index(axis=1))



class TestMapToInt(object):
    """Testing if the integer mapper works."""

    def test_fit_bad_colname_produces_error(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", "one", "one"]})
        mti = preprocessing.MapToInt("blahblahblah")

        with pytest.raises(KeyError):
            mti.fit(input_df)
        

    def test_mapping_without_nans(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", "one", "one"]})
        mti = preprocessing.MapToInt("one")
        mti.fit(input_df)
        expected_output = {"one": 0, "two": 1, "four": 2, "six": 3}
        assert mti.mapping == expected_output

    def test_mapping_with_nans(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", np.nan, "one", "one"]})
        mti = preprocessing.MapToInt("one")
        mti.fit(input_df)
        expected_output = {"one": 0, "two": 1, "four": 2, "six": 3}
        assert mti.mapping == expected_output

    def test_transform_before_fit_produces_error(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", "one", "one"]})
        mti = preprocessing.MapToInt("one")

        with pytest.raises(NotFittedError):
            mti.transform(input_df)

    def test_transform_bad_colname_produces_error(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", "one", "one"]})
        mti = preprocessing.MapToInt("one")
        mti.fit(input_df)
        transform_df = pd.DataFrame({"blahblahblah": ["one", "two", "one", "four",
                                                      "six", "two", "one", "one"]})
        with pytest.raises(KeyError):
            mti.transform(transform_df)

    def test_transform_without_nans(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", "one", "one"]})
        mti = preprocessing.MapToInt("one")
        mti.fit(input_df)
        transformed_df = mti.transform(input_df)
        expected_df = pd.DataFrame({"one": [0, 1, 0, 2, 3, 1, 0, 0]})
        pd.util.testing.assert_frame_equal(transformed_df, expected_df)

    def test_transform_with_nans(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", np.nan, "one"]})
        mti = preprocessing.MapToInt("one")
        mti.fit(input_df)
        transformed_df = mti.transform(input_df)
        expected_df = pd.DataFrame({"one": [0, 1, 0, 2, 3, 1, np.nan, 0]})
        pd.util.testing.assert_frame_equal(transformed_df, expected_df)

    def test_transform_inplace(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", "one", "one"]})
        mti = preprocessing.MapToInt("one", copy=False)
        mti.fit(input_df)
        mti.transform(input_df)
        expected_df = pd.DataFrame({"one": [0, 1, 0, 2, 3, 1, 0, 0]})
        pd.util.testing.assert_frame_equal(input_df, expected_df)

    def test_transform_copy(self):
        input_df = pd.DataFrame({"one": ["one", "two", "one", "four",
                                         "six", "two", "one", "one"]})
        expected_df = input_df.copy()
        mti = preprocessing.MapToInt("one", copy=True)
        mti.fit(input_df)
        transformed_data = mti.transform(input_df)
        pd.util.testing.assert_frame_equal(input_df, expected_df)
        
        
        

class TestOneHotEncoderFromDataFrame(object):
    """Testing if the one-hot encoder wrapper works."""

    def setup_method(self, method):
        self.data = pd.DataFrame({"one": [1, 2, 3, 1],
                                  "two": [2, 2, 2, 5],
                                  "three": [0, 5, 0, 5]})
        self.data = self.data[["one", "two", "three"]]

    def test_correct_dtype_passed(self):
        ohe = preprocessing.OneHotEncoderFromDataFrame(dtype=np.int)
        assert ohe.dtype == np.int

    def test_correct_handle_unknown_string_passed(self):
        ohe = preprocessing.OneHotEncoderFromDataFrame(handle_unknown="ignore")
        assert ohe.handle_unknown == "ignore"

    def test_encode_all_columns(self):
        ohe = preprocessing.OneHotEncoderFromDataFrame(categorical_feature_names="all")
        ohe.fit(self.data)
        transformed_data = ohe.transform(self.data)
        expected_data = pd.DataFrame({"onehot_col1": [1., 0, 0, 1],
                                      "onehot_col2": [0., 1, 0, 0],
                                      "onehot_col3": [0., 0, 1, 0],
                                      "onehot_col4": [1., 1, 1, 0],
                                      "onehot_col5": [0., 0, 0, 1],
                                      "onehot_col6": [1., 0, 1, 0],
                                      "onehot_col7": [0., 1, 0, 1]})

        pd.util.testing.assert_frame_equal(transformed_data.sort_index(axis=1),
                                           expected_data.sort_index(axis=1))

    def test_encode_some_columns(self):
        ohe = preprocessing.OneHotEncoderFromDataFrame(categorical_feature_names=["one", "three"])
        ohe.fit(self.data)
        transformed_data = ohe.transform(self.data)
        expected_data = pd.DataFrame({"two": [2, 2, 2, 5],
                                      "onehot_col1": [1., 0, 0, 1],
                                      "onehot_col2": [0., 1, 0, 0],
                                      "onehot_col3": [0., 0, 1, 0],
                                      "onehot_col4": [1., 0, 1, 0],
                                      "onehot_col5": [0., 1, 0, 1]})

        pd.util.testing.assert_frame_equal(transformed_data.sort_index(axis=1),
                                           expected_data.sort_index(axis=1))

    def test_copy_data_works(self):
        ohe = preprocessing.OneHotEncoderFromDataFrame(categorical_feature_names=["one", "three"],
                                                       copy=True)
        ohe.fit(self.data)
        transformed_data = ohe.transform(self.data)
        expected_data = pd.DataFrame({"one": [1, 2, 3, 1],
                                      "two": [2, 2, 2, 5],
                                      "three": [0, 5, 0, 5]})

        pd.util.testing.assert_frame_equal(self.data.sort_index(axis=1),
                                           expected_data.sort_index(axis=1))
        

    def test_inplace_transform_works(self):
        ohe = preprocessing.OneHotEncoderFromDataFrame(categorical_feature_names=["one", "three"],
                                                       copy=False)
        data = self.data.copy()
        ohe.fit(self.data)
        ohe.transform(self.data)
        expected_data = pd.DataFrame({"two": [2, 2, 2, 5],
                                      "onehot_col1": [1., 0, 0, 1],
                                      "onehot_col2": [0., 1, 0, 0],
                                      "onehot_col3": [0., 0, 1, 0],
                                      "onehot_col4": [1., 0, 1, 0],
                                      "onehot_col5": [0., 1, 0, 1]})

        pd.util.testing.assert_frame_equal(self.data.sort_index(axis=1),
                                           expected_data.sort_index(axis=1))

    def test_encoding_subset_columns(self):
        ohe = preprocessing.OneHotEncoderFromDataFrame(categorical_feature_names=["one", "three"],
                                                       copy=True)
        shifted_data = self.data[2:]
        ohe.fit(shifted_data)
        transformed_data = ohe.transform(shifted_data)
        self.data = pd.DataFrame({"one": [1, 2, 3, 1],
                                  "two": [2, 2, 2, 5],
                                  "three": [0, 5, 0, 5]})
        expected_data = pd.DataFrame({"two": [2, 5],
                                      "onehot_col1": [0., 1],
                                      "onehot_col2": [1., 0],
                                      "onehot_col3": [1., 0],
                                      "onehot_col4": [0., 1]},
                                      index=[2, 3])
        print(transformed_data)
        print(expected_data)
        pd.util.testing.assert_frame_equal(transformed_data.sort_index(axis=1),
                                           expected_data.sort_index(axis=1))
        
        
        

class TestCreateScoreDifferential(object):
    """Testing if score differentials are properly created."""

    def test_bad_home_score_colname(self):
        csd = preprocessing.CreateScoreDifferential("badcol", "away_score", "offense_home")
        data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                             "away_score": [10, 0, 5, 15],
                             "offense_home": [True, True, True, True]})
        with pytest.raises(KeyError):
            csd.transform(data)
            
    def test_bad_away_score_colname(self):
        csd = preprocessing.CreateScoreDifferential("home_score", "badcol", "offense_home")
        data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                             "away_score": [10, 0, 5, 15],
                             "offense_home": [True, True, True, True]})
        with pytest.raises(KeyError):
            csd.fit(data)
            csd.transform(data)
            
    def test_bad_offense_home_colname(self):
        csd = preprocessing.CreateScoreDifferential("home_score", "away_score", "badcol")
        data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                             "away_score": [10, 0, 5, 15],
                             "offense_home": [True, True, True, True]})
        with pytest.raises(KeyError):
            csd.fit(data)
            csd.transform(data)
            
    def test_differential_column_already_exists(self):
        csd = preprocessing.CreateScoreDifferential("home_score",
                                                    "away_score",
                                                    "offense_home",
                                                    score_differential_colname="used_col")
        data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                             "away_score": [10, 0, 5, 15],
                             "offense_home": [True, True, True, True],
                             "used_col": [0, 0, 0, 0]})
        with pytest.raises(KeyError):
            csd.fit(data)
            csd.transform(data)

    def test_differential_works_offense_is_home(self):
        csd = preprocessing.CreateScoreDifferential("home_score",
                                                    "away_score",
                                                    "offense_home",
                                                    score_differential_colname="score_diff")
        input_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                   "away_score": [10, 0, 5, 15],
                                   "offense_home": [True, True, True, True]})
        expected_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                      "away_score": [10, 0, 5, 15],
                                      "offense_home": [True, True, True, True],
                                      "score_diff": [-9, 2, -2, -11]})
        
        csd.fit(input_data)
        transformed_data = csd.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_data.sort_index(axis=1),
                                           transformed_data.sort_index(axis=1))

    def test_differential_works_offense_is_away(self):
        csd = preprocessing.CreateScoreDifferential("home_score",
                                                    "away_score",
                                                    "offense_home",
                                                    score_differential_colname="score_diff")
        input_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                   "away_score": [10, 0, 5, 15],
                                   "offense_home": [False, False, False, False]})
        expected_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                      "away_score": [10, 0, 5, 15],
                                      "offense_home": [False, False, False, False],
                                      "score_diff": [9, -2, 2, 11]})
        
        csd.fit(input_data)
        transformed_data = csd.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_data.sort_index(axis=1),
                                           transformed_data.sort_index(axis=1))

    def test_differential_works_offense_is_mix(self):
        csd = preprocessing.CreateScoreDifferential("home_score",
                                                    "away_score",
                                                    "offense_home",
                                                    score_differential_colname="score_diff")
        input_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                   "away_score": [10, 0, 5, 15],
                                   "offense_home": [True, True, False, False]})
        expected_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                      "away_score": [10, 0, 5, 15],
                                      "offense_home": [True, True, False, False],
                                      "score_diff": [-9, 2, 2, 11]})
        
        csd.fit(input_data)
        transformed_data = csd.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_data.sort_index(axis=1),
                                           transformed_data.sort_index(axis=1))

    def test_differential_with_copied_data(self):
        csd = preprocessing.CreateScoreDifferential("home_score",
                                                    "away_score",
                                                    "offense_home",
                                                    score_differential_colname="score_diff",
                                                    copy=True)
        input_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                   "away_score": [10, 0, 5, 15],
                                   "offense_home": [True, True, True, True]})
        expected_input_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                            "away_score": [10, 0, 5, 15],
                                            "offense_home": [True, True, True, True]})
        expected_transformed_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                      "away_score": [10, 0, 5, 15],
                                      "offense_home": [True, True, True, True],
                                      "score_diff": [-9, 2, -2, -11]})
        
        csd.fit(input_data)
        transformed_data = csd.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_input_data.sort_index(axis=1),
                                           input_data.sort_index(axis=1))
        pd.util.testing.assert_frame_equal(expected_transformed_data.sort_index(axis=1),
                                           transformed_data.sort_index(axis=1))
        
    def test_differential_with_inplace_data(self):
        csd = preprocessing.CreateScoreDifferential("home_score",
                                                    "away_score",
                                                    "offense_home",
                                                    score_differential_colname="score_diff",
                                                    copy=False)
        input_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                   "away_score": [10, 0, 5, 15],
                                   "offense_home": [True, True, True, True]})
        expected_data = pd.DataFrame({"home_score": [1, 2, 3, 4],
                                      "away_score": [10, 0, 5, 15],
                                      "offense_home": [True, True, True, True],
                                      "score_diff": [-9, 2, -2, -11]})
        csd.fit(input_data)
        csd.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_data.sort_index(axis=1),
                                           input_data.sort_index(axis=1))
        
        
class TestDropColumns(object):

    def test_bad_colname(self):
        dc = preprocessing.DropColumns(["real", "fake"])
        data = pd.DataFrame({"real": [1, 2, 3],
                             "also_real": [4, 5, 6]})
        with pytest.raises(KeyError):
            dc.transform(data)

    def test_delete_single_column(self):
        dc = preprocessing.DropColumns(["one"])
        input_data = pd.DataFrame({"one": [1, 2, 3],
                                   "two": [4, 5, 6],
                                   "three": [7, 8, 9]})
        transformed_data = dc.transform(input_data)
        expected_data = pd.DataFrame({"two": [4, 5, 6],
                                      "three": [7, 8, 9]})
        pd.util.testing.assert_frame_equal(expected_data.sort_index(axis=1),
                                           transformed_data.sort_index(axis=1))

    def test_delete_multiple_columns(self):
        dc = preprocessing.DropColumns(["one", "two"])
        input_data = pd.DataFrame({"one": [1, 2, 3],
                                   "two": [4, 5, 6],
                                   "three": [7, 8, 9]})
        transformed_data = dc.transform(input_data)
        expected_data = pd.DataFrame({"three": [7, 8, 9]})
        pd.util.testing.assert_frame_equal(expected_data.sort_index(axis=1),
                                           transformed_data.sort_index(axis=1))

    def test_delete_inplace(self):
        dc = preprocessing.DropColumns(["one", "two"], copy=False)
        input_data = pd.DataFrame({"one": [1, 2, 3],
                                   "two": [4, 5, 6],
                                   "three": [7, 8, 9]})
        dc.transform(input_data)
        expected_data = pd.DataFrame({"three": [7, 8, 9]})
        pd.util.testing.assert_frame_equal(expected_data.sort_index(axis=1),
                                           input_data.sort_index(axis=1))


class TestConvertToNumpy(object):
    def test_transform_called_before_fit(self):
        ctn = preprocessing.ConvertToNumpy(np.float)
        data = pd.DataFrame()
        with pytest.raises(NotFittedError):
            ctn.transform(data)

    def test_transform_data_has_wrong_columns(self):
        ctn = preprocessing.ConvertToNumpy(np.float)
        fit_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        transform_data = pd.DataFrame({
            "one": [1, 2, 3],
            "three": [7, 8, 9]
        })
        ctn.fit(fit_data)
        with pytest.raises(KeyError):
            ctn.transform(transform_data)

    def test_transform_floats(self):
        ctn = preprocessing.ConvertToNumpy(np.float)
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4.5, 5.6, 6.7]
        })
        ctn.fit(input_data[["one", "two"]])
        transformed_data = ctn.transform(input_data)
        expected_data = np.array([[1.0, 4.5],
                                  [2.0, 5.6],
                                  [3.0, 6.7]])
        np.testing.assert_allclose(transformed_data, expected_data)

    def test_transform_ints(self):
        ctn = preprocessing.ConvertToNumpy(np.int)
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4.5, 5.6, 6.7]
        })
        ctn.fit(input_data[["one", "two"]])
        transformed_data = ctn.transform(input_data)
        expected_data = np.array([[1, 4],
                                  [2, 5],
                                  [3, 6]])
        np.testing.assert_allclose(transformed_data, expected_data)

    def test_transform_ordering(self):
        ctn = preprocessing.ConvertToNumpy(np.float)
        input_data = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4.5, 5.6, 6.7],
            "c": [7, 8, 9]
        })
        ctn.fit(input_data[["c", "b", "a"]])
        transformed_data = ctn.transform(input_data)
        expected_data = np.array([[7.0, 4.5, 1.0],
                                  [8.0, 5.6, 2.0],
                                  [9.0, 6.7, 3.0]])
        np.testing.assert_allclose(transformed_data, expected_data)

        ctn.fit(input_data[["b", "a", "c"]])
        transformed_data = ctn.transform(input_data)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(transformed_data, expected_data)
            
        
        

class TestCheckColumnNames(object):
    """Testing whether column names are properly checked."""

    def test_transform_called_before_fit(self):
        ccn = preprocessing.CheckColumnNames()
        data = pd.DataFrame()

        with pytest.raises(NotFittedError):
            ccn.transform(data)

    def test_transform_data_has_wrong_columns(self):
        ccn = preprocessing.CheckColumnNames()
        input_data = pd.DataFrame({"one": [1, 2],
                                   "two": [3, 4]})
        ccn.fit(input_data)
        test_data = pd.DataFrame({"one": [1, 2],
                                  "three": [3, 4]})

        with pytest.raises(KeyError):
            ccn.transform(test_data)

    def test_transform_reorders_columns(self):
        ccn = preprocessing.CheckColumnNames()
        input_data = pd.DataFrame({"one": [1, 2],
                                   "two": [3, 4],
                                   "three": [5, 6]})
        test_data = pd.DataFrame({"one": [7, 8],
                                   "two": [9, 10],
                                   "three": [11, 12]})
        expected_data = test_data.copy()
        #Ensure columns are in a particular order:
        input_data = input_data[["one", "two", "three"]]
        test_data = test_data[["two", "one", "three"]]
        expected_data = expected_data[["one", "two", "three"]]

        with pytest.raises(AssertionError):
            pd.util.testing.assert_frame_equal(test_data, expected_data)
        
        ccn.fit(input_data)
        pd.util.testing.assert_frame_equal(ccn.transform(test_data), expected_data)
        

    def test_transform_drops_unnecessary_columns(self):
        ccn = preprocessing.CheckColumnNames()
        input_data = pd.DataFrame({"one": [1, 2],
                                   "two": [3, 4],
                                   "three": [5, 6]})
        test_data = pd.DataFrame({"one": [7, 8],
                                   "two": [9, 10],
                                   "three": [11, 12],
                                   "four": [13, 14]})
        expected_data = pd.DataFrame({"one": [7, 8],
                                      "two": [9, 10],
                                      "three": [11, 12]})
        #Ensure columns are in a particular order:
        input_data = input_data[["one", "two", "three"]]
        expected_data = expected_data[["one", "two", "three"]]

        ccn.fit(input_data)
        pd.util.testing.assert_frame_equal(ccn.transform(test_data), expected_data)
        

    def test_transform_with_user_specified_colums(self):
        ccn = preprocessing.CheckColumnNames(column_names=["c", "b", "a"])
        input_data = pd.DataFrame({"e": [-2, -1, 0],
                                   "a": [1, 2, 3],
                                   "b": [4, 5, 6],
                                   "c": [7, 8, 9],
                                   "d": [10, 11, 12]})
        expected_data = pd.DataFrame({"c": [7, 8, 9],
                                      "b": [4, 5, 6],
                                      "a": [1, 2, 3]})
        expected_data = expected_data[["c", "b", "a"]]
        transformed_data = ccn.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_data, transformed_data)
