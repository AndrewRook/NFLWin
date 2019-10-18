from __future__ import print_function, division

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.validation import NotFittedError
from sklearn.pipeline import Pipeline

from nflwin import preprocessing

class TestGenerateFeaturesAddFeatures:
    def test_data_added_right(self):
        feature_generator = preprocessing.GenerateFeatures(None, None, None)
        feature_generator.residual_trees_ = None # Mock this

class TestOneHotEncoding:
    def test_gets_right_unique_values(self):
        input_data = pd.DataFrame(
            {"one": ["a", "b", "banana", "b", "a"]}
        )
        cleaner = preprocessing.OneHotEncode("one")
        cleaner.fit(input_data)
        expected_output = sorted(["a", "b", "banana"])
        actual_output = sorted(cleaner.unique_values_)
        assert expected_output == actual_output

    def test_encodes_properly(self):
        input_data = pd.DataFrame(
            {"one": ["a", "b", "banana", "a"]}
        )
        cleaner = preprocessing.OneHotEncode("one")
        cleaner.fit(input_data)
        expected_output = pd.DataFrame({
            "one": ["a", "b", "banana", "a"],
            "one_a": np.array([1, 0, 0, 1]),
            "one_b": np.array([0, 1, 0, 0])
        })
        actual_output = cleaner.transform(input_data)
        pd.util.testing.assert_frame_equal(
            expected_output.reindex(sorted(expected_output.columns), axis=1),
            actual_output.reindex(sorted(actual_output.columns), axis=1),
            check_dtype=False
        )

    def test_encodes_when_category_not_present(self):
        fit_data = pd.DataFrame(
            {"one": ["a", "b", "banana", "a"]}
        )
        cleaner = preprocessing.OneHotEncode("one")
        cleaner.fit(fit_data)
        transform_data = pd.DataFrame(
            {"one": ["b"]}
        )
        expected_output = pd.DataFrame({
            "one": ["b"],
            "one_a": np.array([0], dtype=np.uint8),
            "one_b": np.array([1], dtype=np.uint8)
        })
        actual_output = cleaner.transform(transform_data)
        pd.util.testing.assert_frame_equal(
            expected_output.reindex(sorted(expected_output.columns), axis=1),
            actual_output.reindex(sorted(actual_output.columns), axis=1),
            check_dtype=False
        )

    def test_errors_when_new_category_present(self):
        fit_data = pd.DataFrame(
            {"one": ["a", "b", "banana", "a"]}
        )
        cleaner = preprocessing.OneHotEncode("one")
        cleaner.fit(fit_data)
        transform_data = pd.DataFrame(
            {"one": ["b", "apple"]}
        )
        with pytest.raises(KeyError):
            actual_output = cleaner.transform(transform_data)


class TestCalculateDerivedVariable:
    def test_simple_arithmetic(self):
        input_data = pd.DataFrame({"one": [1, 2, 3]})
        formula = "one * 3"
        expected_output = pd.DataFrame(
            {"one": [1, 2, 3],
            "two": [3., 6., 9.]}
        )
        cleaner = preprocessing.CalculateDerivedVariable(
            "two", formula
        )
        actual_output = cleaner.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_output, actual_output)
        # should work in-place:
        pd.util.testing.assert_frame_equal(input_data, expected_output)

    def test_np_function(self):
        input_data = pd.DataFrame({"one": [10, 100, 1000]})
        formula = "np.log10(one) + 3"
        expected_output = pd.DataFrame({
            "one": [10, 100, 1000],
            "two": [4., 5., 6.]
        })
        cleaner = preprocessing.CalculateDerivedVariable(
            "two", formula
        )
        actual_output = cleaner.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_output, actual_output)
        # should work in-place:
        pd.util.testing.assert_frame_equal(input_data, expected_output)

    def test_multiple_columns(self):
        input_data = pd.DataFrame({
            "one": [10, 100, 1000],
            "two": [0, 9, 12]
        })
        formula = "np.log10(one) + two"
        expected_output = pd.DataFrame({
            "one": [10, 100, 1000],
            "two": [0, 9, 12],
            "three": [1., 11., 15.]
        })
        cleaner = preprocessing.CalculateDerivedVariable(
            "three", formula
        )
        actual_output = cleaner.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_output, actual_output)
        # should work in-place:
        pd.util.testing.assert_frame_equal(input_data, expected_output)

    def test_boolean_logic(self):
        input_data = pd.DataFrame({"one": [1, 2, 3]})
        formula = "one != 2"
        expected_output = pd.DataFrame(
            {"one": [1, 2, 3],
            "two": [1., 0., 1.]}
        )
        cleaner = preprocessing.CalculateDerivedVariable(
            "two", formula
        )
        actual_output = cleaner.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_output, actual_output)
        # should work in-place:
        pd.util.testing.assert_frame_equal(input_data, expected_output)

    def test_overwrite_columns(self):
        input_data = pd.DataFrame({"one": [1, 2, 3]})
        formula = "one * 3"
        expected_output = pd.DataFrame(
            {"one": [3., 6., 9.]}
        )
        cleaner = preprocessing.CalculateDerivedVariable(
            "one", formula
        )
        actual_output = cleaner.transform(input_data)
        pd.util.testing.assert_frame_equal(expected_output, actual_output)
        # should work in-place:
        pd.util.testing.assert_frame_equal(input_data, expected_output)


class TestDataFrameToNumpy:
    def test_simple_case(self):
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        expected_output = np.array([
            [1, 4],
            [2, 5],
            [3, 6]
        ], dtype=np.float)
        cleaner = preprocessing.DataFrameToNumpy()
        cleaner.fit(input_data)
        actual_output = cleaner.transform(input_data)
        np.testing.assert_allclose(actual_output, expected_output)

    def test_dtype_int(self):
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        expected_output = np.array([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        cleaner = preprocessing.DataFrameToNumpy(dtype=np.int)
        cleaner.fit(input_data)
        actual_output = cleaner.transform(input_data)
        np.testing.assert_allclose(actual_output, expected_output)
        assert issubclass(actual_output.dtype.type, np.integer)

    def test_dtype_float(self):
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        expected_output = np.array([
            [1, 4],
            [2, 5],
            [3, 6]
        ])
        cleaner = preprocessing.DataFrameToNumpy(dtype=np.float)
        cleaner.fit(input_data)
        actual_output = cleaner.transform(input_data)
        np.testing.assert_allclose(actual_output, expected_output)
        assert issubclass(actual_output.dtype.type, np.float)

    def test_swapped_column_order(self):
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        swapped_data = pd.DataFrame({
            "two": [4, 5, 6]
        })
        swapped_data["one"] = [1, 2, 3]
        assert np.array_equal(input_data.columns, swapped_data.columns) == False
        expected_output = np.array([
            [1, 4],
            [2, 5],
            [3, 6]
        ], dtype=np.float)
        cleaner = preprocessing.DataFrameToNumpy()
        cleaner.fit(input_data)
        actual_output = cleaner.transform(swapped_data)
        np.testing.assert_allclose(actual_output, expected_output)

    def test_too_few_columns(self):
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        bad_data = pd.DataFrame({
            "one": [1, 2, 3],
            "three": [4, 5, 6]
        })
        cleaner = preprocessing.DataFrameToNumpy()
        cleaner.fit(input_data)
        with pytest.raises(KeyError):
            actual_output = cleaner.transform(bad_data)

    def test_too_many_columns(self):
        input_data = pd.DataFrame({
            "one": [1, 2, 3],
            "two": [4, 5, 6]
        })
        extra_data = pd.DataFrame({
            "one": [1, 2, 3],
            "three": [4, 5, 6],
            "two": [7, 8, 9]
        })
        expected_output = np.array([
            [1, 7],
            [2, 8],
            [3, 9]
        ], dtype=np.float)
        cleaner = preprocessing.DataFrameToNumpy()
        cleaner.fit(input_data)
        actual_output = cleaner.transform(extra_data)
        np.testing.assert_allclose(actual_output, expected_output)


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
