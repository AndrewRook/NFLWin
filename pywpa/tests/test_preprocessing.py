from __future__ import print_function, division

import pandas as pd
import pytest
from sklearn.utils.validation import NotFittedError

from pywpa import preprocessing

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
        
