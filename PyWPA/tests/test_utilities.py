from __future__ import print_function, division

import os

import nfldb
from nose.tools import assert_equal
from nose.tools import raises
import pandas as pd

import pywpa.utilities as utils

class TestMakeSQLString(object):
    """testing the _make_query_string function"""

    def setUp(self):
        pass

class TestConnectDB(object):
    """testing the connect_db function"""
    def setUp(self):
        self.curr_config_home = nfldb.db._config_home
        
    def tearDown(self):
        nfldb.db._config_home = self.curr_config_home

    @raises(IOError)
    def test_no_config_error(self):
        nfldb.db._config_home = "/boogaboogabooga"

        utils.connect_db()

    def test_engine_works(self):
        engine = utils.connect_db()
        test_query = ("SELECT play.description "
                      "from play "
                      "WHERE play.gsis_id = '2009080950' AND play.play_id=721;")
        
        plays_df = pd.read_sql(test_query, engine)
        
        assert_equal(plays_df.iloc[0]['description'],
                     u'(6:55) L.White left guard for 3 yards, TOUCHDOWN.')
