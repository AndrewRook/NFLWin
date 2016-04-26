from __future__ import print_function, division

import os

import nfldb
from nose.tools import assert_equal
from nose.tools import assert_in
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

class TestMakeQueryString(object):
    """testing the _make_query_string function"""
    
    def test_no_args(self):
        expected_string = ("SELECT play.gsis_id, play.drive_id, "
                           "play.play_id, play.time, play.pos_team, "
                           "play.yardline, play.down, play.yards_to_go, "
                           "GREATEST((agg_play.defense_frec_tds * 6), "
                           "(agg_play.defense_int_tds * 6), "
                           "(agg_play.defense_misc_tds * 6), "
                           "(agg_play.fumbles_rec_tds * 6), "
                           "(agg_play.kicking_rec_tds * 6), "
                           "(agg_play.kickret_tds * 6), "
                           "(agg_play.passing_tds * 6), "
                           "(agg_play.puntret_tds * 6), "
                           "(agg_play.receiving_tds * 6), "
                           "(agg_play.rushing_tds * 6), "
                           "(agg_play.kicking_xpmade * 1), "
                           "(agg_play.passing_twoptm * 2), "
                           "(agg_play.receiving_twoptm * 2), "
                           "(agg_play.rushing_twoptm * 2), "
                           "(agg_play.kicking_fgm * 3), "
                           "(agg_play.defense_safe * 2)) AS play_points, "
                           "game.home_team, game.away_team, "
                           "(game.home_score > game.away_score) AS home_won "
                           "FROM play INNER JOIN agg_play "
                           "ON play.gsis_id = agg_play.gsis_id "
                           "AND play.drive_id = agg_play.drive_id "
                           "AND play.play_id = agg_play.play_id "
                           "INNER JOIN game on play.gsis_id = game.gsis_id "
                           "WHERE game.home_score != game.away_score AND game.finished = TRUE "
                           "ORDER BY play.gsis_id, play.drive_id, play.play_id;")
        assert_equal(expected_string, utils._make_query_string())

        def test_single_year(self):
            """Test that adding a single year constraint works"""
            expected_substring = ("WHERE game.home_score != game.away_score "
                                  "AND game.finished = TRUE AND "
                                  "game.season_year = 2013")
            assert_in(expected_substring,
                      utils._make_query_string(season_year=[2013]))

        def test_single_season_type(self):
            """Test that adding a single season type constraint works"""
            expected_substring = ("WHERE game.home_score != game.away_score "
                                  "AND game.finished = TRUE AND "
                                  "game.season_type = Regular")
            assert_in(expected_substring,
                      utils._make_query_string(season_type=["Regular"]))

        def test_multiple_year(self):
            """Test that adding a multiple year constraint works"""
            expected_substring = ("WHERE game.home_score != game.away_score "
                                  "AND game.finished = TRUE AND "
                                  "game.season_year in (2013, 2010)")
            assert_in(expected_substring,
                      utils._make_query_string(season_year=[2013, 2010]))

        def test_single_season_type(self):
            """Test that adding a single season type constraint works"""
            expected_substring = ("WHERE game.home_score != game.away_score "
                                  "AND game.finished = TRUE AND "
                                  "game.season_type in (Regular, Postseason")
            assert_in(expected_substring,
                      utils._make_query_string(season_type=["Regular", "Postseason"]))
            
            
