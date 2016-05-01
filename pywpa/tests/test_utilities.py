from __future__ import print_function, division

import os

import nfldb
import pandas as pd
import pytest

import pywpa.utilities as utils


class TestConnectNFLDB(object):
    """testing the connect_nfldb function"""
    def setup_method(self, method):
        self.curr_config_home = nfldb.db._config_home
        
    def teardown_method(self, method):
        nfldb.db._config_home = self.curr_config_home

    def test_no_config_error(self):
        nfldb.db._config_home = "/boogaboogabooga"

        with pytest.raises(IOError):
            utils.connect_nfldb()

    def test_engine_works(self):
        engine = utils.connect_nfldb()
        test_query = ("SELECT play.description "
                      "from play "
                      "WHERE play.gsis_id = '2009080950' AND play.play_id=721;")
        
        plays_df = pd.read_sql(test_query, engine)
        
        assert (plays_df.iloc[0]['description'] ==
                u'(6:55) L.White left guard for 3 yards, TOUCHDOWN.')

class TestMakeNFLDBQueryString(object):
    """testing the _make_nfldb_query_string function"""
    
    def test_no_args(self):
        expected_string = ("SELECT play.gsis_id, play.drive_id, "
                           "play.play_id, play.time, play.pos_team, "
                           "play.yardline, play.down, play.yards_to_go, "
                           "GREATEST("
                           "(agg_play.fumbles_rec_tds * 6), "
                           "(agg_play.kicking_rec_tds * 6), "
                           "(agg_play.passing_tds * 6), "
                           "(agg_play.receiving_tds * 6), "
                           "(agg_play.rushing_tds * 6), "
                           "(agg_play.kicking_xpmade * 1), "
                           "(agg_play.passing_twoptm * 2), "
                           "(agg_play.receiving_twoptm * 2), "
                           "(agg_play.rushing_twoptm * 2), "
                           "(agg_play.kicking_fgm * 3)) AS offense_play_points, "
                           "GREATEST("
                           "(agg_play.defense_frec_tds * 6), "
                           "(agg_play.defense_int_tds * 6), "
                           "(agg_play.defense_misc_tds * 6), "
                           "(agg_play.kickret_tds * 6), "
                           "(agg_play.puntret_tds * 6), "
                           "(agg_play.defense_safe * 2)) AS defense_play_points, "
                           "game.home_team, game.away_team, "
                           "(game.home_score > game.away_score) AS home_won "
                           "FROM play INNER JOIN agg_play "
                           "ON play.gsis_id = agg_play.gsis_id "
                           "AND play.drive_id = agg_play.drive_id "
                           "AND play.play_id = agg_play.play_id "
                           "INNER JOIN game on play.gsis_id = game.gsis_id "
                           "WHERE game.home_score != game.away_score AND game.finished = TRUE "
                           "ORDER BY play.gsis_id, play.drive_id, play.play_id;")
        assert expected_string == utils._make_nfldb_query_string()

    def test_single_year(self):
        """Test that adding a single year constraint works"""
        expected_substring = ("WHERE game.home_score != game.away_score "
                              "AND game.finished = TRUE AND "
                              "game.season_year = 2013")
        assert expected_substring in utils._make_nfldb_query_string(season_years=[2013])

    def test_single_season_type(self):
        """Test that adding a single season type constraint works"""
        expected_substring = ("WHERE game.home_score != game.away_score "
                              "AND game.finished = TRUE AND "
                              "game.season_type = 'Regular'")
        assert expected_substring in utils._make_nfldb_query_string(season_types=["Regular"])

    def test_multiple_year(self):
        """Test that adding a multiple year constraint works"""
        expected_substring = ("WHERE game.home_score != game.away_score "
                              "AND game.finished = TRUE AND "
                              "game.season_year in (2013,2010)")
        assert expected_substring in utils._make_nfldb_query_string(season_years=[2013, 2010])

    def test_multiple_season_type(self):
        """Test that adding a single season type constraint works"""
        expected_substring = ("WHERE game.home_score != game.away_score "
                              "AND game.finished = TRUE AND "
                              "game.season_type in ('Regular','Postseason'")
        assert expected_substring in utils._make_nfldb_query_string(season_types=["Regular", "Postseason"])

            
class TestAggregateNFLDBScores(object):
    """Testing the _aggregate_nfldb_scores function"""

    def test_single_game_offense_points(self):
        input_df = pd.DataFrame({'gsis_id': [0, 0, 0, 0, 0, 0, 0, 0],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'KC', 'NE', 'NE', 'NE', 'NE'],
                                 'home_team': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
                                 'away_team': ['NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE'],
                                 'offense_play_points': [0, 0, 3, 0, 0, 6, 1, 0],
                                 'defense_play_points': [0, 0, 0, 0, 0, 0, 0, 0]
                                 })
        expected_df = pd.DataFrame({'gsis_id': [0, 0, 0, 0, 0, 0, 0, 0],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'KC', 'NE', 'NE', 'NE', 'NE'],
                                 'home_team': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
                                 'away_team': ['NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE']
                                 })
        #Have to append the score columns manually:
        expected_df[['curr_home_score', 'curr_away_score']] = pd.DataFrame([(0, 0),
                                                                            (0, 0),
                                                                            (0, 0),
                                                                            (3, 0),
                                                                            (3, 0),
                                                                            (3, 0),
                                                                            (3, 6),
                                                                            (3, 7),])
        
        input_df = utils._aggregate_nfldb_scores(input_df)
        pd.util.testing.assert_frame_equal(input_df, expected_df)

    def test_single_game_defense_points(self):
        input_df = pd.DataFrame({'gsis_id': [0, 0, 0, 0, 0, 0, 0, 0],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'KC', 'NE', 'NE', 'NE', 'NE'],
                                 'away_team': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
                                 'home_team': ['NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE'],
                                 'offense_play_points': [0, 0, 3, 0, 0, 6, 1, 0],
                                 'defense_play_points': [0, 0, 0, 0, 0, 0, 0, 0]
                                 })
        expected_df = pd.DataFrame({'gsis_id': [0, 0, 0, 0, 0, 0, 0, 0],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'KC', 'NE', 'NE', 'NE', 'NE'],
                                 'away_team': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
                                 'home_team': ['NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE']
                                 })
        #Have to append the score columns manually:
        expected_df[['curr_home_score', 'curr_away_score']] = pd.DataFrame([(0, 0),
                                                                            (0, 0),
                                                                            (0, 0),
                                                                            (0, 3),
                                                                            (0, 3),
                                                                            (0, 3),
                                                                            (6, 3),
                                                                            (7, 3),])
        
        input_df = utils._aggregate_nfldb_scores(input_df)
        pd.util.testing.assert_frame_equal(input_df, expected_df)

    def test_multiple_games(self):
        input_df = pd.DataFrame({'gsis_id': [0, 0, 0, 1, 1, 1, 1, 1],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'NYJ', 'NE', 'NE', 'NE', 'NE'],
                                 'home_team': ['KC', 'KC', 'KC', 'NYJ', 'NYJ', 'NYJ', 'NYJ', 'NYJ'],
                                 'away_team': ['DEN', 'DEN', 'DEN', 'NE', 'NE', 'NE', 'NE', 'NE'],
                                 'offense_play_points': [0, 0, 3, 0, 0, 6, 1, 0],
                                 'defense_play_points': [0, 0, 0, 0, 0, 0, 0, 0]
                                 })
        expected_df = pd.DataFrame({'gsis_id': [0, 0, 0, 1, 1, 1, 1, 1],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'NYJ', 'NE', 'NE', 'NE', 'NE'],
                                 'home_team': ['KC', 'KC', 'KC', 'NYJ', 'NYJ', 'NYJ', 'NYJ', 'NYJ'],
                                 'away_team': ['DEN', 'DEN', 'DEN', 'NE', 'NE', 'NE', 'NE', 'NE']
                                 })
        #Have to append the score columns manually:
        expected_df[['curr_home_score', 'curr_away_score']] = pd.DataFrame([(0, 0),
                                                                            (0, 0),
                                                                            (0, 0),
                                                                            (0, 0),
                                                                            (0, 0),
                                                                            (0, 0),
                                                                            (0, 6),
                                                                            (0, 7),])
        
        input_df = utils._aggregate_nfldb_scores(input_df)
        pd.util.testing.assert_frame_equal(input_df, expected_df)

    def test_missing_xp(self):
        input_df = pd.DataFrame({'gsis_id': [0, 0, 0, 0, 0, 0, 0, 0],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'NE', 'KC', 'KC', 'KC', 'KC'],
                                 'home_team': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
                                 'away_team': ['NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE'],
                                 'offense_play_points': [0, 0, 0, 0, 0, 0, 6, 0],
                                 'defense_play_points': [0, 0, 6, 0, 0, 0, 0, 0]
                                 })
        expected_df = pd.DataFrame({'gsis_id': [0, 0, 0, 0, 0, 0, 0, 0],
                                 'yardline': [0, 0, 0, -15, 0, 0, 0, -15],
                                 'pos_team': ['KC', 'KC', 'KC', 'NE', 'KC', 'KC', 'KC', 'KC'],
                                 'home_team': ['KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC', 'KC'],
                                 'away_team': ['NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE', 'NE']
                                 })
        #Have to append the score columns manually:
        expected_df[['curr_home_score', 'curr_away_score']] = pd.DataFrame([(0, 0),
                                                                            (0, 0),
                                                                            (0, 0),
                                                                            (0, 7),
                                                                            (0, 7),
                                                                            (0, 7),
                                                                            (0, 7),
                                                                            (7, 7),])
        
        input_df = utils._aggregate_nfldb_scores(input_df)
        pd.util.testing.assert_frame_equal(input_df, expected_df)


    

        
