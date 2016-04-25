"""Utility functions that don't fit in the main modules"""
from __future__ import print_function, division

import time

import nfldb
import pandas as pd
import sqlalchemy as sql


QUARTER_MAPPING = {'Q1': 1,
                   'Q2': 2,
                   'Q3': 3,
                   'Q4': 4,
                   'OT': 5,
                   'OT2': 5,
                   }

def connect_db():
    """Connect to the nfldb database.

    Rather than using the builtin method we make our own,
    since we're going to use SQLAlchemy as the engine. However,
    we can still make use of the information in the nfldb config
    file to get information like username and password, which
    means this function doesn't need any arguments.

    Parameters
    ----------
    None

    Returns
    -------
    SQLAlchemy engine object
        A connected engine, ready to be used to query the DB.

    Raises
    ------
    IOError
        If it can't find the config file.
    """
    
    db_config, paths_tried = nfldb.db.config()
    if db_config is None:
        raise IOError("get_play_data: could not find database config! Looked"
                      " in these places: {0}".format(paths_tried))
    db_config["drivername"] = "postgres"
    db_config["username"] = db_config["user"]
    del db_config["user"]
    del db_config["timezone"]

    engine = sql.create_engine(sql.engine.url.URL(**db_config))

    return engine
    
    
def get_play_data(**kwargs):
    """"""
    engine = connect_db()

    sql_string = """SELECT play.gsis_id, play.drive_id, play.play_id,
    play.time, play.pos_team, play.yardline, play.down, play.yards_to_go,
    GREATEST((agg_play.defense_frec_tds * 6), (agg_play.defense_int_tds * 6),
    (agg_play.defense_misc_tds * 6), (agg_play.fumbles_rec_tds * 6),
    (agg_play.kicking_rec_tds * 6), (agg_play.kickret_tds * 6),
    (agg_play.passing_tds * 6), (agg_play.puntret_tds * 6),
    (agg_play.receiving_tds * 6), (agg_play.rushing_tds * 6),
    (agg_play.kicking_xpmade * 1), (agg_play.passing_twoptm * 2),
    (agg_play.receiving_twoptm * 2), (agg_play.rushing_twoptm * 2),
    (agg_play.kicking_fgm * 3), (agg_play.defense_safe * 2)) AS play_points
    FROM play
    INNER JOIN agg_play
    ON play.gsis_id = agg_play.gsis_id AND play.drive_id = agg_play.drive_id AND play.play_id = agg_play.play_id
    ORDER BY play.gsis_id, play.drive_id, play.play_id;"""

    _make_query_string([2015], ["Regular"])

    plays_df = pd.read_sql(sql_string, engine)
    print(len(plays_df))
    #TODO: add home team, away team, and winning team to query. 

def _make_query_string(season_years, season_type):
    """Construct the query string to get all the play data.

    This way is a little more compact and robust than specifying
    the string in the function that uses it.

    """
    
    play_fields = ['gsis_id', 'drive_id', 'play_id',
                   'time', 'pos_team', 'yardline', 'down',
                   'yards_to_go']

    play_points = ("GREATEST((agg_play.defense_frec_tds * 6),"
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
        "(agg_play.defense_safe * 2))")

    query_string = "SELECT "
    query_string += "play." + ", play.".join(play_fields)
    query_string += ", " + play_points
    query_string += " FROM play INNER JOIN agg_play"
    query_string += (" ON play.gsis_id = agg_play.gsis_id"
        " AND play.drive_id = agg_play.drive_id"
        " AND play.play_id = agg_play.play_id")
    query_string += " ORDER BY play.gsis_id, play.drive_id, play.play_id;"
    print(query_string)
    

def parse_plays(game):
    """"""
    home_team = game.home_team
    winning_team = game.winner

    data_dict = {"home_team": [],
                 "winning_team": [],
                 "offense_team": [],
                 "down": [],
                 "distance": [],
                 "yardline": [],
                 "quarter": [],
                 "time_left": [],
                 "home_score": [],
                 "away_score": []
                 }
    for play in game.plays:
        offense_team = play.pos_team

        down = play.down
        distance = play.yards_to_go

        yardline = play.yardline._offset

        #print(play.description)
        quarter = play.time.phase.name
        time_left = (15*60 - play.time.elapsed) #seconds
        
        home_score, away_score = play.score(before=True)
        if offense_team != "UNK" and down is not None and quarter != "Half":
            data_dict['offense_team'].append(offense_team)
            data_dict['winning_team'].append(winning_team)
            data_dict['home_team'].append(home_team)
            data_dict['down'].append(down)
            data_dict['distance'].append(distance)
            data_dict['yardline'].append(yardline)
            data_dict['quarter'].append(QUARTER_MAPPING[quarter])
            data_dict['time_left'].append(time_left)
            data_dict['home_score'].append(home_score)
            data_dict['away_score'].append(away_score)

    return pd.DataFrame(data_dict)
    
    
if __name__ == "__main__":
    get_play_data(season_type=["Postseason"],
                  season_year=[2015])
