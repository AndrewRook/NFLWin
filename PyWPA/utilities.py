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
    
    
def get_play_data(season_years=None, season_types=["Regular", "Postseason"]):
    """"""
    engine = connect_db()

    sql_string = _make_query_string(season_years=season_years, season_types=season_types)

    plays_df = pd.read_sql(sql_string, engine)

    #TODO: Go through each play to get the current aggregate home and away scores. 
    print(len(plays_df))

def _make_query_string(season_years=None, season_types=None):
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
        "(agg_play.defense_safe * 2)) "
        "AS play_points")

    game_fields = ("game.home_team, game.away_team, "
                   "(game.home_score > game.away_score) AS home_won")

    where_clause = ("WHERE game.home_score != game.away_score "
                    "AND game.finished = TRUE")

    if season_years is not None:
        where_clause += " AND game.season_year"# in ({0})"
        if len(season_years) == 1:
            where_clause += " = {0}".format(season_years[0])
        else:
            where_clause += (" in ({0})"
                            "".format(",".join([str(year) for year in season_years])))
    if season_types is not None:
        where_clause += " AND game.season_type"
        if len(season_types) == 1:
            where_clause += " = '{0}'".format(season_types[0])
        else:
            where_clause += " in ('{0}')".format("','".join(season_types))

    query_string = "SELECT "
    query_string += "play." + ", play.".join(play_fields)
    query_string += ", " + play_points
    query_string += ", " + game_fields
    query_string += " FROM play INNER JOIN agg_play"
    query_string += (" ON play.gsis_id = agg_play.gsis_id"
        " AND play.drive_id = agg_play.drive_id"
        " AND play.play_id = agg_play.play_id")
    query_string += " INNER JOIN game on play.gsis_id = game.gsis_id"
    query_string += " " + where_clause
    query_string += " ORDER BY play.gsis_id, play.drive_id, play.play_id;"

    return query_string
    

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
    get_play_data(season_years=[2015])
