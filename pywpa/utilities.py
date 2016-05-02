"""Utility functions that don't fit in the main modules"""
from __future__ import print_function, division

import nfldb
import numpy as np
import pandas as pd
import sqlalchemy as sql


QUARTER_MAPPING = {'Q1': 1,
                   'Q2': 2,
                   'Q3': 3,
                   'Q4': 4,
                   'OT': 5,
                   'OT2': 5,
                   }

def connect_nfldb():
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
    
    
def get_nfldb_play_data(season_years=None, season_types=["Regular", "Postseason"]):
    """
    

    Parameters
    ----------
    season_years : list (default=None)
        A list of all years to get data for (earliest year in nfldb is 2009).
        If ``None``, get data from all available seasons.
    season_types : list (default=["Regular", "Postseason"])
        A list of all parts of seasons to get data for (acceptable values are
        "Preseason", "Regular", and "Postseason"). If ``None``, get data from
        all three season types.
    """
    engine = connect_nfldb()

    sql_string = _make_nfldb_query_string(season_years=season_years, season_types=season_types)

    plays_df = pd.read_sql(sql_string, engine)

    #Fix yardline, quarter and time elapsed:
    def yardline_time_fix(row):
        try:
            yardline = int(row['yardline'][1:-1])
        except TypeError:
            yardline = np.nan
        split_time = row['time'].split(",")
        return yardline, split_time[0][1:], float(split_time[1][:-1])
    
    plays_df[['yardline', 'quarter', 'time']] = pd.DataFrame(plays_df.apply(yardline_time_fix, axis=1).values.tolist())

    #Set NaN downs (kickoffs, etc) to 0:
    plays_df['down'] = plays_df['down'].fillna(value=0).astype(np.int8)

    #Aggregate scores:
    plays_df = _aggregate_nfldb_scores(plays_df)
    
    return plays_df

def _aggregate_nfldb_scores(play_df):
    """Aggregate the raw nfldb data to get the score of every play."""

    # First, add the yardline of the subsequent play to the df
    play_df['next_yardline'] = play_df['yardline'].shift(-1)

    #Set up the dictionary to keep track of things:
    curr_home_score = 0
    curr_away_score = 0
    curr_gsis_id = play_df.iloc[0].gsis_id
    argdict = {"curr_home_score": 0, "curr_away_score": 0, "curr_gsis_id": play_df.iloc[0].gsis_id}

    #Define an internal function to actually compute the score of a given play:
    def compute_current_scores(play, argdict):
        #If new game, set scores to zero:
        if play.gsis_id != argdict['curr_gsis_id']:
            argdict['curr_home_score'] = 0
            argdict['curr_away_score'] = 0
            argdict['curr_gsis_id'] = play.gsis_id

        #Get current score at start of play:
        home_score_to_return = argdict['curr_home_score']
        away_score_to_return = argdict['curr_away_score']
        
        #Check if an extra point is missing from the data:
        if play.offense_play_points == 6 and play.next_yardline < 0:
            play.offense_play_points += 1
        if play.defense_play_points == 6 and play.next_yardline < 0:
            play.defense_play_points += 1

        #Update scores, if necessary:
        if play.pos_team == play.home_team:
            argdict['curr_home_score'] += play.offense_play_points
            argdict['curr_away_score'] += play.defense_play_points
        else:
            argdict['curr_home_score'] += play.defense_play_points
            argdict['curr_away_score'] += play.offense_play_points
        return home_score_to_return, away_score_to_return

    #Apply function to data:
    #TODO (AndrewRook): Make the .apply function go faster, currently it's a large bottleneck
    aggregate_scores = play_df.apply(compute_current_scores, axis=1, args=(argdict,))
    aggregate_scores = pd.DataFrame(aggregate_scores.values.tolist())
    play_df[['curr_home_score', 'curr_away_score']] = aggregate_scores

    #Drop unnecessary columns:
    play_df.drop(labels=["next_yardline", "offense_play_points", "defense_play_points"],
                 axis=1, inplace=True)

    return play_df


def _make_nfldb_query_string(season_years=None, season_types=None):
    """Construct the query string to get all the play data.

    This way is a little more compact and robust than specifying
    the string in the function that uses it.

    """
    
    play_fields = ['gsis_id', 'drive_id', 'play_id',
                   'time', 'pos_team', 'yardline', 'down',
                   'yards_to_go']

    offense_play_points = ("GREATEST("
        "(agg_play.fumbles_rec_tds * 6), "
        "(agg_play.kicking_rec_tds * 6), "
        "(agg_play.passing_tds * 6), "
        "(agg_play.receiving_tds * 6), "
        "(agg_play.rushing_tds * 6), "
        "(agg_play.kicking_xpmade * 1), "
        "(agg_play.passing_twoptm * 2), "
        "(agg_play.receiving_twoptm * 2), "
        "(agg_play.rushing_twoptm * 2), "
        "(agg_play.kicking_fgm * 3)) "
        "AS offense_play_points")
    defense_play_points = ("GREATEST("
        "(agg_play.defense_frec_tds * 6), "
        "(agg_play.defense_int_tds * 6), "
        "(agg_play.defense_misc_tds * 6), "
        "(agg_play.kickret_tds * 6), "
        "(agg_play.puntret_tds * 6), "
        "(agg_play.defense_safe * 2)) "
        "AS defense_play_points")

    game_fields = ("game.home_team, game.away_team, "
                   "(game.home_score > game.away_score) AS home_won")

    where_clause = ("WHERE game.home_score != game.away_score "
                    "AND game.finished = TRUE")

    if season_years is not None:
        where_clause += " AND game.season_year"
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
    query_string += ", " + offense_play_points
    query_string += ", " + defense_play_points
    query_string += ", " + game_fields
    query_string += " FROM play INNER JOIN agg_play"
    query_string += (" ON play.gsis_id = agg_play.gsis_id"
        " AND play.drive_id = agg_play.drive_id"
        " AND play.play_id = agg_play.play_id")
    query_string += " INNER JOIN game on play.gsis_id = game.gsis_id"
    query_string += " " + where_clause
    query_string += " ORDER BY play.gsis_id, play.drive_id, play.play_id;"

    return query_string
    
    
if __name__ == "__main__":
    import time
    start = time.time()
    print(len(get_nfldb_play_data(season_years=[2015])))
    print("took {0:.2f}s".format(time.time() - start))
