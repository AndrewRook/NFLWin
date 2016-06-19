"""Utility functions that don't fit in the main modules"""
from __future__ import print_function, division

import numpy as np
import pandas as pd


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
    import nfldb
    import sqlalchemy as sql
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
    """Get play-by-play data from the nfldb database.

    We use a specialized query and then postprocessing because, while possible to
    do using the objects created by ``nfldb``, it is *orders of magnitude slower*.
    This is due to the more general nature of ``nfldb``, which is not really designed
    for this kind of data mining. Since we need to get a lot of data in a single way,
    it's much simpler to interact at a lower level with the underlying postgres
    database.


    Parameters
    ----------
    season_years : list (default=None)
        A list of all years to get data for (earliest year in nfldb is 2009).
        If ``None``, get data from all available seasons.
    season_types : list (default=["Regular", "Postseason"])
        A list of all parts of seasons to get data for (acceptable values are
        "Preseason", "Regular", and "Postseason"). If ``None``, get data from
        all three season types.

    Returns
    -------
    Pandas DataFrame
        The play by play data, with the following columns:
        
        * **gsis_id:** The official NFL GSIS_ID for the game.
        * **drive_id:** The id of the drive, starts at 1 and increases by 1 for each new drive.
        * **play_id:** The id of the play in ``nfldb``. Note that sequential plays have
          increasing but not necessarily sequential values. With ``drive_id`` and ``gsis_id``,
          works as a unique identifier for a given play.
        * **quarter:** The quarter, prepended with "Q" (e.g. ``Q1`` means the first quarter). 
          Overtime periods are denoted as ``OT``, ``OT2``, and theoretically ``OT3`` if one were to
          ever be played.
        * **seconds_elapsed:** seconds elapsed since the start of the quarter.
        * **offense_team:** The abbreviation of the team currently with possession of the ball.
        * **yardline:** The current field position. Goes from -49 to 49, where negative numbers
          indicate that the team with possession is on its own side of the field.
        * **down:** The down. kickoffs, extra points, and similar have a down of 0.
        * **yards_to_go:** How many yards needed in order to get a first down (or touchdown).
        * **home_team:** The abbreviation of the home team.
        * **away_team:** The abbreviation of the away team.
        * **curr_home_score:** The home team's score at the start of the play.
        * **curr_away_score:** The away team's score at the start of the play. 
        * **offense_won:** A boolean - ``True`` if the offense won the game, ``False`` otherwise. (The
          database query skips tied games.)

    Notes
    -----
    ``gsis_id``, ``drive_id``, and ``play_id`` are not necessary to make the model, but
    are included because they can be useful for computing things like WPA.
    """
    
    engine = connect_nfldb()

    sql_string = _make_nfldb_query_string(season_years=season_years, season_types=season_types)

    plays_df = pd.read_sql(sql_string, engine)

    #Fix yardline, quarter and time elapsed:
    def yardline_time_fix(row):
        try:
            yardline = float(row['yardline'][1:-1])
        except TypeError:
            yardline = np.nan
        split_time = row['time'].split(",")
        return yardline, split_time[0][1:], float(split_time[1][:-1])
    
    plays_df[['yardline', 'quarter', 'seconds_elapsed']] = pd.DataFrame(plays_df.apply(yardline_time_fix, axis=1).values.tolist())
    plays_df.drop('time', axis=1, inplace=True)

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
        if play.offense_team == play.home_team:
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
                   'time', 'pos_team AS offense_team', 'yardline', 'down',
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
                   "((game.home_score > game.away_score AND play.pos_team = game.home_team) "
                   "OR (game.away_score > game.home_score AND play.pos_team = game.away_team)) AS offense_won")

    where_clause = ("WHERE game.home_score != game.away_score "
                    "AND game.finished = TRUE "
                    "AND play.pos_team != 'UNK' "
                    "AND (play.time).phase not in ('Pregame', 'Half', 'Final')")

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
