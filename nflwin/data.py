try:
    from ConfigParser import RawConfigParser
except ImportError:
    from configparser import RawConfigParser
import os
import sys

import pandas as pd
import sqlalchemy as sa

def connect_nfldb(config_path=''):
    """Connect to an existing nfldb database.

    First looks for the nfldb configuration file. A total of three
    possible file paths are tried before giving
    up and raising an error. The file paths, in order, are:
    ``config_path``, ``sys.prefix/share/nfldb/config.ini`` and
    ``$XDG_CONFIG_HOME/nfldb/config.ini``.

    Notes
    -----
    This function borrows heavily from ``nfldb.db.config``.
    """
    _config_home = os.getenv('XDG_CONFIG_HOME')
    if not _config_home:
        home = os.getenv('HOME')
    if not home:
        _config_home = ''
    else:
        _config_home = os.path.join(home, '.config')

    paths = [
        config_path,
        os.path.join(sys.prefix, 'share', 'nfldb', 'config.ini'),
        os.path.join(_config_home, 'nfldb', 'config.ini'),
    ]
    tried = []
    cp = RawConfigParser()
    for p in paths:
        tried.append(p)
        try:
            with open(p) as fp:
                cp.readfp(fp)
                db_config = {
                    "drivername": "postgres",
                    "username": cp.get("pgsql", "user"),
                    "password": cp.get("pgsql", "password"),
                    "database": cp.get("pgsql", "database"),
                    "host": cp.get("pgsql", "host"),
                    "port": cp.getint("pgsql", "port")
                }

                engine = sa.create_engine(sa.engine.url.URL(**db_config))
    
                return engine
        except IOError:
            pass
    
    #Couldn't find a config:
    raise IOError("connect_nfldb: couldn't find a configuration file. "
                  "looked here: {0}".format(", ".join(paths)))

def _create_game_query(game_table):
    subquery_columns = [game_table.c.home_team,
                        game_table.c.away_team,
                        game_table.c.season_year]
    subquery_columns.append(
        (sa.case(
            [
             (game_table.c.season_type == "Regular", 0)
            ],
            else_=1) * 100 + game_table.c.week).label("game_order"))
    subquery_columns.append(
        (game_table.c.home_score > game_table.c.away_score).label("home_win"))
    subquery = (sa.select(subquery_columns)
                .where(game_table.c.season_type != "Preseason")
                .alias("subquery"))

    query_columns = [sa.func.max(game_table.c.gsis_id).label("gsis_id"),
                     sa.func.max(game_table.c.season_year).label("season_year"),
                     sa.func.max(game_table.c.season_type).label("season_type"),
                     sa.func.max(game_table.c.week).label("week")]
    query_columns.append(
        sa.func.sum(sa.case(
            [(sa.and_(game_table.c.home_team == subquery.c.home_team,
                      subquery.c.home_win == True), 1),
             (sa.and_(game_table.c.home_team == subquery.c.away_team,
                      subquery.c.home_win == False), 1)
            ], else_=0)).label("home_wins"))
    query_columns.append(
        sa.func.sum(sa.case(
            [(sa.and_(game_table.c.home_team == subquery.c.home_team,
                      subquery.c.home_win == False), 1),
             (sa.and_(game_table.c.home_team == subquery.c.away_team,
                      subquery.c.home_win == True), 1)
            ], else_=0)).label("home_losses"))
    query_columns.append(
        sa.func.sum(sa.case(
            [(sa.and_(game_table.c.away_team == subquery.c.home_team,
                      subquery.c.home_win == True), 1),
             (sa.and_(game_table.c.away_team == subquery.c.away_team,
                      subquery.c.home_win == False), 1)
            ], else_=0)).label("away_wins"))
    query_columns.append(
        sa.func.sum(sa.case(
            [(sa.and_(game_table.c.away_team == subquery.c.home_team,
                      subquery.c.home_win == False), 1),
             (sa.and_(game_table.c.away_team == subquery.c.away_team,
                      subquery.c.home_win == True), 1)
            ], else_=0)).label("away_losses"))

    query = (sa.select(query_columns)
             .select_from(game_table.join(subquery, sa.and_(
                 game_table.c.season_year == subquery.c.season_year,
                 sa.case(
                     [
                         (game_table.c.season_type == "Regular", 0)
                     ],
                     else_=1) * 100 + game_table.c.week > subquery.c.game_order,
                 sa.or_(
                     game_table.c.home_team == subquery.c.home_team,
                     game_table.c.home_team == subquery.c.away_team,
                     game_table.c.away_team == subquery.c.home_team,
                     game_table.c.away_team == subquery.c.away_team))))
             .where(sa.and_(
                 game_table.c.season_type != "Preseason",
                 game_table.c.finished == True))
             .group_by(game_table.c.gsis_id))
                      
                     
    return query

def _create_play_query(play_table, agg_play_table):
    offense_play_points_column = sa.func.greatest(
        agg_play_table.c.fumbles_rec_tds * 6,
        agg_play_table.c.kicking_rec_tds * 6,
        agg_play_table.c.passing_tds * 6,
        agg_play_table.c.receiving_tds * 6,
        agg_play_table.c.rushing_tds * 6,
        agg_play_table.c.kicking_xpmade * 1,
        agg_play_table.c.passing_twoptm * 2,
        agg_play_table.c.receiving_twoptm * 2,
        agg_play_table.c.rushing_twoptm * 2,
        agg_play_table.c.kicking_fgm * 3).label("offense_play_points")
    defense_play_points_column = sa.func.greatest(
        agg_play_table.c.defense_frec_tds * 6,
        agg_play_table.c.defense_int_tds * 6,
        agg_play_table.c.defense_misc_tds * 6,
        agg_play_table.c.kickret_tds * 6,
        agg_play_table.c.puntret_tds * 6,
        agg_play_table.c.defense_safe * 2).label("defense_play_points")
    subquery = (sa.select([agg_play_table.c.gsis_id,
                            agg_play_table.c.drive_id,
                            agg_play_table.c.play_id,
                            offense_play_points_column,
                            defense_play_points_column,
                            play_table.c.pos_team])
                 .select_from(agg_play_table.join(play_table, sa.and_(
                     agg_play_table.c.gsis_id == play_table.c.gsis_id,
                     agg_play_table.c.drive_id == play_table.c.drive_id,
                     agg_play_table.c.play_id == play_table.c.play_id)))
                 .where(
                     play_table.c.pos_team != "UNK").alias("subquery"))

    play_columns = [play_table.c.gsis_id,
                    play_table.c.drive_id,
                    play_table.c.play_id,
                    play_table.c.time,
                    play_table.c.yardline,
                    play_table.c.down,
                    play_table.c.yards_to_go,
                    play_table.c.pos_team,
                    sa.func.sum(sa.case(
                        [(play_table.c.pos_team == subquery.c.pos_team, subquery.c.offense_play_points)],
                        else_=subquery.c.defense_play_points)).label("offense_points"),
                    sa.func.sum(sa.case(
                        [(play_table.c.pos_team == subquery.c.pos_team, subquery.c.defense_play_points)],
                        else_=subquery.c.offense_play_points)).label("defense_points"),
                    ]
    query = (sa.select(play_columns)
             .select_from(play_table.join(subquery, sa.and_(
                 play_table.c.gsis_id == subquery.c.gsis_id,
                 sa.or_(
                     play_table.c.drive_id > subquery.c.drive_id,
                     sa.and_(
                         play_table.c.drive_id == subquery.c.drive_id,
                         play_table.c.play_id > subquery.c.play_id
                     )
                 ))))
              .where(play_table.c.pos_team != "UNK")
             .group_by(play_table.c.gsis_id, play_table.c.drive_id, play_table.c.play_id))
    return query
                 
    

def query_nfldb(engine, season_years=None, season_types=["Regular", "Postseason"]):
    """"""
    metadata = sa.MetaData(engine)
    tables = {}
    tables["team"] = sa.Table("team", metadata, autoload=True)
    tables["play"] = sa.Table("play", metadata, autoload=True)
    tables["agg_play"] = sa.Table("agg_play", metadata, autoload=True)
    tables["game"] = sa.Table("game", metadata, autoload=True)

    game_query = _create_game_query(tables["game"])
    play_query = _create_play_query(tables["play"], tables["agg_play"])
    #timezone mapping
    with engine.connect() as conn:
        df = pd.read_sql(play_query.limit(50), conn)

    print(df)


if __name__ == "__main__":
    engine = connect_nfldb()
    #register_game_time_type(engine)
    query_nfldb(engine)
