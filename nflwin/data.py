try:
    from ConfigParser import RawConfigParser
except ImportError:
    from configparser import RawConfigParser
import os
import sys
import time

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
                     sa.func.max(game_table.c.week).label("week"),
                     sa.func.max(game_table.c.home_team).label("home_team"),
                     sa.func.max(game_table.c.away_team).label("away_team")
                     ]
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

def make_nfldb_query(tables):

    p = tables["play"]
    ap = tables["agg_play"]
    g = (_create_game_query(tables["game"])).alias("game")
    offense_points_clause = sa.func.greatest(
        ap.c.fumbles_rec_tds * 6,
        ap.c.kicking_rec_tds * 6,
        ap.c.passing_tds * 6,
        ap.c.receiving_tds * 6,
        ap.c.rushing_tds * 6,
        ap.c.kicking_xpmade * 1,
        ap.c.passing_twoptm * 2,
        ap.c.receiving_twoptm * 2,
        ap.c.rushing_twoptm * 2,
        ap.c.kicking_fgm * 3)
    defense_points_clause = sa.func.greatest(
        ap.c.defense_frec_tds * 6,
        ap.c.defense_int_tds * 6,
        ap.c.defense_misc_tds * 6,
        ap.c.kickret_tds * 6,
        ap.c.puntret_tds * 6,
        ap.c.defense_safe * 2)

    home_team_points = sa.case(
        [(p.c.pos_team == g.c.home_team, offense_points_clause),
         (p.c.pos_team == g.c.away_team, defense_points_clause)],
         else_=0)
    away_team_points = sa.case(
        [(p.c.pos_team == g.c.away_team, offense_points_clause),
         (p.c.pos_team == g.c.home_team, defense_points_clause)],
         else_=0)
    agg_home_team_points = (sa.func.sum(home_team_points).over(
        partition_by=ap.c.gsis_id,
        order_by=[ap.c.drive_id, ap.c.play_id],
        range_=(None, 0)) - home_team_points).label("agg_home_team_points")
    agg_away_team_points = (sa.func.sum(away_team_points).over(
        partition_by=ap.c.gsis_id,
        order_by=[ap.c.drive_id, ap.c.play_id],
        range_=(None, 0)) - away_team_points).label("agg_away_team_points")
    columns_to_select = [ap.c.drive_id,
                         ap.c.play_id,
                         g,
                         p.c.pos_team,
                         agg_home_team_points,
                         agg_away_team_points]
    
    
    query = sa.select(columns_to_select).select_from(
        ap.join(p, sa.and_(ap.c.gsis_id == p.c.gsis_id, ap.c.play_id == p.c.play_id))
        .join(g, ap.c.gsis_id == g.c.gsis_id)).order_by(ap.c.gsis_id, ap.c.play_id)
    return query


def query_nfldb(engine, season_years=None, season_types=["Regular", "Postseason"]):
    """"""
    metadata = sa.MetaData(engine)
    tables = {}
    tables["team"] = sa.Table("team", metadata, autoload=True)
    tables["play"] = sa.Table("play", metadata, autoload=True)
    tables["agg_play"] = sa.Table("agg_play", metadata, autoload=True)
    tables["game"] = sa.Table("game", metadata, autoload=True)

    with engine.connect() as conn:

        test_query = (make_nfldb_query(tables))#.limit(100)
        start = time.time()
        df = pd.read_sql(test_query, conn)
        print("Took {0:.2f}s".format(time.time() - start))
    print(df.shape)

    return df

if __name__ == "__main__":
    engine = connect_nfldb()
    #register_game_time_type(engine)
    df = query_nfldb(engine)
    df.to_csv("test_data.csv", index=False)
