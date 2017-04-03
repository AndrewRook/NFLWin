try:
    from ConfigParser import RawConfigParser
except ImportError:
    from configparser import RawConfigParser
import os
import sqlalchemy as sql
import sys

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

                engine = sql.create_engine(sql.engine.url.URL(**db_config))
    
                return engine
        except IOError:
            pass
    
    #Couldn't find a config:
    raise IOError("connect_nfldb: couldn't find a configuration file. "
                  "looked here: {0}".format(", ".join(paths)))

def query_nfldb(engine, season_years=None, season_types=["Regular", "Postseason"]):
    """"""
    metadata = sql.MetaData(engine)
    #team = sql.Table("team", metadata, autoload=True)
    play = sql.Table("play", metadata,
                     #sql.Column("time", GameTime()),
                     autoload=True)
    #agg_play = sql.Table("agg_play", metadata, autoload=True)
    #game = sql.Table("game", metadata, autoload=True)

    print(dir(play.c.time))
    select = sql.sql.select([play.c.time]).limit(10)
    conn = engine.connect()
    import pandas as pd
    df = pd.read_sql(select, conn)
    #print(df)
    #result = conn.execute(select)
    #for row in result:
    #    print(row)
    # for column in sorted([c.name for c in play.columns]):
    #     print(column)
    #print(agg_play.columns)

# class GameTime(sql.types.TypeDecorator):
#     impl = sql.types.Unicode
#     def process_bind_param(self, value, dialect):
#         return "({0},{1})".format(value[0], value[1])
#     def process_result_value(self, value, dialect):
#         values = value[1:-1].split(",")
#         values[1] = int(values[1])
#         return values
            

# class GameTime(sql.types.UserDefinedType):
#     name = "game_time"

#     def __init__(self):
#         pass
#     def get_col_spec(self):
#         return "{0}(quarter, seconds_elapsed)".format(name)

#     def result_processor(self, dialect, coltype):
#         def process(value):
#             values = value[1:-1].split(",")
#             values[1] = int(values[1])
#             return values
#         return process
    
#     def bind_processor(self, dialect):
#         def process(value):
#             return "({0},{1})".format(value[0], value[1])
#         return process
#     def test(self, dialect, coltype):
#         def process(value):
#             return value[1:-1].split(",")
#         return process

# import re
# import psycopg2
# from psycopg2.extensions import adapt, register_adapter, AsIs
# class GameTime(object):
#     def __init__(self, quarter, seconds_elapsed):
#         self.quarter = quarter
#         self.seconds_elapsed = seconds_elapsed

# def adapt_game_time(game_time):
#     return AsIs("'({0},{1})'".format(adapt(game_time.quarter), adapt(game_time.seconds_elapsed)))

# def cast_game_time(value, cur):
#     if value is None:
#         return None
#     m = re.match(r"\(([^)]+),([^)]+)\)", value)
#     if m:
#         return GameTime(m.group(1), int(m.group(2)))
#     else:
#         raise psycopg2.InterfaceError("bad point representation: %r" % value)

# def register_game_time_type(engine):
#     register_adapter(GameTime, adapt_game_time)
#     rs = engine.execute("SELECT NULL::game_time")
#     game_time_old = rs.cursor.description[0][1]
#     GAME_TIME = psycopg2.extensions.new_type((game_time_old,), "game_time", cast_game_time)
#     psycopg2.extensions.register_type(GAME_TIME)


if __name__ == "__main__":
    engine = connect_nfldb()
    #register_game_time_type(engine)
    query_nfldb(engine)
