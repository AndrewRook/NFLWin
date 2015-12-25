'''
This module has functions that handle the preprocessing of data
to put it in the proper format for loading.
'''

from __future__ import print_function, division
import os
import time

import nfldb

import config as cf

def process_season(season_year, season_type):
    '''
    Query the nfldb play table for a season,
    then add score data and save it to a csv.

    Arguments:
    season_year: The season to process (between 2009 and
        today).
    season_type: A string or list of strings
        with one or more of the following: Preseason, Regular,
        or Postseason.

    Returns:
    None, but saves the data to a file in cf.DATA_DIR named
        cf.DATA_PREFIX+"_{0:d}.csv".format(season_year).
    '''
    db = nfldb.connect()
    query = nfldb.Query(db)

    #Get all the games for the season:
    query.game(season_year=season_year, season_type=season_type).play(down__ge=0) #down__ge=0 pulls out kickoffs and stuff.

    #Create the filename:
    filename = os.path.join(cf.DATA_DIR,
                            "{0:s}_{1:d}.csv".format(cf.DATA_PREFIX,season_year))
    #Open the file handle:
    count = 0
    game_dict = {} #Using a dict minimizes table lookups
    with open(filename, 'wb') as handle:
        num_plays = len(query.as_plays())
        for play in query.as_plays():
            #Print a status update:
            if count % (num_plays//20) == 0:
                print("{0:d}: {1:d} of {2:d} plays processed".format(season_year,
                                                                     count,
                                                                     num_plays))
            count += 1

            #process play:
            quarter, time_left, score_diff, is_offense_home, \
              down, distance, field_position, is_offense_winner = \
              process_play(db, play, game_dict=game_dict)
            

            #Write the next line of the file:
            handle.write("{0},{1},{2},{3},{4},{5},{6},{7}\n".format(quarter,
                                                  time_left,
                                                  score_diff,
                                                  int(is_offense_home),
                                                  down,
                                                  distance,
                                                  field_position,
                                                  int(is_offense_winner)
                                                  ))
            # if count >= 1000:
            #     break

def process_play(db, play, game_dict=None):
    '''
    Go through a play and extract the information needed to compute
    win efficiency.

    Arguments:
    db: The nfldb database connection
    play: an instance of nfldb.Play
    game_home_dict (None): If not none, use this object as
        a cache to limit the number of database hits.

    Returns:
    quarter: the quarter of the game, based on config.QUARTER_MAPPING.
    time_left: The number of seconds left in the quarter.
    score_diff: The difference in score (home - away).
    is_offense_home: A boolean indicating if the offense is the home team.
    down: The down.
    distance: The number of yards to go for a first down.
    field_position: The number of yards from your own endzone you are.
    is_offense_winner: Did the offense win the game?
    '''
    #Get the score at the start of the play:
    home_score, away_score = play.score(before=True)
    score_diff = home_score - away_score

    #Find whether the offense is the home team:
    team_with_ball = play.pos_team
    if game_dict is not None:
        try:
            game_dict[play.gsis_id]
        except KeyError: #Need to add the game first:
            game = nfldb.Game.from_id(db, play.gsis_id)
            game_dict[play.gsis_id] = game
        home_team = game_dict[play.gsis_id].home_team
        winning_team = game_dict[play.gsis_id].winner
    else:
        game = nfldb.Game.from_id(db, play.gsis_id)
        home_team = game.home_team
        winning_team = game.winner
        
    is_offense_home = team_with_ball == home_team
    is_offense_winner = winning_team == team_with_ball
    
    #Get other parameters
    field_position = play.yardline._offset + 50 #converts to 1-99
    down = play.down
    distance = play.yards_to_go
    quarter = cf.QUARTER_MAPPING[play.time.phase.name]
    time_left = (15*60 - play.time.elapsed) #seconds
    return (quarter, time_left, score_diff,
            is_offense_home, down, distance,
            field_position, is_offense_winner)


if __name__ == "__main__":
    start = time.time()
    process_season(2013,['Regular', 'Postseason'])
    print("took {0:.2f}s".format(time.time()-start))
