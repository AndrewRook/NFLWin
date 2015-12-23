from __future__ import print_function, division

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import nfldb
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

def rescale_data(quarter,
                 time_remaining,
                 score_diff,
                 is_offense_home,
                 down,
                 distance,
                 field_position):
    '''
    Rescale the data from a play or set of plays
    to be uniform among every dimension.

    Arguments:
    quarter: The quarter the game is in (1,2,3,4, or 5 for anything in OT).
    time_remaining: Seconds counting down from the start of the quarter
        (e.g. the start of the quarter is 15*60, the end of the quarter
        is 0).
    score_diff: The score differential (home - away).
    is_offense_home: Is the offense the home team? Boolean true/false.
    down: What down is it (1,2,3, or 4).
    distance: How many yards to go for the first down.
    field_position: How many yards from your own endzone you are (1 is your one-yard
        line, 99 is 1 yard from a touchdown).

    Returns a tuple of:
    quarter_scaled: 1->0, 2->0.25, 3->0.5, 4->0.75, 5->1.
    time_remaining_scaled: 15*60->0, 0->1.
    score_diff_scaled: -inf->0, +inf->1, on a logistic scale with k=0.1.
    is_offense_home_scaled: True->1, False->0.
    down_scaled: 1->0, 2->0.33, 3->0.67, 4->1.
    distance_scaled: 1->0, +inf->1, on a logistic scale with k=0.2.
    field_position_scaled: 99 yards to go->0, 1 yard to go->1.
    '''
    quarter_scaled = (quarter-1)/4.
    time_remaining_scaled = 1. - (time_remaining-1)/(15*60.)
    score_diff_scaled = 1/(1+np.exp(-0.1*score_diff))
    is_offense_home_scaled = is_offense_home*1.
    down_scaled = (down-1)/3.
    distance_scaled = 2*(1/(1+np.exp(-0.2*(distance-1)))-0.5)
    field_position_scaled = (field_position-1)/99.
    return (quarter_scaled,
            time_remaining_scaled,
            score_diff_scaled,
            is_offense_home_scaled,
            down_scaled,
            distance_scaled,
            field_position_scaled)

if __name__ == "__main__":
    db = nfldb.connect()
    q = nfldb.Query(db)

    q.game(season_year=2015, week=1, season_type='Regular')
    start = time.time()
    for p in q.as_plays():
        woo = (p.score(before=True), p.description)
    print(time.time()-start)

    # start = time.time()
    # q.game(season_type=['Regular','Postseason'])
    # print(time.time()-start)
    # print(len(q.as_plays()), len(q.as_games()))
    # print(time.time()-start)

    
    # #np.random.seed(891)
    # num_samples = 120*16*16*5
    # print("num_samples = {0:d}".format(num_samples))
    # quarter = np.random.randint(1, 6, num_samples)
    # time_remaining = np.random.randint(0, 15*60+1, num_samples)
    # score_diff = np.random.randint(-50, 50+1, num_samples)
    # is_offense_home = np.random.randint(0, 2, num_samples).astype(np.bool)
    # down = np.random.randint(1, 5, num_samples)
    # distance = np.random.randint(1, 36, num_samples)
    # field_position = np.random.randint(1, 99, num_samples)
    # team_won = np.random.randint(0, 2, num_samples)
    # #team_won[1] = 0

    # start = time.time()
    # output = rescale_data(quarter,
    #                       time_remaining,
    #                       score_diff,
    #                       is_offense_home,
    #                       down,
    #                       distance,
    #                       field_position)
    # print("rescaled: {0:.2f}s".format(time.time()-start))
    # distance_array = np.zeros((num_samples,len(output)),dtype=np.float)
    # for i in range(len(output)):
    #     distance_array[:,i] = output[i]
    # print("created 2d array: {0:.2f}s".format(time.time()-start))

    # knn = KNeighborsClassifier(n_neighbors=100, algorithm='ball_tree', leaf_size=30)
    # knn.fit(distance_array,team_won)
    # print("finished fitting: {0:.2f}".format(time.time()-start))

    # print(knn.predict_proba(distance_array[1030,:].reshape(1,-1)))
