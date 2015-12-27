'''
This module hosts configuration information,
such as where data is stored.
'''
import os

#The directory this file is in:
PYWPA_DIR = os.path.dirname(os.path.realpath(__file__))

DATA_DIR = os.path.join(os.path.dirname(PYWPA_DIR),
                        'PyWPA',
                        'data')
MODEL_DIR = os.path.join(os.path.dirname(PYWPA_DIR),
                         'PyWPA',
                         'model')

DATA_PREFIX = 'nfldb_processed_data'

MODEL_FILENAME = os.path.join(MODEL_DIR,'PyWPA_model.pkl')

QUARTER_MAPPING = {'Q1': 1,
                   'Q2': 2,
                   'Q3': 3,
                   'Q4': 4,
                   'OT': 5,
                   'OT2': 5,
                   }

DATA_COLUMNS = ['quarter',
                'time_remaining',
                'score_diff',
                'is_offense_home',
                'down',
                'distance',
                'field_position',
                'is_offense_winner'
                ]
