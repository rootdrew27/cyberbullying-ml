"""
File: filename.py
Author: Andrew Root
Date: 8/5/24
Description: This script is used to find optimal hyperparameters for an XGBoost model trained on Dataset 1 (cyberbullying_tweets.csv)
"""

# The cross validation score for parameter sets is calculated and used as the fitness metric.

# The different cyberbullying types are relabeled such that the dataset consists of a Cyberbullying and a Not Cyberbullying class. 

# IMPORTS

import numpy as np

import logging

# GLOBALS
RUN = 'hyp_search_w_re'
PATH_TO_DATA = './data/bullying_light_clean_3.csv'
np.random.seed(115)
logging.basicConfig(
    filename=f'log/{RUN}.log',
    encoding='utf-8',
    format='{asctime} - {levelname} - {message}',
    level=logging.DEBUG,
    style='{',
    datefmt='%Y-%m-%d %H:%M'
)

# FUNCTIONS

# BEGIN

if __name__ == '__main__':
    

    
    
# END 