"""
File: get_hyps_exp4.py
Author: Andrew Root
Date: 9/3/24
Description: This script is used to find optimal hyperparameters for Experiment 4 (i.e. exp4.ipynb).
"""

# The cross validation score for parameter sets is calculated and used as the fitness metric.

# The different cyberbullying types are relabeled such that the dataset consists of a Cyberbullying and a Not Cyberbullying class. 

# IMPORTS
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform, rv_continuous
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from nltk.stem import SnowballStemmer, WordNetLemmatizer, LancasterStemmer, PorterStemmer
import nltk
nltk.download('wordnet')

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import json
import random
import sys
import logging
from pathlib import Path

# GLOBALS
RUN = 'exp4_hyp_search'
TRAIN_DATA_NAME = '48000_cyberbullying_tweets_basic_clean.csv' 
TEST_DATA_NAME = 'hatespeech_tweets_basic_clean.csv'
RANDOM_SEED = 115

logging.basicConfig(
    filename=f'log/{RUN}.log',
    filemode='w+',
    encoding='utf-8',
    format='{asctime} - {levelname} - {message}',
    level=logging.DEBUG,
    style='{',
    datefmt='%Y-%m-%d %H:%M'
)

# CLASSES
class RandomRatio(rv_continuous):
    def __init__(self, ratio, *args, **kwargs):
        self.low = kwargs.pop('low', 0)
        self.high = kwargs.pop('high', 1)
        self.ratio = ratio
        super().__init__(*args, **kwargs)

    def rvs(self, *args, **kwargs):
        if np.random.rand() < 0.5:
            return None
        else:
            return self.ratio + np.random.uniform(self.low, self.high, *args)
        
class RandomRatio2(rv_continuous):
    def __init__(self, ratio, *args, **kwargs):
        self.low = kwargs.pop('low', 0)
        self.high = kwargs.pop('high', 1)
        self.ratio = ratio
        super().__init__(*args, **kwargs)

    def rvs(self, *args, **kwargs):
        if np.random.rand() < 0.5:
            return [1, 1]
        else:
            return [1, self.ratio + np.random.uniform(self.low, self.high, *args)]
        
class StreamToLogger():
    def __init__(self, logger, log_level=logging.DEBUG):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf:str):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

sys.stdout = StreamToLogger(logging.getLogger(), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger(), logging.ERROR)