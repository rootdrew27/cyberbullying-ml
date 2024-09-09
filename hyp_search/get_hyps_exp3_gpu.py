"""
File: get_hyps_exp3.py
Author: Andrew Root
Date: 8/25/24
Description: This script is used to find optimal hyperparameters for Experiment 3 (i.e. exp3.ipynb).
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
RUN = 'exp3_hyp_search_gpu'
TRAIN_DATA_NAME = '48000_cyberbullying_tweets_basic_clean.csv' 
TEST_DATA_NAME = 'hatespeech_tweets_basic_clean.csv'
RANDOM_SEED = 115

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

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

# FUNCTIONS
def get_OOV_feats(train_data:pd.Series, test_data:pd.Series, print_oov_feats:bool=False):
    v1 = CountVectorizer()
    v2 = CountVectorizer()

    v1.fit(train_data)
    v2.fit(test_data)

    feats_train = v1.get_feature_names_out()
    feats_test = v2.get_feature_names_out()

    oov_feats = np.setdiff1d(feats_test, feats_train)
    if print_oov_feats: print(f"OOV features: {oov_feats}")

    return oov_feats

def balance_train_and_test(notcb_train:pd.DataFrame, cb_train:pd.DataFrame, notcb_test:pd.DataFrame, cb_test:pd.DataFrame, random_state:int):
    """
    This functions does not relabel the data that is moved!
    """
    x1 = notcb_train.shape[0]
    x2 = cb_train.shape[0]
    y1 = notcb_test.shape[0]
    y2 = cb_test.shape[0]

    x_hat = (y2 * x1 - x2 * y1) / (y2 + x2)

    
    if x_hat < 0: # take from the test and add to training
        np.testing.assert_allclose(ratio:=(y1 + x_hat)/(y2), (x1 - x_hat)/(x2), atol=1e-5)
        notcb_test = notcb_test.sample(frac=1, random_state=random_state)
        x_hat *= -1 
        data_to_move = notcb_test[:int(x_hat)]
        notcb_test = notcb_test[int(x_hat):]
        train_df = pd.concat([notcb_train, cb_train, data_to_move])
        test_df = pd.concat([notcb_test, cb_test])

    else: # take from training, add to test
        np.testing.assert_allclose(ratio:=(y1 + x_hat)/(y2), (x1 - x_hat)/(x2), atol=1e-5)
        notcb_train = notcb_train.sample(frac=1, random_state=random_state)
        data_to_move = notcb_train[:int(x_hat)]
        notcb_train = notcb_train[int(x_hat):]
        train_df = pd.concat([notcb_train, cb_train])
        test_df = pd.concat([notcb_test, cb_test, data_to_move])
    
    return train_df, test_df, x_hat, ratio # the new ratio of noncb to cb (in both sets)

def get_top_n_results(random_search: RandomizedSearchCV, n:int):
    results = pd.DataFrame(random_search.cv_results_) 
    results.sort_values(by='mean_test_score', inplace=True, ascending=False)
    top_n = results.head(n)
    top_n_params = list(top_n['params'])
    for d in top_n_params:
        d['vectorizer'] = d['vectorizer'].__class__.__name__
        if not isinstance(d['vectorizer__preprocessor'], (type(None))):
            d['vectorizer__preprocessor'] = d['vectorizer__preprocessor'].__name__ 
        else: 
            d['vectorizer__preprocessor'] = None

    top_n.loc[:,'params'] = top_n_params
    return top_n[['params', 'mean_test_score', 'std_test_score', 'mean_fit_time']]
# BEGIN

if __name__ == '__main__':
    
    logging.info('Starting Program')    

    path_to_data = Path(sys.argv[1])

    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer('english')
    porter = PorterStemmer()
    lancaster = LancasterStemmer()

    def pp_WordNetLemmatizer(text):
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    def pp_SnowballStemmer(text):
        return ' '.join([stemmer.stem(word) for word in text.split()])

    def pp_PorterStemmer(text):
        return ' '.join([porter.stem(word) for word in text.split()])

    def pp_LancasterStemmer(text):
        return ' '.join([lancaster.stem(word) for word in text.split()])

    # Define hyperparameter search space
    default_xgb_param_dists = {
        'vectorizer': [CountVectorizer(), TfidfVectorizer()],
        'vectorizer__ngram_range': [(1,1), (1,2), (1,3)],
        'vectorizer__analyzer': ['word', 'word', 'word', 'word', 'word', 'char', 'char_wb'],
        'vectorizer__max_df': uniform(0.299, 0.7),
        'vectorizer__preprocessor': [None, pp_WordNetLemmatizer, pp_SnowballStemmer, pp_PorterStemmer, pp_LancasterStemmer],
        'vectorizer__min_df': [1, 1, 2, 3],
        'classifier__booster': ['gbtree', 'dart'],
        'classifier__max_depth': randint(4, 20),
        'classifier__learning_rate': uniform(0.01, 0.5),
        'classifier__n_estimators': randint(100, 500),
        'classifier__min_child_weight': uniform(0, 4),
        'classifier__subsample': uniform(0.5, 0.5),
        'classifier__colsample_bytree': uniform(0.5, 0.5),
        'classifier__gamma': uniform(0, 1),
        'classifier__reg_alpha': loguniform(1e-2, 10),
        'classifier__reg_lambda': loguniform(1e-2, 10),
        'classifier__colsample_bynode': uniform(0.5, 0.5),
        'classifier__random_state': randint(1, 10000)
    }

    default_catboost_param_dists = {
        'vectorizer': [CountVectorizer(), TfidfVectorizer()],
        'vectorizer__ngram_range': [(1,1), (1,2), (1,3)],
        'vectorizer__analyzer': ['word', 'word', 'word', 'word', 'word', 'char', 'char_wb'],
        'vectorizer__max_df': uniform(0.299, 0.5),
        'vectorizer__preprocessor': [None, pp_WordNetLemmatizer, pp_SnowballStemmer, pp_PorterStemmer, pp_LancasterStemmer],
        'vectorizer__min_df': [1, 1, 2, 3, 4],
        'classifier__depth': randint(3, 10),
        'classifier__learning_rate': uniform(0.001, 0.3),
        'classifier__n_estimators': randint(50, 1200),
        'classifier__min_data_in_leaf': randint(1, 20),
        'classifier__grow_policy': ['SymmetricTree'],
        'classifier__score_function': ['L2', 'Cosine'],
        'classifier__colsample_bylevel': uniform(0.5, 0.5),
        'classifier__l2_leaf_reg': uniform(0, 6),
        'classifier__random_state': randint(1, 10000)
    }

    ## TEST 1 ##

    train_df = pd.read_csv(path_to_data / TRAIN_DATA_NAME)
    test_df = pd.read_csv(path_to_data / TEST_DATA_NAME)

    train_df.dropna(inplace=True)
    train_df.drop_duplicates(inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    test_df.dropna(inplace=True)
    test_df.drop_duplicates(inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    test_df.rename(columns={'class': 'label'}, inplace=True)

    # relabel the data such that the offensive class is cyberbullying and the nonoffensive class is not cyberbullying (the hatespeech class is not included)
    train_df['label'] = train_df['label'].apply(lambda x: 0 if x == 'notcb' else 1)
    test_df['label'] = test_df['label'].map({0: 2, 1: 1, 2: 0})

    # Balance the datasets
    notcb_train = train_df[train_df['label'] == 0]
    cb_train = train_df[train_df['label'] == 1]
    notcb_test = test_df[test_df['label'] == 0]
    cb_test = test_df[test_df['label'] == 1]
    balanced_train, balanced_test, x_hat, ratio = balance_train_and_test(notcb_train, cb_train, notcb_test, cb_test, random_state=RANDOM_SEED)

    logging.info(f"TEST 1:Balanced datasets. x_hat: {x_hat}, ratio: {ratio}")

    # Prepare data for cross-validation
    balanced_train = balanced_train.sample(frac=1, random_state=RANDOM_SEED)

    x_train = balanced_train['tweet']
    y_train = balanced_train['label']

    param_dist = default_xgb_param_dists.copy()
    param_dist['classifier__scale_pos_weight'] = RandomRatio(ratio=ratio, low=-0.17, high=0.8)

    # default pipeline
    pipeline = Pipeline([
        ('vectorizer', 'passthrough'),
        ('classifier', XGBClassifier(objective='binary:logistic', tree_method="hist", device="cuda"))
    ])

    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    rs1_xgb = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=50, cv=skf, 
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_SEED, verbose=3
    )

    rs1_xgb.fit(x_train, y_train)

    top_25 = get_top_n_results(rs1_xgb, 25)

    logging.info(f"TEST 1: Top 25 results for XGBoost:\n{top_25}")

    with open(f'{RUN}_top25_xgb_results.json', 'w+') as f:
        json.dump(top_25.to_dict(), f)

    # Test with Catboost

    pipeline = Pipeline([
        ('vectorizer', 'passthrough'),
        ('classifier', CatBoostClassifier(task_type='GPU', devices='0'))
    ])

    param_dist = default_catboost_param_dists.copy() 
    param_dist['classifier__class_weights'] = RandomRatio2(ratio=ratio, low=-0.17, high=0.8)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    rs2_cat = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=50, cv=skf, 
        scoring='f1_macro', n_jobs=-1, random_state=RANDOM_SEED, verbose=3
    )

    rs2_cat.fit(x_train, y_train)

    top_25 = get_top_n_results(rs2_cat, 25)

    # log the top 5 results
    logging.info(f"TEST 1: Top 25 results for CatBoost:\n{top_25}")

    # Save the top 5 results for Test 1
    with open(f'{RUN}_top25_catboost_results.json', 'w+') as f:
        json.dump(top_25.to_dict(), f)
    

# END PROGRAM