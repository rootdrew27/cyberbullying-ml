from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import StratifiedKFold

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

import numpy as np
import pandas as pd

from pathlib import Path
import random
import itertools
import json
import sys
import logging

RANDOM_SEED = 115
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TRAIN_DATA_NAME = '48000_cyberbullying_tweets_basic_clean.csv' 
TEST_DATA_NAME = 'hatespeech_tweets_basic_clean.csv'

RUN = 'cv_search_xgb_d1'

logging.basicConfig(
    filename=f'log/{RUN}.log',
    filemode='w+',
    encoding='utf-8',
    format='{asctime} - {levelname} - {message}',
    level=logging.DEBUG,
    style='{',
    datefmt='%Y-%m-%d %H:%M'
)

if __name__ == '__main__':

    logging.info('Starting the program...')

    data_path = Path(sys.argv[1])

    train_data_path = data_path / TRAIN_DATA_NAME
    test_data_path = data_path / TEST_DATA_NAME

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    # remove duplicates
    train_df = train_df.drop_duplicates()
    # remove rows with missing values
    train_df = train_df.dropna()

    train_df.reset_index(drop=True, inplace=True)
    train_df['label'] = train_df['label'].apply(lambda label : 0 if label == 'notcb' else 1)

    label2id = {'notcb': 0, 'cb': 1}
    id2label = {0: 'notcb', 1: 'cb'}
    # split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(train_df['tweet'], train_df['label'], test_size=0.05, shuffle=True, random_state=RANDOM_SEED)
    snowballer = SnowballStemmer('english')
    porter = PorterStemmer()

    def pp_SnowballStemmer(text):
        return ' '.join([snowballer.stem(word) for word in text.split()])

    # def pp_PorterStemmer(text):
    #     return ' '.join([porter.stem(word) for word in text.split()])
    
    xgb_param_grid = {
        'vectorizer': [CountVectorizer, TfidfVectorizer],
        'vectorizer__ngram_range': [(1,1), (1,3)],
        'vectorizer__preprocessor': [pp_SnowballStemmer, None],
        'vectorizer__max_df': [0.5, 0.75],
        'classifier__learning_rate': [0.01, 0.1],
        'classifier__n_estimators': [100, 500],
        'classifier__colsample_bytree': [0.5, 0.75],
        'classifier__max_depth': [6, 12],
        'classifier': [XGBClassifier]
    }

    iters = (v for _, v in xgb_param_grid.items())
    keys = xgb_param_grid.keys()
    param_sets = [params for params in itertools.product(*iters)]
    # calculate the scale_pos_weight
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logging.info(f'The positive weight is {pos_weight:.4f}')
    all_results = {'fit_time': [], 'score_time': [], 'test_f1_macro': [], 'test_f1_weighted': [], 'params': []}

    N_SPLITS = 5

    for idx, params in enumerate(param_sets):
        logging.info(f'Starting run {idx}...\n')

        t = {k: v for k, v in zip(keys, params)}

        vect_params = {}
        classifier_params = {}

        for k in keys:
            if 'vectorizer__' in k:
                p_name = k.split('__')[1]
                vect_params[p_name] = t[k]
            if 'classifier__' in k:
                p_name = k.split('__')[1]
                classifier_params[p_name] = t[k]

        vect = t['vectorizer'](**vect_params)
        classifier = t['classifier'](early_stopping_rounds=7, eval_metric='logloss', scale_pos_weight=pos_weight, **classifier_params)

        x_train_copy = vect.fit_transform(x_train)
        x_val_copy = vect.transform(x_val)
        y_train_copy = y_train
        y_val_copy = y_val

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

        cv_results: dict = cross_validate(
            classifier,
            x_train_copy,
            y_train_copy,
            cv=skf,
            scoring=['f1_macro', 'f1_weighted'],
            verbose=10,
            return_estimator=False,
            params={'eval_set':[(x_val_copy, y_val_copy)]},
            n_jobs=-1
        )

        logging.info(f'The cv results for run {idx} are:\n{pd.DataFrame(cv_results)}\n')

        cv_results['params'] = [t for _ in range(N_SPLITS)]
        for key, values in cv_results.items():
            all_results[key].extend(values)

    with open(f'{RUN}__results.json', 'w+') as f:
        json.dump(all_results, f, default=str)