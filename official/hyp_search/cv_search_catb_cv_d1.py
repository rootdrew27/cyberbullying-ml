from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_validate, KFold

from nltk.stem.snowball import SnowballStemmer
# from nltk.stem.porter import PorterStemmer

import numpy as np
import pandas as pd

from pathlib import Path
import random
import itertools
import json
import os
import sys
import logging

RANDOM_SEED = 115
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

TRAIN_DATA_NAME = '48000_cyberbullying_tweets_basic_clean.csv' 
FNAME = os.path.basename(__file__).split('.')[0]

logging.basicConfig(
    filename=f'log/{FNAME}.log',
    filemode='w+',
    encoding='utf-8',
    format='{asctime} - {levelname} - {message}',
    level=logging.DEBUG,
    style='{',
    datefmt='%Y-%m-%d %H:%M'
)

if __name__ == '__main__':
    try:
        logging.info(f'Starting: {FNAME}\nTrain data name: {TRAIN_DATA_NAME}\nRandom seed: {RANDOM_SEED}')

        data_path = Path(sys.argv[1])

        train_data_path = data_path / TRAIN_DATA_NAME

        train_df = pd.read_csv(train_data_path)
        # remove duplicates
        train_df = train_df.drop_duplicates()
        # remove rows with missing values
        train_df = train_df.dropna()

        train_df.reset_index(drop=True, inplace=True)
        train_df['label'] = train_df['label'].apply(lambda label : 0 if label == 'notcb' else 1)

        # split data into train and validation sets
        x_train, x_val, y_train, y_val = train_test_split(
            train_df['tweet'],
            train_df['label'],
            test_size=0.1,
            shuffle=True,
            stratify=train_df['label'],
            random_state=RANDOM_SEED
        )

        snowballer = SnowballStemmer('english')
        # porter = PorterStemmer()

        def pp_SnowballStemmer(text):
            return ' '.join([snowballer.stem(word) for word in text.split()])

        # def pp_PorterStemmer(text):
        #     return ' '.join([porter.stem(word) for word in text.split()])

        catb_param_grid = {
            'vectorizer': [CountVectorizer],
            'vectorizer__ngram_range': [(1,1), (1,3)],
            'vectorizer__preprocessor': [pp_SnowballStemmer, None],
            'vectorizer__max_df': [0.5, 0.9],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__n_estimators': [1000],
            'classifier__rsm': [0.75, 1],
            'classifier__depth': [6, 10],
            'classifier': [CatBoostClassifier]
        }

        iters = (v for _, v in catb_param_grid.items())
        keys = catb_param_grid.keys()
        param_sets = [params for params in itertools.product(*iters)]
        # calculate the scale_pos_weight
        # pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        # logging.info(f'The positive weight is {pos_weight:.4f}\n')
        all_results = {
            'fit_time_mean': [],
            'fit_time_std_dev': [],
            'roc_auc_score': [],
            'average_precision': [],
            'f1_macro_mean': [],
            'f1_macro_std_dev': [],
            'f1_weighted_mean': [],
            'f1_weighted_std_dev': [],
            'params': []
        }

        N_SPLITS = 6

        logging.info(f'Starting CV for {len(param_sets)} folds (i.e. {len(param_sets) * N_SPLITS} splits)')
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
            classifier = t['classifier'](early_stopping_rounds=7, eval_metric='Logloss', **classifier_params)

            x_train_copy = vect.fit_transform(x_train.copy())
            x_val_copy = vect.transform(x_val.copy())
            y_train_copy = y_train.copy()
            y_val_copy = y_val.copy()

            skf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

            cv_results: dict = cross_validate(
                classifier,
                x_train_copy,
                y_train_copy,
                cv=skf,
                scoring=['roc_auc_ovr', 'average_precision', 'f1_score', 'f1_weighted'],
                verbose=10,
                return_estimator=False,
                params={'eval_set': [(x_val_copy, y_val_copy)]},
                n_jobs=-1
            )

            logging.info(f'The cv results for run {idx} are:\n{pd.DataFrame(cv_results)}\n')

            all_results['params'].append(t)
            all_results['fit_time_mean'].append(np.mean(cv_results['fit_time']))
            all_results['fit_time_std_dev'].append(np.std(cv_results['fit_time']))
            all_results['roc_auc_score_mean'].append(np.mean(cv_results['roc_auc_score']))
            all_results['roc_auc_socre_std_dev'].append(np.std(cv_results['roc_auc_score']))
            all_results['average_precision_mean'].append(np.mean(cv_results['average_precision']))
            all_results['average_precision_std_dev'].append(np.std(cv_results['average_precision']))
            all_results['f1_macro_mean'].append(np.mean(cv_results['test_f1_macro']))
            all_results['f1_macro_std_dev'].append(np.std(cv_results['test_f1_macro']))
            all_results['f1_weighted_mean'].append(np.mean(cv_results['test_f1_weighted']))
            all_results['f1_weighted_std_dev'].append(np.std(cv_results['test_f1_weighted']))

            with open(f'results/{FNAME}__results.json', 'w+') as f:
                json.dump(all_results, f, default=str)

        logging.info(f'Finished running folds. See the results in {FNAME}__results.json')
    except Exception as e:
        logging.error(f'Error: {e}')
        exit(1)