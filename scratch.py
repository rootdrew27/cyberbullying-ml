import pandas as pd
import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold  
from sklearn.pipeline import Pipeline
import random

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

RANDOM_SEED = 115
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

df = pd.read_csv('C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\venv\\cyberbullying-ml\\data\\en_only\\hatespeech_tweets_basic_clean.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
df['class'] = df['class'].apply(lambda x : 1 if x == 0 else 0)
x_train, x_val, y_train, y_val = train_test_split(df['tweet'], df['class'], test_size=.2)

# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', XGBClassifier(objective='binary:logistic', early_stopping_rounds=10, eval_metric='logloss', eval_set=[(x_val, y_val)]))
])

# Define the parameter distribution
param_dist = {
    'vectorizer__max_df': [0.875, 0.9, 0.925, 0.95, 0.975, 1.0, 1.0],
    'vectorizer__min_df': [0.0, 0.0, 0.01, 0.05, 0.1],
    'vectorizer__ngram_range': [(1, 1), (1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
    'vectorizer__analyzer': ['word', 'word', 'word', 'word', 'word', 'char', 'char_wb'],
    'classifier__n_estimators': randint(50, 1000),
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__learning_rate': loguniform(1e-3, 1e-1),
    'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=2,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    random_state=RANDOM_SEED,
    verbose=1
)

# Fit the random search
random_search.fit(
    x_train,
    y_train,
    eval_vectorizer__eval_set=(x_val, y_val),
    eval_vectorizer__pipeline=pipeline,
    # classifier__eval_set=[(it.x_val, y_val)]
)

# Create a dataframe of the results
results_df = pd.DataFrame(random_search.cv_results_)
results_df = results_df.sort_values('rank_test_score')