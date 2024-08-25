# Test a model trained on Dataset 1, balanced with Dataset 2, with relabeling, on Dataset 3



import sys
import re

import pandas as pd
import numpy as np

from scipy.stats.distributions import uniform, randint

import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sentence_transformers import SentenceTransformer

from xgboost import XGBClassifier

# SET RANDOM SEED 
np.random.seed(115)

# GET DATA

df= pd.read_csv('C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\venv\\cyberbullying-ml\\data\\clean\\bullying_light_clean_2.csv', encoding='utf-8')
df2 = pd.read_csv('C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\venv\\cyberbullying-ml\\data\\clean\\hatespeech_light_clean_2.csv', encoding='utf-8')

df3 = pd.read_csv('C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\venv\\cyberbullying-ml\\data\\clean\\tweet_blm_light_clean.csv', encoding='utf-8')

df.dropna(axis=0, inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

df2.dropna(axis=0, inplace=True)
df2.drop_duplicates(inplace=True)
df2.reset_index(drop=True, inplace=True)

df3.dropna(axis=0, inplace=True)
df3.drop_duplicates(inplace=True)
df3.reset_index(drop=True, inplace=True)

# PREPARE DATA 
# Train and test dataset should each be split 50/50 with regards to cyberbullying and non-cyberbullying labels

# split data by class
religion = df[df['cyberbullying_type'] == 'religion']
age = df[df['cyberbullying_type'] == 'age']
gender = df[df['cyberbullying_type'] == 'gender']
ethnicity = df[df['cyberbullying_type'] == 'ethnicity']
other_cyberbullying = df[df['cyberbullying_type'] == 'other_cyberbullying']
not_cyberbullying = df[df['cyberbullying_type'] == 'not_cyberbullying']

# undersample data
undersample_value = 5462 # (21047 + 6267) / 5  
ethnicity_us = ethnicity.sample(undersample_value)
other_cyberbullying_us = other_cyberbullying.sample(undersample_value)
religion_us = religion.sample(undersample_value)
age_us = age.sample(undersample_value)
gender_us = gender.sample(undersample_value)

df = pd.concat([religion_us, age_us, gender_us, ethnicity_us, other_cyberbullying_us, not_cyberbullying], axis=0)
# convert str labels to integers
df['cyberbullying_type'] = df['cyberbullying_type'].replace({
        'religion': 0,
        'age': 1,
        'gender': 2,
        'ethnicity': 3,
        'other_cyberbullying': 4,
        'not_cyberbullying': 5
})

df.cyberbullying_type = df.cyberbullying_type.apply(lambda x : 0 if x == 5 else 1) # RELABEL
df2['class'] = df2['class'].apply(lambda x: 1 if x == 0 else 0)

not_cyberbullying = df2[df2['class'] == 0]

# add the additional 'not cyberbullying' data to the training dataset
not_cyberbullying.rename(columns={'tweet': 'tweet_text', 'class': 'cyberbullying_type'}, inplace=True)

df = pd.concat([df, not_cyberbullying])

df3 = df3.sample(frac=1, random_state=115) #shuffle
hate = df3[df3['hate label'] == 1]
not_hate = df3[df3['hate label'] == 0]

not_hate = not_hate.sample(3084, random_state=115)
df3 = pd.concat([hate, not_hate])
# VECTORIZE

# BOW
vect1 = CountVectorizer()
x_train_1 = vect1.fit_transform(df.tweet_text)
x_test_1 = vect1.transform(df3.text)

# SBERT
vect2 = SentenceTransformer('all-MiniLM-L6-v2')
x_train_2 = vect2.encode(df.tweet_text.to_list(), show_progress_bar=True)
x_test_2 = vect2.encode(df3.text.to_list(), show_progress_bar=True)

# TF-IDF
vect3 = TfidfVectorizer()
x_train_3 = vect3.fit_transform(df.tweet_text)
x_test_3 = vect3.transform(df3.text)

y_train = df.cyberbullying_type
y_test = df3['hate label']

best_hyps = [
{'booster': 'dart', 'n_estimators': 400, 'learning_rate': np.float64(0.03345), 'max_depth': np.int64(22), 'min_child_weight': np.int64(34), 'subsample': np.float64(0.137), 'colsample_bytree': np.float64(0.002), 'colsample_bylevel': np.float64(0.896), 'colsample_bynode': np.float64(0.222), 'alpha': np.float64(10.75), 'lambda': np.float64(4.575), 'gamma': np.float64(12.998)},

{'booster': 'dart', 'n_estimators': 400, 'learning_rate': np.float64(0.00081), 'max_depth': np.int64(42), 'min_child_weight': np.int64(10), 'subsample': np.float64(0.656), 'colsample_bytree': np.float64(0.878), 'colsample_bylevel': np.float64(0.669), 'colsample_bynode': np.float64(0.994), 'alpha': np.float64(2.772), 'lambda': np.float64(3.873), 'gamma': np.float64(12.306)},

{'booster': 'gbtree', 'n_estimators': 500, 'learning_rate': np.float64(0.01054), 'max_depth': np.int64(45), 'min_child_weight': np.int64(16), 'subsample': np.float64(0.664), 'colsample_bytree': np.float64(0.322), 'colsample_bylevel': np.float64(0.883), 'colsample_bynode': np.float64(0.368), 'alpha': np.float64(5.411), 'lambda': np.float64(9.142), 'gamma': np.float64(2.988)},

{'booster': 'dart', 'n_estimators': 200, 'learning_rate': np.float64(0.00575), 'max_depth': np.int64(45), 'min_child_weight': np.int64(38), 'subsample': np.float64(0.537), 'colsample_bytree': np.float64(0.057), 'colsample_bylevel': np.float64(0.875), 'colsample_bynode': np.float64(0.96), 'alpha': np.float64(14.002), 'lambda': np.float64(7.166), 'gamma': np.float64(6.601)},

{'booster': 'gbtree', 'n_estimators': 700, 'learning_rate': np.float64(0.00811), 'max_depth': np.int64(39), 'min_child_weight': np.int64(8), 'subsample': np.float64(0.872), 'colsample_bytree': np.float64(0.056), 'colsample_bylevel': np.float64(0.977), 'colsample_bynode': np.float64(0.857), 'alpha': np.float64(12.274), 'lambda': np.float64(11.876), 'gamma': np.float64(0.018)}
]

print('Starting testing')
with open('results2_w_re.txt', 'w') as f:
        for i, hyps in enumerate(best_hyps):
               m1 = XGBClassifier(objective='multi:softmax', num_class=6, device='gpu', n_jobs=-1, verbosity=1, random_state=115, **hyps)
               m2 = XGBClassifier(objective='multi:softmax', num_class=6, device='gpu', n_jobs=-1, verbosity=1, random_state=115, **hyps)
               m3 = XGBClassifier(objective='multi:softmax', num_class=6, device='gpu', n_jobs=-1, verbosity=1, random_state=115, **hyps)
               print('Starting fit 1')
               m1.fit(x_train_1, y_train)
               print('Starting fit 2')
               m2.fit(x_train_2, y_train)
               print('Starting fit 3')
               m3.fit(x_train_3, y_train)
               
               m1.save_model(f'models/w_re_m1_{i}.json')
               m2.save_model(f'models/w_re_m2_{i}.json')
               m3.save_model(f'models/w_re_m3_{i}.json')
               
               preds1 = m1.predict(x_test_1)
               preds2 = m2.predict(x_test_2)
               preds3 = m3.predict(x_test_3)
               
               report1 = classification_report(y_test, preds1, digits=4)
               report2 = classification_report(y_test, preds2, digits=4)
               report3 = classification_report(y_test, preds3, digits=4)
               
               f.write(f'Hyperparameters:\n\n{hyps}\n\n')
               f.write(f'Classification Report:\n\n{report1}\n\n')
               f.write(f'Classification Report:\n\n{report2}\n\n')
               f.write(f'Classification Report:\n\n{report3}\n\n')