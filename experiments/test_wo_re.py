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
pathToData = '/data/clean/'
filename1 = 'bullying_light_clean_2.csv'
filename2 = 'hatespeech_light_clean_2.csv'

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

# split data by class
religion = df[df['cyberbullying_type'] == 'religion']
age = df[df['cyberbullying_type'] == 'age']
gender = df[df['cyberbullying_type'] == 'gender']
ethnicity = df[df['cyberbullying_type'] == 'ethnicity']
other_cyberbullying = df[df['cyberbullying_type'] == 'other_cyberbullying']
not_cyberbullying = df[df['cyberbullying_type'] == 'not_cyberbullying']

# undersample data
undersample_value = 5462  
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

df2['class'] = df2['class'].apply(lambda label: 1 if label == 0 else 0)

not_cyberbullying = df2[df2['class'] == 0]

# add the additional 'not cyberbullying' data to the training dataset
not_cyberbullying.rename(columns={'tweet': 'tweet_text', 'class': 'cyberbullying_type'}, inplace=True)
not_cyberbullying.loc[:, 'cyberbullying_type'] = 5 
df = pd.concat([df, not_cyberbullying])



df3 = df3.sample(frac=1, random_state=115) #shuffle
hate = df3[df3['hate label'] == 1]
not_hate = df3[df3['hate label'] == 0]

not_hate = not_hate.sample(3084, random_state=115)
df3 = pd.concat([hate, not_hate])

# VECTORIZE
print('Starting Vectorization')

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
y_test = df3['hate label'] # 1 is hate speech. 0 is not.

best_hyps = [
{'booster': 'dart', 'n_estimators': 600, 'learning_rate': np.float64(0.22151), 'max_depth': np.int64(7), 'min_child_weight': np.int64(28), 'subsample': np.float64(0.6), 'colsample_bytree': np.float64(0.797), 'colsample_bylevel': np.float64(0.824), 'colsample_bynode': np.float64(0.003), 'alpha': np.float64(14.748), 'lambda': np.float64(12.322), 'gamma': np.float64(9.215)},

{'booster': 'gbtree', 'n_estimators': 700, 'learning_rate': np.float64(0.07616), 'max_depth': np.int64(24), 'min_child_weight': np.int64(10), 'subsample': np.float64(0.56), 'colsample_bytree': np.float64(0.802), 'colsample_bylevel': np.float64(0.018), 'colsample_bynode': np.float64(0.081), 'alpha': np.float64(0.744), 'lambda': np.float64(8.253), 'gamma': np.float64(1.869)},

{'booster': 'dart', 'n_estimators': 700, 'learning_rate': np.float64(0.27362), 'max_depth': np.int64(3), 'min_child_weight': np.int64(25), 'subsample': np.float64(0.921), 'colsample_bytree': np.float64(0.6), 'colsample_bylevel': np.float64(0.034), 'colsample_bynode': np.float64(0.046), 'alpha': np.float64(13.56), 'lambda': np.float64(1.598), 'gamma': np.float64(2.889)},

{'booster': 'dart', 'n_estimators': 600, 'learning_rate': np.float64(0.16411), 'max_depth': np.int64(10), 'min_child_weight': np.int64(9), 'subsample': np.float64(0.476), 'colsample_bytree': np.float64(0.099), 'colsample_bylevel': np.float64(0.523), 'colsample_bynode': np.float64(0.033), 'alpha': np.float64(2.662), 'lambda': np.float64(9.187), 'gamma': np.float64(13.668)},

{'booster': 'dart', 'n_estimators': 400, 'learning_rate': np.float64(0.09002), 'max_depth': np.int64(6), 'min_child_weight': np.int64(39), 'subsample': np.float64(0.566), 'colsample_bytree': np.float64(0.182), 'colsample_bylevel': np.float64(0.114), 'colsample_bynode': np.float64(0.229), 'alpha': np.float64(14.869), 'lambda': np.float64(11.521), 'gamma': np.float64(13.942)}
]

print('Starting testing')
with open('results2_wo_re.txt', 'a') as f:
        for i, hyps in enumerate(best_hyps):
               m1 = XGBClassifier(objective='multi:softmax', num_class=6, device='gpu', n_jobs=-1, verbosity=1, random_state=115, **hyps)
               m2 = XGBClassifier(objective='multi:softmax', num_class=6, device='gpu', n_jobs=-1, verbosity=1, random_state=115, **hyps)
               m3 = XGBClassifier(objective='multi:softmax', num_class=6, device='gpu', n_jobs=-1, verbosity=1, random_state=115, **hyps)
               print('Fitting m1')
               m1.fit(x_train_1, y_train)
               print('Fitting m2')
               m2.fit(x_train_2, y_train)
               print('Fitting m3')
               m3.fit(x_train_3, y_train)
               
               m1.save_model(f'models/m1_{i}.json')
               m2.save_model(f'models/m2_{i}.json')
               m3.save_model(f'models/m3_{i}.json')
               
               preds1 = m1.predict(x_test_1)
               preds2 = m2.predict(x_test_2)
               preds3 = m3.predict(x_test_3)
               
               preds1 = [0 if pred == 5 else 1 for pred in preds1]
               preds2 = [0 if pred == 5 else 1 for pred in preds2]
               preds3 = [0 if pred == 5 else 1 for pred in preds3]
               
               report1 = classification_report(y_test, preds1, digits=4)
               report2 = classification_report(y_test, preds2, digits=4)
               report3 = classification_report(y_test, preds3, digits=4)
               
               f.write(f'Hyperparameters:\n\n{hyps}\n\n')
               f.write(f'Classification Report:\n\n{report1}\n\n')
               f.write(f'Classification Report:\n\n{report2}\n\n')
               f.write(f'Classification Report:\n\n{report3}\n\n')
       
       
       
        
       
       