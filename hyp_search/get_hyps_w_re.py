# This script combines the specific cyberbullying labels of Dataset 1 such that the dataset only cyberbullying and non-cyberbullying labels. 

import pandas as pd
import numpy as np

import re

from scipy.stats.distributions import uniform, randint
import sys

import sklearn.model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import cross_val_score

import time
import random

import xgboost as xgboost
from xgboost import XGBClassifier
import logging