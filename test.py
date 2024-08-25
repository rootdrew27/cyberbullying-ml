import numpy as np
import pandas as pd
from pathlib import Path
import sys

if __name__ == '__main__':

    CLEAN_DATA_PATH = Path('./data/clean/')
    # Specify the training and the testing data
    training_data = CLEAN_DATA_PATH / sys.argv[1] 
    testing_data = CLEAN_DATA_PATH / sys.argv[2]

    # Load the data
    training_data = pd.read_csv(training_data)
    testing_data = pd.read_csv(testing_data)

        

