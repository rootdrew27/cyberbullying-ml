import pandas as pd
import numpy as np

import logging

RUN = 'baseline'

if __name__ == '__main__':
    np.random.seed(115) # Sets the random seed for numpy and scipy functions
    
    logging.basicConfig(filename=f'logs/{RUN}.log', encoding='utf-8', format='{asctime} - {levelname} - {message}', level=logging.INFO, style='{', datefmt='%Y-%m-%d %H:%M')
    
    logging.info('Starting Program')
    
    # Create multiple splits of the dataset and perform cross validation on each train split to find the best hyperparameters for that split.
    
    # Write the results to a log file and a results file
    
    