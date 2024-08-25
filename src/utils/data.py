# Helpers for working with data
import pandas as pd
import numpy as np

def balance_train_and_test(notcb_train:pd.DataFrame, cb_train:pd.DataFrame, notcb_test:pd.DataFrame, cb_test:pd.DataFrame, random_state:int):
    """
    This functions does not relabel the data that is moved!
    """
    x1 = notcb_train.shape[0]
    x2 = cb_train.shape[0]
    y1 = notcb_test.shape[0]
    y2 = cb_test.shape[0]

    x_hat = (y2 * x1 - x2 * y1) / (y2 + x2)

    # check
    np.testing.assert_allclose(ratio:=(y1 + x_hat)/(y2), (x1 - x_hat)/(x2), atol=1e-5)
    if x_hat >= 0: # take from the test and add to training
        notcb_test = notcb_test.sample(frac=1, random_state=random_state)
        data_to_move = notcb_test[:int(x_hat)]
        notcb_test = notcb_test[int(x_hat):]
        train_df = pd.concat([notcb_train, cb_train, data_to_move])
        test_df = pd.concat([notcb_test, cb_test])
    else:
        x_hat *= -1 
        notcb_train = notcb_train.sample(frac=1, random_state=random_state)
        data_to_move = notcb_train[:int(x_hat)]
        notcb_train = notcb_train[int(x_hat):]
        train_df = pd.concat([notcb_train, cb_train])
        test_df = pd.concat([notcb_test, cb_test, data_to_move])
    

    return train_df, test_df, ratio # the new ratio of noncb to cb (in both sets)
    
