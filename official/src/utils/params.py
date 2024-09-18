import pandas as pd
from pathlib import Path
from glob import glob
import json

PATH_TO_PARAMS = Path('C:\\Users\\rooty\\UWEC\\Research\\CyberBullyingML\\cyberbullyingml\\cyberbullying-ml\\official\\hyp_search\\results')

def get_topn_param_sets(algo:str,dataset:str, n:int=10, sort_condition:str='f1_macro_mean'):
    """
    Get the parameter sets from the hyperparameter search.
    """
    try:
        param_sets = pd.DataFrame()
        param_files = glob(str(PATH_TO_PARAMS / f'*{algo}*{dataset}*'))
        for file in param_files:
            with open(file, 'r') as f:
                param_sets = pd.concat([param_sets, pd.DataFrame(json.load(f))])

        return param_sets.sort_values(by=sort_condition, ascending=False).head(n)

    except Exception as e:
        print(f"Error getting param sets: {e}")
        return None