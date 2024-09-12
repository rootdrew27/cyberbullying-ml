import json
from pathlib import Path

def append_results_to_json(results, out_path:Path):

    with open(out_path, 'r') as j:
        data = json.load(j)

    data['results'].append(results)

    with open(out_path, 'w') as j:            
        json.dump(data, j)
        

def create_results_file(out_path:Path):
    if out_path.exists():
        print(f"Results file already exists. Not creating new one")
        return
    
    with open(out_path, 'w') as j:
        json.dump({'meta':{'name':out_path.name, 'description':''},'results':[]}, j)

