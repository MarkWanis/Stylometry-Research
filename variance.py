import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

def find_variance(results_dir: str):
    embeddings = []

    for f in tqdm(Path(results_dir + "Aelfric/chunks").glob('B1.1.1_chunk_0.json')):
        with open(f, 'r') as file:
            data = json.load(file)
            embeddings.append(data[list(data.keys())[0]]['embedding'])

    for f in tqdm(Path(results_dir + "Unknown/chunks").glob('A1.1_chunk_0.json')):
            with open(f, 'r') as file:
                data = json.load(file)
                embeddings.append(data[list(data.keys())[0]]['embedding'])

    return np.sum(np.var(embeddings, axis=0))

def find_items(results_dir: str):
    embeddings = []

    '''with open(Path(results_dir + "Aelfric/chunks/B1.1.1_chunk_0.json"), 'r') as file:
        data = json.load(file)
        for key in data.keys():
            print(key)
        for value in data.values():
            if 'embedding' in value:
                print(data[value])
        #print(data.items())'''
        
    return len(list(Path(results_dir + "Unknown/chunks").glob('*.json')))


print(find_items('data/sorted_embeddings/'))
