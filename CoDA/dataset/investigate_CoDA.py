from nltk.corpus import wordnet as wn
import json
import numpy as np
import random
import os

random.seed(42)
np.random.seed(42)

print_count = 2

data_path = "/mounts/work/kerem/Projects/CoDA/dataset"
data_files = [
    'CoDA-clean-hard.json',
    'CoDA-clean-easy.json',
    'CoDA-noisy-hard.json',
    'CoDA-noisy-easy.json',
]

for dataset in data_files:
    input_file = os.path.join(data_path, dataset)
    with open(input_file) as jsonfile:
        CoDA = json.load(jsonfile)    

    noun_groups = CoDA['n']
    noun_random_score = np.mean([1/len(group['candidates']) for group in noun_groups])
    verb_groups = CoDA['v']
    verb_random_score = np.mean([1/len(group['candidates']) for group in verb_groups])

    print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    print(f"Dataset: {dataset}")
    print(f'\nNoun Sample Count: {len(noun_groups)}')
    print(f'\nNoun Random Score: {noun_random_score:.2f}')
    # print(f'mean canddiate count: {np.mean(noun_candidate_counts):.2f} ')
    print(f'\nVerb Sample Count: {len(verb_groups)}')
    print(f'\nVerb Random Score: {verb_random_score:.2f}')
    # print(f'mean canddiate count: {np.mean(verb_candidate_counts):.2f} ')
