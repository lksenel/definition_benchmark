import json
import random
import os


data_folder = "/mounts/work/kerem/Projects/CoDA/dataset"

datasets = [
    "CoDA-clean-hard",
    "CoDA-clean-easy",
    "CoDA-noisy-hard",
    "CoDA-noisy-easy",
]

sample_count = 20

for dataset in datasets:
    with open(os.path.join(data_folder, dataset+'.json'), 'r') as f_in:
        CoDA = json.load(f_in)
    
    CoDA_subset = {}
    for word_type in ['n','v']:
        sample_groups = CoDA[word_type]
        
        random.seed(42)
        random.shuffle(sample_groups)

        # Filter out samples that contains synsets with identical contexts
        clean_sample_groups = []
        no = 0
        while len(clean_sample_groups) < sample_count:
            group = sample_groups[no]
            contexts = []
            for candidate in group['candidates']:
                contexts.append(candidate['contexts'][0])
            
            if len(contexts) == len(set(contexts)):
                clean_sample_groups.append(group)
            no += 1

        CoDA_subset[word_type] = clean_sample_groups

    with open(os.path.join(data_folder, f'{dataset}_{sample_count}_samples.json'), 'w') as f_out:
        CoDA = json.dump(CoDA_subset, f_out)

    