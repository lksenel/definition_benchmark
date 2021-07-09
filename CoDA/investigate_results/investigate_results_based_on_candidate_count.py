import pickle
import json
import nltk
import numpy as np

results_folder = "/mounts/work/kerem/Projects/CoDA/evaluation_results"
data_folder = "/mounts/work/kerem/Projects/CoDA/dataset"

datasets = [
    "CoDA-clean-hard",
    "CoDA-clean-easy",
    "CoDA-noisy-hard",
    "CoDA-noisy-easy"
]
word_types = ["n", "v"]
model = "gpt2-xl"

candidate_count_results = [[],[],[],[],[],[]]

for dataset in datasets:
    with open(f"{data_folder}/{dataset}.json") as handle:
        data = json.load(handle)
    for word_type in word_types:
        with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results.pickle", 'rb') as handle:
            results = pickle.load(handle)
        
        for sample, result in zip(data[word_type], results):
            score = result['alignment_score']
            sample_count = len(sample['candidates'])
            candidate_count_results[sample_count-5].append(score)

        print(f"\nDataset: {dataset} {word_type}")
        for i in range(6):
            print(f"{i+5} Candidates: {np.mean(candidate_count_results[i]):.2f} ({len(candidate_count_results[i])}) - random: {1/(i+5):.2f}")



