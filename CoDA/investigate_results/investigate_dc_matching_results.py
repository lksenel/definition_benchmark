import pickle
import json
import nltk
import numpy as np

results_folder = "/mounts/work/kerem/Projects/CoDA/evaluation_results"
data_folder = "/mounts/work/kerem/Projects/CoDA/dataset"

datasets = [
    "CoDA-clean-easy",
    "CoDA-clean-easy_20_samples"
]
word_types = ["n"]
model = "gpt2-xl"

candidate_count_results = [[],[],[],[],[],[]]

for dataset in datasets:
    with open(f"{data_folder}/{dataset}.json") as handle:
        data = json.load(handle)
    for word_type in word_types:
        with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results_dc_matching.pickle", 'rb') as handle:
            dc_results = pickle.load(handle)
            
        with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results.pickle", 'rb') as handle:
            alignment_results = pickle.load(handle)
        
        align_scores = []
        matching_scores = []
        for no, (sample, dc_result, align_result) in enumerate(zip(data[word_type], dc_results, alignment_results)):
            align_score = align_result['alignment_score']
            matching_score = dc_result['match_score']

            align_scores.append(align_score)
            matching_scores.append(matching_score)

            if dataset == "CoDA-clean-easy_20_samples":
                print(f"\nSample {no+1}")
                print(f"Alignment Score {align_score:.2f}")
                print(f"Matching Score {matching_score:.2f}")


        print(f"\n\nAverage Alignment Score {np.mean(align_scores):.2f}")
        print(f"Average Matching Score {np.mean(matching_scores):.2f}")
     