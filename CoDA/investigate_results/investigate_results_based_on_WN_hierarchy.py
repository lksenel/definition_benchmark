from nltk.corpus import wordnet as wn
import numpy as np
import pickle
import json
import nltk
import os
import sys

a = wn.synset("tangerine.n.01")
print(a.definition())
b = wn.synset("clementine.n.01")
print(b.definition())

results_folder = "/mounts/work/kerem/Projects/CoDA/evaluation_results"
data_folder = "/mounts/work/kerem/Projects/CoDA/dataset"



datasets = [
    # "CoDA-clean-hard",
    "CoDA-clean-easy",
    # "CoDA-noisy-hard",
    # "CoDA-noisy-easy"
]
word_types = ["n"]#, "v"]
model = "gpt2-xl"

max_depth_no = 14
# depth_ranges = [6,9,12,15,20]
depth_ranges = [6,7,8,9,13]

for dataset in datasets:
    with open(f"{data_folder}/{dataset}.json") as handle:
        data = json.load(handle)
    for word_type in word_types:
        print(f"\nDataset: {dataset} {word_type}")

        with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results.pickle", 'rb') as handle:
            results = pickle.load(handle)

        candidate_counts = []
        depth_level_scores = []
        for depth_no in range(max_depth_no):
            depth_level_scores.append([])
            candidate_counts.append([])

        for sample, result in zip(data[word_type], results):
            parent_syn = wn.synset(sample["common_ancestor_info"]["ancestor_name"])
            min_depth = min([len(path) for path in parent_syn.hypernym_paths()]) + 2 # add 2 for grandchildren

            candidate_counts[min_depth].append(len(sample["candidates"]))
            depth_level_scores[min_depth].append(result["alignment_score"])

        lower_depth = 3
        total_sample_count = 0
        total_average_candidate_count = 0
        total_average_accuracy = 0
        for depth_no in range(max_depth_no):
            sample_count = len(candidate_counts[depth_no])
            average_candidate_count = np.mean(candidate_counts[depth_no])
            average_accuracy = np.mean(depth_level_scores[depth_no])*100

            if depth_no in depth_ranges:
                print(f'\nDepth range: {lower_depth}-{depth_no-1}')
                print(f'Sample count: {total_sample_count}')
                print(f'Average candidate count: {total_average_candidate_count/total_sample_count:.2f}')
                print(f'Average Accuracy: {total_average_accuracy/total_sample_count:.2f}%')

                lower_depth = depth_no
                total_sample_count = 0
                total_average_candidate_count = 0
                total_average_accuracy = 0

            if sample_count > 0:
                total_sample_count += sample_count
                total_average_candidate_count += sample_count*average_candidate_count
                total_average_accuracy += sample_count*average_accuracy

            # print(f'\nDepth {depth_no}:')
            # print(f'Sample Count: {sample_count}')
            # print(f'Average candidate count: {average_candidate_count:.2f}')
            # print(f'Average Accuracy: {average_accuracy:.2f}%')
