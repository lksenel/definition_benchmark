import pickle
import json
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import os
import pandas as pd


class Sample:
    def __init__(self, synset_name, definition, contexts, words_in_contexts):#, relatedness):
        self.synset_name = synset_name
        self.definition = definition
        self.contexts = contexts
        self.words_in_contexts = words_in_contexts

results_folder = "/mounts/work/kerem/Projects/CoDA/evaluation_results"
data_folder = "/mounts/work/kerem/Projects/CoDA/dataset"

dataset = "CoDA-clean-easy_20_samples"

word_type = "n"
model = "gpt2-xl"


with open(f"{data_folder}/{dataset}.json") as handle:
    data = json.load(handle)

with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results.pickle", 'rb') as handle:
    results = pickle.load(handle)

with open(f"{results_folder}/human/Timo_{dataset}_{word_type}.pickle", 'rb') as handle:
    [_, timo_results] = pickle.load(handle)

with open(f"{results_folder}/human/Leonie_{dataset}_{word_type}.pickle", 'rb') as handle:
    [_, leonie_results] = pickle.load(handle)

save_path = "/mounts/Users/cisintern/lksenel/Projects/CoDA/investigate_results/heatmaps"
for sample_no, (sample, result) in enumerate(zip(data[word_type], results)):
    scores = result['target_scores']
    AS = result['alignment_score']
    pred_alignment = result['predicted_alignment']

    best_alignment_score = sum([scores[c_no, d_no] for c_no, d_no in enumerate(pred_alignment)] )

    ax = sns.heatmap(scores, annot=True, fmt=".0f")
    ax.set_ylabel("Contexts")
    ax.set_yticklabels(list(range(1,len(pred_alignment)+1)))
    ax.set_xlabel("Definitions")
    ax.set_xticklabels(list(range(1,len(pred_alignment)+1)))
    ax.set_title(f"Alignment Score: {AS:.2f} ({pred_alignment}) \nBest align prob: {best_alignment_score:.0f}, Avg Human AS: {np.mean([timo_results[sample_no], leonie_results[sample_no]]):.2f}")
    
    for text in ax.texts:
        (x,y) = text.get_position()
        def_no = int(x)
        context_no = int(y)

        if def_no == pred_alignment[context_no]: # Check if the current pair is in the predicted alignment
            text.set_fontweight('bold')
            text.set_size(17)

    figure = ax.get_figure()    

    plt.savefig(os.path.join(save_path, f'Sample_{sample_no}.png'))
    plt.clf()
    
