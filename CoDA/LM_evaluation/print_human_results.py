import numpy as np
import pickle
import nltk
import os

import sys

class Sample:
    def __init__(self, synset_name, definition, contexts, words_in_contexts):
        self.synset_name = synset_name
        self.definition = definition
        self.contexts = contexts
        self.words_in_contexts = words_in_contexts
        # self.relatedness = relatedness # only available for variable relatedness version

class SampleResult:
    def __init__(self, sample_group, word_prediction_scores, predicted_alignment, aligment_scores):
        self.sample_group = sample_group
        self.word_prediction_scores = word_prediction_scores
        self.predicted_alignment = predicted_alignment 
        self.aligment_scores = aligment_scores


results_dir = "/mounts/work/kerem/Projects/CoDA/evaluation_results/human"

with open(os.path.join(results_dir,"Leonie_CoDA-clean-easy_20_samples_n.pickle"), 'rb') as handle:
    Leonie_results = pickle.load(handle)

with open(os.path.join(results_dir,"Timo_CoDA-clean-easy_20_samples_n.pickle"), 'rb') as handle:
    Timo_results = pickle.load(handle)

corr = np.corrcoef(Leonie_results[1], Timo_results[1])[0,1]
a=1
