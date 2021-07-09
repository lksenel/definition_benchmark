import xlwt
from xlwt import Workbook

import pickle
import json
import numpy as np
import os



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

wb = Workbook()
columns = ['sample no', 'definition-context', 'model match score', 'aligned together', 'synset', 'definition', 'context word', 'context', 'correct context']


with open(f"{data_folder}/{dataset}.json") as handle:
    data = json.load(handle)

with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results.pickle", 'rb') as handle:
    results = pickle.load(handle)

with open(f"{results_folder}/human/Timo_{dataset}_{word_type}.pickle", 'rb') as handle:
    [_, timo_results] = pickle.load(handle)

with open(f"{results_folder}/human/Leonie_{dataset}_{word_type}.pickle", 'rb') as handle:
    [_, leonie_results] = pickle.load(handle)

save_path = "/mounts/Users/cisintern/lksenel/Projects/CoDA/investigate_results/heatmaps"
sheet = wb.add_sheet(f'failed_samples')
for col_no, column in enumerate(columns):
    sheet.write(0, col_no, column)

row_no = 0
for sample_no, (sample, result) in enumerate(zip(data[word_type], results)):
    scores = result['target_scores']
    AS = result['alignment_score']
    pred_alignment = result['predicted_alignment']
    
    size = len(pred_alignment)

    best_alignment_score = sum([scores[c_no, d_no] for c_no, d_no in enumerate(pred_alignment)] )

    candidates = sample['candidates']
    grandparent = sample['common_ancestor_info']['ancestor_name']

    for def_no in range(size):
        definition = candidates[def_no]['definition']
        synset = candidates[def_no]['synset_name']

        correct_word_in_context = candidates[def_no]['words_in_contexts'][0]
        correct_context = candidates[def_no]['contexts'][0].replace(correct_word_in_context, '<XXX>')

        for context_no in range(size):
            word_in_context = candidates[context_no]['words_in_contexts'][0]
            context = candidates[context_no]['contexts'][0].replace(word_in_context, '<XXX>')
            
            aligned_together = pred_alignment[context_no] == def_no

            # if (aligned_together and (def_no != context_no)) or ( (def_no == context_no) and not aligned_together):
            row_no += 1
            sheet.write(row_no, 0, f"{sample_no}")
            # sheet.write(row_no, 1, f"{grandparent}")
            # sheet.write(row_no, 2, f"{timo_results[sample_no]:.2f}")
            # sheet.write(row_no, 3, f"{leonie_results[sample_no]:.2f}")
            # sheet.write(row_no, 4, f"{AS:.2f}")
            sheet.write(row_no, 1, f"{def_no+1}-{context_no+1}")
            sheet.write(row_no, 2, f"{scores[context_no,def_no]:.1f}")
            sheet.write(row_no, 3, f"{aligned_together}")
            sheet.write(row_no, 4, f"{synset}")
            sheet.write(row_no, 5, f"{definition}")
            sheet.write(row_no, 6, f"{word_in_context}")
            sheet.write(row_no, 7, f"{context}")
            sheet.write(row_no, 8, f"{correct_word_in_context} - {correct_context}")
    row_no += 1
    sheet.write(row_no, 0, f"Accuracy of Human 1: {timo_results[sample_no]:.2f}")
    row_no += 1
    sheet.write(row_no, 0, f"Accuracy of Human 2: {leonie_results[sample_no]:.2f}")
    row_no += 1
    sheet.write(row_no, 0, f"Accuracy of GPT2-xl: {AS:.2f}")
    row_no += 1
    sheet.write(row_no, 0, f"Common grandparent synset {grandparent}")
wb.save('/mounts/Users/cisintern/lksenel/Projects/CoDA/investigate_results/samples_all_together.xls')