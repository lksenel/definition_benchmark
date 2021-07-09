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

columns = ['sample no', 'grandparent', 'Acc. human 1', 'Acc. human 2', 'Acc. GPT2-xl', 'definition-context', 'model match score', 'aligned together', 'synset', 'definition', 'context word', 'context', 'correct context word', 'correct context']


with open(f"{data_folder}/{dataset}.json") as handle:
    data = json.load(handle)

with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results.pickle", 'rb') as handle:
    results = pickle.load(handle)

with open(f"{results_folder}/human/Timo_{dataset}_{word_type}.pickle", 'rb') as handle:
    [_, timo_results] = pickle.load(handle)

with open(f"{results_folder}/human/Leonie_{dataset}_{word_type}.pickle", 'rb') as handle:
    [_, leonie_results] = pickle.load(handle)

save_path = "/mounts/Users/cisintern/lksenel/Projects/CoDA/investigate_results"

f_out = open(save_path+"/samples_incorrect_together.txt", 'w')
for col_no, column in enumerate(columns):
    f_out.write(column+'\t')
f_out.write('\n')

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

            if (aligned_together and (def_no != context_no)) or ( (def_no == context_no) and not aligned_together):
                f_out.write(f"{sample_no}\t")
                f_out.write(f"{grandparent}\t")
                f_out.write(f"{timo_results[sample_no]:.2f}\t")
                f_out.write(f"{leonie_results[sample_no]:.2f}\t")
                f_out.write(f"{AS:.2f}\t")
                f_out.write(f"{def_no+1}-{context_no+1}\t")
                f_out.write(f"{scores[context_no,def_no]:.1f}\t")
                f_out.write(f"{aligned_together}\t")
                f_out.write(f"{synset}\t")
                f_out.write(f"{definition}\t")
                f_out.write(f"{word_in_context}\t")
                f_out.write(f"{context}\t")
                f_out.write(f"{correct_word_in_context}\t")
                f_out.write(f"{correct_context}")
                f_out.write("\n")
f_out.close()