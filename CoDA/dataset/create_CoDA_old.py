import os
os.environ['HF_HOME'] = '/mounts/work/kerem/.cache/huggingface/'
os.environ['TORCH_HOME'] = '/mounts/work/kerem/.cache/torch'

from sentence_transformers import SentenceTransformer, util
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from nltk.corpus import wordnet as wn
import numpy as np
import pickle
import torch
import json
import random
import sys

import log
logger = log.get_logger('root')
logger.propagate = False


def get_cluster_from_linkage_matrix(Z, min_size, max_size, max_sim_limit):
    sample_count = len(Z)+1
    largest_valid_len = 0
    max_cluster_sim = 0
    return_cluster = None
    clusters = []
    for i, row in enumerate(Z):
        if row[0] < sample_count:
            # if it is an original point
            a = [int(row[0])]
        else:
            # otherwise read the cluster that has been created before
            a = clusters[int(row[0]-sample_count)]
        if row[1] < sample_count:
            b = [int(row[1])]
        else:
            b = clusters[int(row[1]-sample_count)]

        max_sim = row[2]
        cluster = a + b
        clusters.append(cluster)
        if (len(cluster) >= min_size) and (len(cluster) <= max_group_size) and (max_sim < max_sim_limit):
            if len(cluster) > largest_valid_len:
                largest_valid_len = len(cluster)
                return_cluster = cluster
                max_cluster_sim = max_sim
    return return_cluster, max_cluster_sim



def extract_word_locations(scores_file):
    word_locations = {}
    with open(scores_file, 'r', encoding='utf-8') as f_in:
        for line_no, line in enumerate(f_in.read().splitlines()):
            comps = line.split('\t')
            word_locations[comps[0]] = [line_no, len(comps)-1]
    
    return word_locations



def get_all_options(syn_list, level):
    new_candidates = []
    for syn in syn_list:
        new_candidates.extend(syn.hyponyms())
    if level > 1:
        return get_all_options(new_candidates, level-1)
    else:
        return new_candidates


class Sample:
    def __init__(self, synset_name, definition, contexts, words_in_contexts):
        self.synset_name = synset_name
        self.definition = definition
        self.contexts = contexts
        self.words_in_contexts = words_in_contexts



class AlignmentSampleCreator:
    def __init__(self, sentence_model, gpu_id, word_locations, contexts_file, scores_file, 
                 relatedness, context_count, min_context_count, max_similarity, min_group_size, max_group_size):

        self.model = SentenceTransformer(sentence_model)
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.word_locations = word_locations
        self.contexts_file = contexts_file
        self.scores_file = scores_file
        self.relatedness = relatedness
        self.context_count = context_count
        self.min_context_count = min_context_count
        self.max_similarity = max_similarity
        self.min_group_size = min_group_size
        self.max_group_size = max_group_size

        self.used_synsets = set()
    
    def get_alignment_samples_random(self, available_synsets: set):
        if len(available_synsets) < self.max_group_size:
            return None

        sampled_synsets = set(random.sample(available_synsets, self.max_group_size))
        available_synsets.difference_update(sampled_synsets)
        self.used_synsets.update(sampled_synsets)
        candidate_synsets = {}
        for syn in sampled_synsets:
            candidate_synsets[syn] = wn.synset(syn).definition()
        sample_group = self.create_samples_from_synsets(candidate_synsets)
        return [sample_group]

    def get_alignment_samples(self, synset: wn.synset):

        candiate_synsets = self.get_candidate_synsets(synset)
        if len(candiate_synsets) < self.min_group_size:
            return None

        candiate_samples = self.create_samples_from_synsets(candiate_synsets)
        sample_groups = self.get_sample_groups(candiate_samples)
        return sample_groups

    def get_candidate_synsets(self, synset: wn.synset):
        options = {} # dictionary of synsets and their definitions
        synset_names = set()

        all_options = get_all_options([synset], self.relatedness)
        for option in all_options:
            option_name = option.name()
            name = option_name.split('.')[0].replace("_", " ")

            if ((option_name not in self.used_synsets) and   # Synset not used before
                    (name not in synset_names) and           # No duplicate synset names
                    (option_name in self.word_locations) and # synset has contexts
                    (self.word_locations[option_name][1] >= min_context_count)): # synset has minimum 'min_context_count' contexts
                
                options[option_name] = option.definition()
                synset_names.add(name)
        return options

    def create_samples_from_synsets(self, candiate_synsets):
        samples = []
        for (synset, definition) in candiate_synsets.items():
            target_line = self.word_locations[synset][0]

            with open(self.contexts_file, 'r', encoding='utf8') as f_context:
                for i, line in enumerate(f_context):
                    if i == target_line:
                        context_line = line
                        break

            with open(self.scores_file, 'r', encoding='utf8') as f_score:
                for i, line in enumerate(f_score):
                    if i == target_line:
                        score_line = line
                        break
            
            context_comps = context_line.strip().split("\t")
            score_comps = score_line.strip().split("\t")
        
            word = context_comps[0]
            score_word = score_comps[0]    
            assert word == score_word   
            assert word == synset

            all_contexts = context_comps[2::2]
            all_context_targets = context_comps[1::2]
            all_context_scores = list(map(float, score_comps[1:]))

            inds = np.argsort(all_context_scores)

            best_contexts = []
            best_targets = []
            for i in range(self.context_count):
                ind = inds[-(i+1)]
                syn_in_context = all_context_targets[ind]
                best_context = all_contexts[ind]
                # best_context = best_context.replace(syn_in_context, self.nonce_word)

                best_contexts.append(best_context)
                best_targets.append(syn_in_context)

            sample = Sample(
                synset_name=word,
                definition=definition,
                contexts=best_contexts,
                words_in_contexts=best_targets
                )

            samples.append(sample)
        return samples


    def get_sample_groups(self, candidate_samples):
        definitions = [sample.definition for sample in candidate_samples]
        synsets = [sample.synset_name for sample in candidate_samples]

        # get sentence embeddings for options
        definition_embs = self.model.encode(definitions, convert_to_tensor=True, show_progress_bar=False)
        remaining_embs = definition_embs
        remaining_synsets = synsets

        sample_groups = []
        while len(remaining_embs) >= self.min_group_size:      
            cosine_similarity_matrix = 1 - pdist(remaining_embs, metric='cosine')

            Z = linkage(cosine_similarity_matrix, 'complete')
            cluster, max_cluster_sim = get_cluster_from_linkage_matrix(Z, self.min_group_size, self.max_group_size, self.max_similarity)

            if cluster != None:
                sample_group = []
                # print(f'\n\nNew synset group of size {len(cluster)} is formed\n')
                for no in cluster:
                    synset = remaining_synsets[no]
                    ind = synsets.index(synset)
                    sample_group.append(candidate_samples[ind])
                    self.used_synsets.add(synset)
                #     print(f'{synset}:\t{syn_group[synset]}')
                # print(f'Maximum inner group similarity is {max_cluster_sim}')
                sample_groups.append(sample_group)

                # Remove the used synsets from the remaining_embs and remaining_synsets
                remaining_embs = remaining_embs[[i not in cluster for i in range(len(remaining_embs))], :]
                remaining_synsets = [remaining_synsets[i] for i in range(len(remaining_synsets)) if i not in cluster]
            else:
                break
        return sample_groups

random.seed(42)

contexts_dataset = 'semcor_plus_omsti'
sentence_model = 'distilbert-base-nli-stsb-mean-tokens'
contex_scoring_model = 'bert-base-uncased'

context_count = 1
min_context_count = 1
min_group_size = 5
max_group_size = 10
max_similarity = 1
candidate_relatedness_level = 0 # 0=random synsets 1=same parent, 2=same grandparent etc.

word_types = ['n','v']
gpu_id = 7
print_step = 5000

output_folder = '/mounts/work/kerem/data/definition_benchmark_data/WDLaMPro_align_context'
# file_name = f'WDLaMPro_align_{contexts_dataset}_{context_count}_contexts_min_{min_context_count}_contexts_max_sim_{max_similarity}_relatedness_{candidate_relatedness_level}'
file_name = f'WDLaMPro_align_{contexts_dataset}_{context_count}_contexts_min_{min_context_count}_contexts_relatedness_{candidate_relatedness_level}'

output_file = f"{output_folder}/{file_name}.pickle"
config_file = f"{output_folder}/{file_name}_details.txt"
# details_file = f"{output_folder}/dataset_number_details.txt"

with open(config_file, 'w', encoding='utf-8') as f_config:
    f_config.write(f'Sentence transformer model: {sentence_model}\n')
    f_config.write(f'Context scoring model: {contex_scoring_model}\n')
    f_config.write(f'Context count per synset: {context_count}\n')
    f_config.write(f'Required minimum context count for a synset: {min_context_count}\n')
    f_config.write(f'Minimum group size: {min_group_size}\n')
    f_config.write(f'Maximum group size: {max_group_size}\n')
    f_config.write(f'Maximum definition similarity within a group: {max_similarity}\n')
    f_config.write(f'Relatedness level for group samples: {candidate_relatedness_level}\n')
 
    f_config.write(f'\n\nSome statistics for the resulting dataset:\n')

WDLAMPro_allign = {}
for word_type in word_types:
    WDLAMPro_allign[word_type] = []

    if word_type == 'n':
        typ = 'noun'
    elif word_type == 'v':
        typ = 'verb'

    contexts_file = f'/mounts/work/kerem/datasets/WSD_Training_Corpora/{contexts_dataset}_{typ}_contexts.txt'
    scores_file = f'/mounts/work/kerem/datasets/WSD_Training_Corpora/{contexts_dataset}_{typ}_contexts_scores_{contex_scoring_model}.txt'
    word_locations = extract_word_locations(scores_file)

    sample_creator = AlignmentSampleCreator(
        sentence_model=sentence_model,
        gpu_id=gpu_id,
        word_locations=word_locations,
        contexts_file=contexts_file,
        scores_file=scores_file,
        relatedness=candidate_relatedness_level,
        context_count=context_count,
        min_context_count=min_context_count,
        max_similarity=max_similarity,
        min_group_size=min_group_size,
        max_group_size=max_group_size
    )

    logger.info(f'Creating for word type {word_type}')

    # Uncoment to create random sample groups
    # available_synsets = set([syn for syn in list(word_locations.keys()) if word_locations[syn][1] >= min_context_count])
    # while len(available_synsets) >= max_group_size:
    #     sample_groups = sample_creator.get_alignment_samples_random(available_synsets)
    #     WDLAMPro_allign[word_type].extend(sample_groups)

    #     if (len(available_synsets)) % 27 == 0:
    #         logger.info(f'{len(available_synsets)} synsets remained')

    # with open(config_file, 'a', encoding='utf-8') as f_config:
    #     f_config.write(f'\n{typ}:\n')
    #     f_config.write(f'Sample Count: {len(WDLAMPro_allign[word_type])}\n')
    #     counts = [len(sample) for sample in WDLAMPro_allign[word_type]]
    #     f_config.write(f'Average number of synsets per sample: {np.mean(counts):.2f}\n')


    all_synsets = list(wn.all_synsets(word_type))
    for sample_no, syn in enumerate(all_synsets):
        sample_groups = sample_creator.get_alignment_samples(syn)
        if sample_groups:
            WDLAMPro_allign[word_type].extend(sample_groups)
        
        if (sample_no+1) % print_step == 0:
            logger.info(f'({sample_no+1}/{len(all_synsets)}) samples processed')

    with open(config_file, 'a', encoding='utf-8') as f_config:
        f_config.write(f'\n{typ}:\n')
        f_config.write(f'Sample Count: {len(WDLAMPro_allign[word_type])}\n')
        counts = [len(sample) for sample in WDLAMPro_allign[word_type]]
        f_config.write(f'Average number of synsets per sample: {np.mean(counts):.2f}\n')

with open(output_file, 'wb') as handle:
    pickle.dump(WDLAMPro_allign, handle, protocol=pickle.HIGHEST_PROTOCOL)

