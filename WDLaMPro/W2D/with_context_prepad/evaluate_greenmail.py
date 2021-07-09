import argparse
import os
os.environ['TORCH_HOME']='/mounts/work/kerem/.cache/torch/'

import json
import sys
import time
import nltk
from nltk.corpus import wordnet as wn

import numpy as np
import torch
import random

import pickle

import log
import definition_processor
from definition_processor import DefinitionProcessor, pattern, DEFINITION_TOKEN

sys.path.insert(1, '/mounts/work/kerem/gitlab_projects/bertram_with_informative_contexts/context_evaluator')
import context_evaluator
from context_evaluator import ContextEvaluator

# import subprocess
# BashCommand = "export TORCH_HOME=/mounts/work/kerem/.cache/torch/"
# process = subprocess.Popen(BashCommand.split(),
#                      stdout=subprocess.PIPE, 
#                      stderr=subprocess.PIPE)
# stdout, stderr = process.communicate()


logger = log.get_logger('root')
logger.propagate = False

class SampleResult:
    def __init__(self, word, tokenized_word, prediction_score, prediction_probs, answer):
        self.word = word
        self.tokenized_word = tokenized_word
        self.prediction_score = prediction_score
        self.prediction_probs = prediction_probs
        self.answer = answer
    
def get_result(word, tokenized_word, prediction_probs, answer):
    option_count = len(prediction_probs)
    prediction_score = np.where(prediction_probs.argsort() == answer)[0][0] / (option_count - 1)
    return SampleResult(word, tokenized_word, prediction_score, prediction_probs, answer)

def get_print_result(results: SampleResult, definitions, option_words, answer):
    print_str = "===============================================================================================\n"
    print_str += f'\nWORD: {results.word} ({results.tokenized_word}),     Number of Definitions: {len(definitions)}\n\n'       
    print_str += f'\nTrue Definition:\n{definitions[answer]}\n\n'   
    print_str += f'Prediction Rank Score: {results.prediction_score:.2f}\n\n'
   
    ranks = (-results.prediction_probs).argsort().argsort()

    print_str += f'Definitions (True={answer+1}):\n'
    for def_no, definition in enumerate(definitions):
        print_str += f'\n{def_no+1}) {definition} ({option_words[def_no]})\n'
        print_str += f'Prediction Prob: {results.prediction_probs[def_no]}, Prediction Rank: {ranks[def_no]+1}\n'
    print_str += '\n'
        
    return print_str

def get_candidates(syn):
    definition_set = set()
    option_names = set()
    options = {}

    # Add the target first
    target_name = syn.name().split('.')[0].replace("_", " ")
    target_def = syn.definition()
    options[syn.name()] = target_def
    option_names.add(target_name)
    definition_set.add(target_def)

    for hypernym in syn.hypernyms():
        for option in hypernym.hyponyms():
            option_name = option.name().split('.')[0].replace("_", " ")
            definition = option.definition()
            if (option_name not in option_names) and (definition not in definition_set):
                options[option.name()] = definition
                option_names.add(option_name)
                definition_set.add(definition)   

    definitions = list(options.values())
    option_words = list(options.keys())
    
    return option_words, definitions

def mask_contexts( contexts: [str], word: str) -> [str]:
    # All CE models use bert mask token
    mask_token = "[MASK]"
    masked_contexts = [context.replace(word, mask_token) for context in contexts]
    return masked_contexts
    


if __name__ == '__main__':

    model_cls = 'gpt-2'
    model_name = 'gpt2-xl'
    output_folder = '/mounts/work/kerem/gitlab_projects/definition_benchmark/W2D/with_context_prepad/results/greenmail_results'
    contexts_file = '/mounts/work/kerem/gitlab_projects/definition_benchmark/W2D/with_context_prepad/greenmail_contexts.txt'
    max_batch_size = 5
    gpu_id = 7
    seed = 42

    random.seed(seed)
   
    model_type = 'lm'

    def_processor = DefinitionProcessor(
        model_cls=model_cls,
        pretrained_model=model_name,
        gpu_id=gpu_id,
        max_batch_size=max_batch_size,
        inferred_embs=None
    )

    synset = 'greenmail.n.01'
    embedding = None

    synset = wn.synset(synset)
    
    option_words, definitions = get_candidates(synset)
    answer = option_words.index(synset.name())
    
    synset_word = synset.name().split('.')[0].replace("_", " ")  

    with open(contexts_file) as context_file:
        contexts = context_file.read().splitlines()
        
    context_count = len(contexts)

    prediction_probs, tokenized_word = def_processor.process(synset_word, definitions, embedding, '')
    prediction_score = np.where(prediction_probs.argsort() == answer)[0][0] / (len(option_words) - 1)
    
    base_result = get_result(synset.name(), tokenized_word, prediction_probs, answer)
    
    results = []
    prediction_scores = []
    for no, context in enumerate(contexts):
        prediction_probs, tokenized_word = def_processor.process(synset_word, definitions, embedding, context)
        prediction_score = np.where(prediction_probs.argsort() == answer)[0][0] / (len(option_words) - 1)
        prediction_scores.append(prediction_score)

        results.append(get_result(synset.name(), tokenized_word, prediction_probs, answer))
    
    sort_indices = (-np.asarray(prediction_scores)).argsort()
    
    print_file = os.path.join(output_folder, f'{model_name}_{synset.name()}_{context_count}_{base_result.prediction_score:.2f}.txt')
    with open(print_file, "w", encoding='UTF-8') as f_out:
        f_out.write(get_print_result(base_result, definitions, option_words, answer))

        f_out.write('\n=====================================================\n')
        f_out.write('Rank Scores when single contexts are prepadded:\n\n')
        for no, ind in enumerate(sort_indices):
            f_out.write(f'{no+1}) RS: {prediction_scores[ind]:.2f}\tContext: {contexts[ind].strip()}\n')
        
        f_out.write('\n=====================================================\n')
        f_out.write('Detailed results for single contexts:\n\n')
        for no, ind in enumerate(sort_indices):
            f_out.write(f'{no+1}) RS: {prediction_scores[ind]:.2f}\tContext: {contexts[ind].strip()}\n')
            f_out.write(get_print_result(results[ind], definitions, option_words, answer))
