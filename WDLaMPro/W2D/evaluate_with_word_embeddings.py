import argparse
import os

import nltk
from nltk.corpus import wordnet as wn

from scipy.spatial.distance import cosine
import numpy as np
import random

import pickle

import results
import log

import sys
import time

logger = log.get_logger('root')
logger.propagate = False

class SampleResult:
    def __init__(self, word, tokenized_word, prediction_score, cos_sims, answer):
        self.word = word
        self.tokenized_word = tokenized_word
        self.prediction_score = prediction_score
        self.cos_sims = cos_sims
        self.answer = answer
    
def get_result(word, tokenized_word, cos_sims, answer):
    option_count = len(cos_sims)
    a = np.where(cos_sims.argsort() == answer)[0][0]
    prediction_score = a / (option_count - 1)
    return SampleResult(word, tokenized_word, prediction_score, cos_sims, answer)

def get_print_result(result: SampleResult, definitions, option_words, answer, detailed=False):
    print_str = "===============================================================================================\n"
    print_str += f'\nWORD: {result.word} ({result.tokenized_word}),     Number of Definitions: {len(definitions)}\n\n'
    print_str += f'True Definition:\n{definitions[answer]}\n\n'
    print_str += f'Prediction Score: {result.prediction_score:.2f}\n'
   
    print_str += '\n'
    if detailed:
        ranks = (-result.cos_sims).argsort().argsort()

        print_str += f'Definitions (True={answer+1}):\n'
        for def_no, definition in enumerate(definitions):
            print_str += f'\n{def_no+1}) {definition} ({option_words[def_no]})\n'
            print_str += f'Cosine Similarity: {result.cos_sims[def_no]}, Prediction Rank: {ranks[def_no]+1}\n'
        print_str += '\n'
        
    return print_str
       
    

min_option_count = 5
word_type = "v"
model_name = f'fastText_{word_type}'

output_folder = f"/mounts/work/kerem/data/definition_benchmark_data/W2D/word_embedding_evaluations"

emb_path = "/mounts/work/kerem/embedding_algorithms/fastText/crawl-300d-2M-subword.vec"

class My_Embeddings:
    def __init__(self, emb_path: str):
        self.embs, self.emb_dim = self._load_embeddings(emb_path)
        logger.info(f"Embeddings are loaded from {emb_path}, there are {len(self.embs)} tokens with {self.emb_dim} dimensions")
        
    def get_mean_emb(self, text):
        tokenized_text = nltk.word_tokenize(text)
        tokens = []
        mean_emb = np.zeros(self.emb_dim)
        for token in tokenized_text:
            try:
                mean_emb += self.embs[token]
                tokens.append(token)
            except:
                continue
        
        if len(tokens) == 0:
            raise ValueError(f'Text: {text} does not contain any tokens in the embedding dictionary')
        
        mean_emb = mean_emb/len(tokens)
        return mean_emb, tokens
    
    def can_embed(self, text):
        tokenized_text = nltk.word_tokenize(text)
        return any([(token in self.embs) for token in tokenized_text])
    
    def _load_embeddings(self, path):
        embs = {}
        with open(path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                count += 1
                terms = line.split()
                word = terms[0]
                embs[word] = np.array([np.float(entry) for entry in terms[1:]])
                
#                 if count == 10000:
#                     break
        dim = len(embs[word])
        return embs, dim
    
    
    
emb_model = My_Embeddings(emb_path)

f_out = open(os.path.join(output_folder, f'{model_name}_some_results.txt'), "w", encoding='UTF-8')
f_out_detailed = open(os.path.join(output_folder, f'{model_name}_some_detailed_results.txt'), "w", encoding='UTF-8')
f_out.close()  
f_out_detailed.close()       


   
all_synsets = list(wn.all_synsets(word_type))

save_step = 200
logger.info(f'Choosing definitions using {model_name} model\n')
all_results = {} 
processed_count = 0
for sample_no, syn in enumerate(all_synsets):
    definition_set = set()
    option_names = set()
    options = {}

    # Add the target first
    target_name = syn.name().split('.')[0].replace("_", " ")
    target_def = syn.definition()
    
    if (not emb_model.can_embed(target_name)) or (not emb_model.can_embed(target_def)):
        continue
    
    options[syn.name()] = target_def
    option_names.add(target_name)
    definition_set.add(target_def)

    for hypernym in syn.hypernyms():
        for option in hypernym.hyponyms():
            option_name = option.name().split('.')[0].replace("_", " ")
            definition = option.definition()
            if (option_name not in option_names) and (definition not in definition_set) and (emb_model.can_embed(option_name)) and (emb_model.can_embed(definition)):
                options[option.name()] = definition
                option_names.add(option_name)
                definition_set.add(definition)        

    if len(options) >= min_option_count:   
        processed_count += 1 
        
        word = target_name
        definitions = list(options.values())
        option_words = list(options.keys())    
        answer = option_words.index(syn.name())
            
        target_emb, tokenized_word = emb_model.get_mean_emb(word)  
        cos_sims = []
        for definition in definitions:
            candidate_emb, _ = emb_model.get_mean_emb(definition)   
            cos_sim = 1 - cosine(target_emb, candidate_emb)
            cos_sims.append(cos_sim)
        
        sample_result = get_result(syn.name(), tokenized_word, np.array(cos_sims), answer)  
        all_results[syn.name()] = sample_result

        if (processed_count) % save_step == 0:     
            logger.info(f'Progress ({sample_no+1}/{len(all_synsets)}) {processed_count} samples processed')

            if (processed_count) % (save_step*5) == 0:
                with open(os.path.join(output_folder, f'{model_name}_results.pickle'), "wb") as handle:
                    pickle.dump(all_results, handle) 

            if len(option_words) < 50:
                with open(os.path.join(output_folder, f'{model_name}_some_results.txt'), "a", encoding='UTF-8') as f_out:
                    f_out.write(get_print_result(sample_result, definitions, option_words, answer))
               
                with open(os.path.join(output_folder, f'{model_name}_some_detailed_results.txt'), "a", encoding='UTF-8') as f_out:
                    f_out.write(get_print_result(sample_result, definitions, option_words, answer, detailed=True))

    with open(os.path.join(output_folder, f'{model_name}_results.pickle'), "wb") as handle:
        pickle.dump(all_results, handle) 
        
            
        
    