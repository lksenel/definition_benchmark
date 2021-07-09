import argparse
import os
import sys
import time
import json
import pickle

from nltk.corpus import wordnet as wn

from itertools import permutations
import numpy as np
import torch
import random

from evaluator import Evaluator
import log

logger = log.get_logger('root')
logger.propagate = False

def get_print_result(sample_group: dict, group_results: list, nonce_word):
    candidates = sample_group['candidates']
    info = sample_group['common_ancestor_info']

    print_str = "\n===============================================================================================\n"
    print_str += f'Number of Candidates: {len(candidates)}\n\n'

    print_str += f"\nCommon {info['relation']} synset: {info['ancestor_name']}\n{wn.synset(info['ancestor_name']).definition()}\n\n"
    
    for candidate_no, candidate in enumerate(candidates):
        print_str += f"\n{candidate_no+1}) Synset: {candidate['synset_name']} ({candidate['words_in_contexts'][0]})\n"
        print_str += f"Definition: {candidate['definition']}\n"
        print_str += f"Context: {candidate['contexts'][0]}\n"
               
    for pattern_result in group_results:
        print_str += "\n\n"
        print_str += f"Pattern: {pattern_result['pattern']}\n"
        print_str += f"Predicted alignment: {pattern_result['predicted_alignment']}\n"
        print_str += f"Alignment Score: {pattern_result['alignment_score']}\n"
        print_str += f"\nPredicted contexts: {pattern_result['predicted_contexts']}\n"
        print_str += f"DC Match Score: {pattern_result['match_score']}\n"


    return print_str

def find_best_allignment(scores: np.array):
    # scores should be n x n  array where n is number of candidates in a synset group
    syn_count = scores.shape[0]
    all_permutations = list(permutations(range(syn_count))) 

    best_score = -10**10 
    for perm_no, permutation in enumerate(all_permutations):
        score = 0
        for no in range(syn_count):
            score += scores[no, permutation[no]]

        if score > best_score:
            best_score = score
            predicted_alignment = np.array(permutation)
            # print(f"\nPermutation No: {perm_no}\nBest Score: {best_score}\nPredicted alignment: {predicted_alignment}")

    aligment_score = sum(np.array(range(syn_count)) == predicted_alignment) / syn_count
    return predicted_alignment, aligment_score

def find_best_matches(scores: np.array):
    # scores should be n x n  array where n is number of candidates in a synset group
    # First dimension is contexts and second dimension is definitions.
    syn_count = scores.shape[0]

    predicted_contexts = []
    for def_no in range(syn_count):
        context_scores = scores[:,def_no]
        predicted_contexts.append(np.argmax(context_scores))

    predicted_contexts = np.array(predicted_contexts)
    match_score = sum(np.array(range(syn_count)) == predicted_contexts) / syn_count
    # logger.info(f"Context scores for definition {def_no+1}: {context_scores}")
    # logger.info(f"Predicted contexts {predicted_contexts}")
    # sys.exit()
    return np.array(predicted_contexts), match_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file parameters
    parser.add_argument('--output_folder', default=None, type=str, required=True)    
    parser.add_argument('--data_file', default=None, type=str, required=True)  
    
    # parameters for the model to generate definitions
    parser.add_argument('--model_cls', choices=['bert','roberta','gpt-2', 'Transformer-XL', 'XLNet', 'T5'], default='bert')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    
    parser.add_argument('--word_type', choices=['n','v'], default='n')
    
    parser.add_argument('--nonce_word', type=str, default='bimp')
    parser.add_argument('--max_batch_size', type=int, default=48,
                       help='maximum batch size')
    
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='id of the gpu that will be used during evaluations') 
    parser.add_argument('--seed', type=int, default=42,
                       help='seed for selecting random train samples for one of few shot evaluation') 
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    with open(args.data_file, 'r') as handle:
        CoDA = json.load(handle)

    data = args.data_file.split("/")[-1][:-5] # don't take .json

    print_file = f'{args.model_name}_on_{data}_{args.word_type}_nonce_{args.nonce_word}_some_results.txt'
    save_file = f'{args.model_name}_on_{data}_{args.word_type}_nonce_{args.nonce_word}_results.pickle'

    # print_file = f'{args.model_name}_on_{data}_{args.word_type}_nonce_{args.nonce_word}_some_results_pattern_3.txt'
    # save_file = f'{args.model_name}_on_{data}_{args.word_type}_nonce_{args.nonce_word}_results_pattern_3.pickle'


    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    f_out = open(os.path.join(args.output_folder, print_file), "w", encoding='UTF-8')
    f_out.close()    
    
    evaluator = Evaluator(
        model_cls=args.model_cls,
        pretrained_model=args.model_name,
        gpu_id=args.gpu_id,
        max_batch_size=args.max_batch_size,
        nonce_word=args.nonce_word,
        word_type=args.word_type
    )

    # sample_result = {}
    # sample_result[''] = []
    # sample_result['target_scores'] = []
    # sample_result['predicted_alignment'] = []
    # sample_result['alignment_score'] = []

    all_results = []
    sample_groups = CoDA[args.word_type]
    for group_no, sample_group in enumerate(sample_groups):
        target_scores = evaluator.align(sample_group)

        group_results = []
        patterns = evaluator.patterns.copy()
        patterns.append("Ensamble")
        for pattern_no, pattern in enumerate(patterns):
            if pattern_no == len(evaluator.patterns):
                scores = np.mean(target_scores,axis=2)
            else:
                scores = target_scores[:,:,pattern_no]

            predicted_alignment, aligment_score = find_best_allignment(scores)
            predicted_contexts, match_score = find_best_matches(scores)

            sample_result = {}
            sample_result['pattern'] = pattern
            sample_result['target_scores'] = target_scores
            sample_result['predicted_alignment'] = predicted_alignment
            sample_result['alignment_score'] = aligment_score
            sample_result['predicted_contexts'] = predicted_contexts
            sample_result['match_score'] = aligment_score
            

            group_results.append(sample_result)

        all_results.append(group_results)
        
        if (group_no+1) % 20 == 0:     
            logger.info(f'{group_no+1}/{len(sample_groups)} synset groups processed')
                    
        if (group_no+1) % (len(sample_groups) // 20) == 0: 
            with open(os.path.join(args.output_folder, print_file), "a", encoding='UTF-8') as f_out:
                f_out.write(get_print_result(sample_group, group_results, args.nonce_word))

            with open(os.path.join(args.output_folder, save_file), "wb") as handle:
                pickle.dump(all_results, handle)         

    with open(os.path.join(args.output_folder, save_file), "wb") as handle:
        pickle.dump(all_results, handle) 
