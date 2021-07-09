import argparse
import os
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

    parser = argparse.ArgumentParser()

    # file parameters
    parser.add_argument('--output_folder', default=None, type=str)

    # parameters for the model to generate definitions
    parser.add_argument('--model_cls', choices=['bert','roberta','gpt-2', 'Transformer-XL', 'XLNet', 'T5'], default='bert')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')

    parser.add_argument('--emb_file', default=None, type=str)

    parser.add_argument('--synsets_file',  type=str, default=None)
    parser.add_argument('--contexts_file', type=str, required=True)

    parser.add_argument('--context_eval_model', type=str, required=True,
                        help="Path to the trained context evaluator model ")
        
    parser.add_argument('--max_batch_size', type=int, default=48, help='maximum batch size')
    
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='id of the gpu that will be used during evaluations') 
    parser.add_argument('--seed', type=int, default=42,
                       help='seed for selecting random train samples for one of few shot evaluation') 
    
    args = parser.parse_args()
    random.seed(args.seed)
   
    model_type = 'mlm' if args.model_cls in ['bert', 'roberta'] else 'lm'         

    def_processor = DefinitionProcessor(
        model_cls=args.model_cls,
        pretrained_model=args.model_name,
        gpu_id=args.gpu_id,
        max_batch_size=args.max_batch_size,
        inferred_embs=args.emb_file!=None
    )

    # CE = ContextEvaluator.load(args.context_eval_model)    
    # CE_device = torch.device(f"cuda:{3}" if torch.cuda.is_available() else "cpu") 
    # CE.to(CE_device)
    # logger.info(f'ContextEvaluator model loaded.')

    embedding=None
    
    if args.synsets_file:
        with open(args.synsets_file) as f_in:   
            synsets = f_in.read().splitlines()
    else:
        min_context_count = 5
        max_context_count = 30
        max_synset_count = 100
        max_option_count = 50
        with open('/mounts/work/kerem/data/definition_benchmark_data/WDNLAMPro.json') as f_data:
            all_synsets = list(json.load(f_data)['n'].keys())

        random.shuffle(all_synsets)
        synsets = all_synsets[:max_synset_count]

    RS = {}
    processed_syn_no = 0
    for syn_no, syn in enumerate(synsets):
        synset = wn.synset(syn)
        
        option_words, definitions = get_candidates(synset)
        answer = option_words.index(synset.name())
        
        synset_word = synset.name().split('.')[0].replace("_", " ")  

        target_word_contexts = []
        with open(args.contexts_file) as context_file:
            nltk_target_word = ' '.join(nltk.word_tokenize(synset_word))
            for line in context_file:
                word = line.split('\t')[0]
                if word == nltk_target_word:
                    target_word_contexts = line.split('\t')[1:max_context_count+1] 
                    break   
          
        # special_result = None
        context_count = len(target_word_contexts)

        if (context_count > min_context_count) and (len(definitions) < max_option_count):
            processed_syn_no += 1
            logger.info(f'({processed_syn_no}/{len(synsets)}) Processing for synset {syn} with {context_count} contexts')       
        else:
            continue

        # masked_contexts = mask_contexts(target_word_contexts)
        # norm_context_weights = torch.nn.functional.softmax(CE(masked_contexts, [context_count]))
        # for no, score in enumerate(norm_context_weights):
        #     logger.info(f'Weight: {score:.4f}\tContext: {masked_contexts[no]}')
        # sys.exit()

        prediction_probs, tokenized_word = def_processor.process(synset_word, definitions, embedding, '')
        prediction_score = np.where(prediction_probs.argsort() == answer)[0][0] / (len(option_words) - 1)
        
        base_result = get_result(synset.name(), tokenized_word, prediction_probs, answer)
        
        
        prediction_scores = []
        for no, context in enumerate(target_word_contexts):
            context = context.strip() + ' . '
            prediction_probs, tokenized_word = def_processor.process(synset_word, definitions, embedding, context)
            # if syn == 'greenmail.n.01' and no == 1:
            #     special_result = get_result(synset.name(), tokenized_word, prediction_probs, answer)
            #     special_context = context
            prediction_score = np.where(prediction_probs.argsort() == answer)[0][0] / (len(option_words) - 1)
            prediction_scores.append(prediction_score)
#             logger.info(f'Context No: {no+1}')

        # Calcualte after padding the correct definition
        definition_pad = pattern.replace(DEFINITION_TOKEN, definitions[answer]) + f' {synset_word}. '
        prediction_probs, tokenized_word = def_processor.process(synset_word, definitions, embedding, definition_pad)
        prediction_score = np.where(prediction_probs.argsort() == answer)[0][0] / (len(option_words) - 1)
        
        sort_indices = (-np.asarray(prediction_scores)).argsort()
        # sort_indices_CE = (-np.asarray(norm_context_weights)).argsort()

        # norm_RS = torch.nn.functional.softmax(prediction_scores)        
        # corr_context_weight = np.corrcoef(norm_RS, norm_context_weights)[0,1]
        # corr_rank_order = np.corrcoef(sort_indices, sort_indices_CE)[0,1]

        # RS[synset_word] = [context_count, base_result.prediction_score, np.mean(prediction_scores), corr_context_weight, corr_rank_order]
        RS[synset_word] = [context_count, base_result.prediction_score, np.mean(prediction_scores)]
        
        print_file = os.path.join(args.output_folder, f'{args.model_name}_{synset.name()}_{context_count}_{base_result.prediction_score:.2f}_{np.mean(prediction_scores):.2f}.txt')
        with open(print_file, "w", encoding='UTF-8') as f_out:
            f_out.write(get_print_result(base_result, definitions, option_words, answer))

            f_out.write('\n=====================================================\n')
            f_out.write('Rank Scores when single contexts are prepadded:\n\n')
            f_out.write(f'0) RS: {prediction_score:.2f}\tContext: {definition_pad}')
            f_out.write('\n------------------------------------------------------------\n')
            for no, ind in enumerate(sort_indices):
                f_out.write(f'CE Rank: {sort_indices_CE.index(ind)+1}\t{no+1}) RS: {prediction_scores[ind]:.2f}\tContext: {target_word_contexts[ind].strip()} . \n')
            
            # f_out.write(f'\n\nCorrelation between normalized CE weights and normalized RS scores: {corr_context_weight}')
            # f_out.write(f'\n\nCorrelation between CE weight ordering and RS ordering: {corr_rank_order}')

            # if special_result:
            #     f_out.write(f'\n\nDetailed results when the following context is padded:\n{special_context}\n\n')
            #     f_out.write(get_print_result(special_result, definitions, option_words, answer))
        with open(os.path.join(args.output_folder, 'RS_results.pickle'), "wb") as f:
                pickle.dump( RS, f)