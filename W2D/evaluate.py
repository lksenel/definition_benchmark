import argparse
import os
import sys
import time
import json

os.environ['HF_HOME'] = '/mounts/work/kerem/.cache/huggingface/'
os.environ['TORCH_HOME'] = '/mounts/work/kerem/.cache/torch'

from nltk.corpus import wordnet as wn

import numpy as np
import torch
import random

import pickle

import results
import definition_processor
from definition_processor import DefinitionProcessor, get_patterns
import log

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

def get_print_result(results: [SampleResult], definitions, option_words, answer, detailed=False):
    print_str = "===============================================================================================\n"
    print_str += f'\nWORD: {results[0].word} ({results[0].tokenized_word}),     Number of Definitions: {len(definitions)}\n\n'
    print_str += f'True Definition:\n{definitions[answer]}\n\n'
    for pattern_no, pattern_results in enumerate(results):    
        print_str += f'Pattern {pattern_no+1}) Prediction Score: {pattern_results.prediction_score:.2f}\n'
   
    print_str += '\n'
    if detailed:
        ranks = []
        for pattern_results in results:
            ranks.append((-pattern_results.prediction_probs).argsort().argsort())

        print_str += f'Definitions (True={answer+1}):\n'
        for def_no, definition in enumerate(definitions):
            print_str += f'\n{def_no+1}) {definition} ({option_words[def_no]})\n'
            for pattern_no, pattern_results in enumerate(results):
                print_str += f'Pattern No: {pattern_no+1} Prediction Prob: {pattern_results.prediction_probs[def_no]}, Prediction Rank: {ranks[pattern_no][def_no]+1}\n'
        print_str += '\n'
        
    return print_str
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # File parameters
    parser.add_argument('--output_folder', default=None, type=str)
    parser.add_argument('--data_file', default=None, type=str)

    # parameters for the model to generate definitions
    parser.add_argument('--model_cls', choices=['bert','roberta','gpt-2', 'Transformer-XL', 'XLNet', 'T5'], default='bert')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')

    parser.add_argument('--emb_file', default=None, type=str)
    
    parser.add_argument('--word_type', type=str, default='n')

    parser.add_argument('--max_def_len', type=int, default=48,
                       help='maximum definition length')
    parser.add_argument('--max_batch_size', type=int, default=48,
                       help='maximum batch size')
        
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='id of the gpu that will be used during evaluations') 
    parser.add_argument('--seed', type=int, default=42,
                       help='seed for selecting random train samples for one of few shot evaluation') 
    
    args = parser.parse_args()
    random.seed(args.seed)
   
    # Load WDLAMPro
    with open(args.data_file) as jsonfile:
        WDLAMPro = json.load(jsonfile) 
    max_sim = float(args.output_folder.split('_')[-1])

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    model_type = 'mlm' if args.model_cls in ['bert', 'roberta'] else 'lm'
    pattern_count = len(get_patterns(model_type, args.word_type))
    
    if args.emb_file:
        model_name = f"{args.emb_file.split('/')[-1].split('.')[0]}_{args.model_name}_{args.word_type}"
    else:
        model_name = f'{args.model_name}_{args.word_type}'
    
    f_out = open(os.path.join(args.output_folder, f'{model_name}_some_results.txt'), "w", encoding='UTF-8')
    f_out_detailed = open(os.path.join(args.output_folder, f'{model_name}_some_detailed_results.txt'), "w", encoding='UTF-8')
    f_out.close()  
    f_out_detailed.close()       

    def_processor = DefinitionProcessor(
        model_cls=args.model_cls,
        pretrained_model=args.model_name,
        gpu_id=args.gpu_id,
        max_def_len=args.max_def_len,
        word_type=args.word_type,
        inferred_embs=args.emb_file!=None
    )
    
    save_step = 500
    embedding = None
    if args.emb_file:
        new_embs = definition_processor.load_embeddings(args.emb_file)
    
    logger.info(f'Evaluating {model_name} model on WDLAMPro max sim {max_sim} - word type {args.word_type}\n')
    all_results = {} 
    processed_count = 0
    for sample_no, (target_syn, candidates) in enumerate(WDLAMPro[args.word_type].items()):
        target_definition = candidates[target_syn]
        target_word = target_syn.split('.')[0].replace("_", " ")

        definitions = list(candidates.values())
        option_words = [word.split('.')[0].replace("_", " ") for word in candidates.keys()]
        answer = option_words.index(target_word)

        if args.emb_file:
            word = target_word
            if word in new_embs:
                embedding = new_embs[word]
            else:
                continue
        
        repeat = int(np.ceil((len(option_words) * pattern_count) / args.max_batch_size))
        inp_definitions = np.array_split(definitions, repeat)
                
        for i in range(repeat):
            if i == 0:
                prediction_probs, tokenized_word = def_processor.process(target_word, list(inp_definitions[i]), embedding)
            else:
                new_probs, _ = def_processor.process(target_word, list(inp_definitions[i]), embedding)
                prediction_probs = np.append(prediction_probs, new_probs, 1)
                    
#             tokenized_word = target_word
#             prediction_probs = np.random.rand(pattern_count, opt_count)
            
        sample_results = []
        for pattern_no in range(prediction_probs.shape[0]):
            sample_results.append(get_result(target_syn, tokenized_word, prediction_probs[pattern_no], answer))


        all_results[target_syn] = sample_results
                        
        if (sample_no + 1) % save_step == 0:     
            logger.info(f'({sample_no+1}/{len(WDLAMPro[args.word_type])}) samples processed')

            if (sample_no + 1) % (save_step*5) == 0:
                with open(os.path.join(args.output_folder, f'{model_name}_results.pickle'), "wb") as handle:
                    pickle.dump(all_results, handle) 
            
            if len(option_words) < 50:
                with open(os.path.join(args.output_folder, f'{model_name}_some_results.txt'), "a", encoding='UTF-8') as f_out:
                    f_out.write(get_print_result(sample_results, definitions, option_words, answer))

                with open(os.path.join(args.output_folder, f'{model_name}_some_detailed_results.txt'), "a", encoding='UTF-8') as f_out:
                    f_out.write(get_print_result(sample_results, definitions, option_words, answer, detailed=True))
      
    with open(os.path.join(args.output_folder, f'{model_name}_results.pickle'), "wb") as handle:
        pickle.dump(all_results, handle) 
        
            
        
    