from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
from transformers import XLNetTokenizer, XLNetLMHeadModel
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM

from scipy import spatial
from typing import List
import numpy as np
import datetime
import argparse
import pickle
import torch
import time
import sys
import os
import re
import io

import log

import subprocess as sp

def get_gpu_memory(gpu_id: int):
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values[gpu_id]


logger = log.get_logger('root')
logger.propagate = False

DEFINITION_TOKEN = '<DEF>'
MASK_TOKEN = '<MASK>'
INFERRED_TOKEN = '<INFERRED_TOKEN>'

MODELS = {
    'bert': (BertTokenizer, BertForMaskedLM),
    'roberta': (RobertaTokenizer, RobertaForMaskedLM),
    'gpt-2': (GPT2Tokenizer, GPT2LMHeadModel),
    'T5': (T5Tokenizer, T5ForConditionalGeneration),
    'XLNet': (XLNetTokenizer, XLNetLMHeadModel),
}

def get_patterns(model_type, word_type):  
    if word_type == 'n':
        if model_type == 'mlm':
            return [
                f'{MASK_TOKEN} is {DEFINITION_TOKEN}',
                f'{MASK_TOKEN} means {DEFINITION_TOKEN}',
                f'{MASK_TOKEN} is defined as {DEFINITION_TOKEN}'
            ]
        elif model_type == 'lm':
            return [
                f'{DEFINITION_TOKEN} is the definition of',
#                 f'{DEFINITION_TOKEN} is the definition of a',
#                 f'{DEFINITION_TOKEN} is the definition of an'
            ]
    elif word_type == 'v':
        if model_type == 'mlm':
            return [
                f'Definition of {MASK_TOKEN} is to {DEFINITION_TOKEN}',
                f'To {DEFINITION_TOKEN} is the definition of {MASK_TOKEN}'
            ]
        elif model_type == 'lm':
            return [
                f'To {DEFINITION_TOKEN} is the definition of'
            ]


def load_embeddings(filename: str):
    logger.info('Loading embeddings from %s', filename)
    w2v = DummyDict()
    w2v.update({w: v for w, v in _load_vectors(filename)})
    w2v.vocab = w2v.keys()
    logger.info('Done loading embeddings')
    return w2v

def _load_vectors(path, skip=False):
    with io.open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if skip:
                skip = False
            else:
                index = line.index(' ')
                word = line[:index]
                yield word, np.array([np.float(entry) for entry in line[index + 1:].split()])

class DummyDict(dict):
    pass

class DefinitionProcessor:    
    def __init__(self, model_cls: str, pretrained_model: str, gpu_id: int,
                 max_def_len: int, word_type: str, inferred_embs: bool):
    
        self.model_type = 'mlm' if model_cls in ['bert', 'roberta'] else 'lm'
        tokenizer_cls, model_cls  = MODELS[model_cls]
        self.tokenizer = tokenizer_cls.from_pretrained(pretrained_model)
        self.model = model_cls.from_pretrained(pretrained_model)
        self.model.eval()
        
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if inferred_embs:
            self.inferred_token = INFERRED_TOKEN
            self.tokenizer.add_tokens([self.inferred_token]) 
            self.model.resize_token_embeddings(len(self.tokenizer))
#         self.new_token_ind = 1 # for new tokens overwrite embedding for the the 1. position in the vocabulary which is [unused0] token.
        
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")       

        self.word_type = word_type
        
        self.mask_token = self.tokenizer.mask_token if self.model_type == 'mlm' else None
        self.mask_id = self.tokenizer.mask_token_id if self.model_type == 'mlm' else None
        self.softmax = torch.nn.functional.softmax

        self.gpu_id = gpu_id
        self.max_def_len = max_def_len
    
        self.model.to(self.device)
#         print(f"Model created!. available GPU memory {get_gpu_memory(self.gpu_id)} MB")
        
        
    def process(self, word, definition, options, embedding):
        
        if embedding is not None:
            # INFERRED_TOKEN is the last token in the vocabulary
            self.model.get_input_embeddings().weight[-1] = torch.tensor(embedding).to(self.device)
        
        if self.model_type == 'mlm':
            inputs, options_ids, start_inds, tokenization_lengths, tokenized_words = self._process_context_batch_for_mlm(word, definition, options, embedding)  
            option_scores = self._evaluate_contexts_for_mlm(inputs, options_ids, start_inds, tokenization_lengths)       
            return option_scores, tokenized_words
        
        elif self.model_type == 'lm':
            inputs, start_inds, options_ids, tokenized_words = self._process_context_batch_for_lm(word, definition, options, embedding)  
            option_scores = self._evaluate_contexts_for_lm(inputs, start_inds, options_ids) 
            return option_scores, tokenized_words
        else:
            raise ValueError(f'{self.model_type} is not a valid model type')
            return 0
    
    def _process_context_batch_for_lm(self, word, definition, options, embedding): 
        # Use prediction probability of only the first token for fairness
        if embedding != None:
            options[options.index(word)] = INFERRED_TOKEN
        
        tokenized_words = []
        options_ids = []
        for option in options:
            option = " " + option
            options_ids.append(self.tokenizer.encode(option, add_special_tokens=False)[0])
            tokenized_words.append(self.tokenizer.tokenize(option)[0])
                    
        start_inds = []
        contexts = []
        for pattern_no, pattern in enumerate(get_patterns(self.model_type, self.word_type)): 
            def_ind = pattern.split().index(DEFINITION_TOKEN)
            if def_ind == 0:   
                defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(definition.capitalize())[:self.max_def_len])
            else:
                defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(definition)[:self.max_def_len])
            context = pattern.replace(DEFINITION_TOKEN, defin)
            contexts.append(context)
            start_inds.append(len(self.tokenizer.encode(context))-1)
            
        inputs = self.tokenizer(contexts, padding=True)
        return inputs, start_inds, options_ids, tokenized_words
        
    def _process_context_batch_for_mlm(self, word, definition, options, embedding): 
        # This assumes masked token (target word is always the first word in the pattern) for nouns
        tokenized_words = []
        tokenization_lengths = set()
        options_ids = []
        for option in options:
            if (embedding != None) and (option==word):
                options_ids.append(self.tokenizer.encode(INFERRED_TOKEN, add_special_tokens=False))
                tokenization_lengths.add(1)
            else:
                if self.word_type == 'n':
                    option = option.capitalize()
                elif self.word_type == 'v':
                    option = " " + option
                else:
                    raise ValueError(f'{self.word_type} is not a valid word type')
  
                options_ids.append(self.tokenizer.encode(option, add_special_tokens=False))  
                tokenized_word = self.tokenizer.tokenize(option)
                tokenized_words.append(tokenized_word)
                tokenization_lengths.add(len(tokenized_word))
               
        defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(definition)[:self.max_def_len])
        
        start_inds = []
        contexts = []
        for pattern_no, pattern in enumerate(get_patterns(self.model_type, self.word_type)):
            context = pattern.replace(DEFINITION_TOKEN, defin)
            
            word_ind = pattern.split().index(MASK_TOKEN)
            if self.word_type == 'n' and word_ind != 0:
                raise ValueError('Target word has to be at the beginning of the pattern for nouns')
            
            for token_count in tokenization_lengths:
                masked_context = context.replace(MASK_TOKEN, ' '.join([self.mask_token]*token_count))
                contexts.append(masked_context) 
            
            start_ind = self.tokenizer.encode(masked_context, add_special_tokens=True).index(self.mask_id)
            start_inds.append(start_ind)
            
        inputs = self.tokenizer(contexts, padding=True)
        return inputs, options_ids, start_inds, tokenization_lengths, tokenized_words
        
    
    def _evaluate_contexts_for_lm(self, inputs, start_inds, options_ids): 
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_mask = torch.tensor(inputs['attention_mask']).to(self.device)
        
        pattern_count = len(get_patterns(self.model_type, self.word_type))
        
        start_inds = torch.tensor(start_inds)
        scores = np.zeros([len(start_inds)])
        
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=seq_mask)[0]

        option_scores = np.zeros((pattern_count, len(options_ids)))
        for input_no in range(pattern_count):
            target_location = start_inds[input_no]
    
            scores = output[input_no, target_location, :] # get the scores for predicted words
            probs = self.softmax(scores,dim=0)[options_ids].cpu().numpy()
            option_scores[input_no,:] = probs
        
        return option_scores
    
    def _evaluate_contexts_for_mlm(self, inputs, options_ids, start_inds, tokenization_lengths):  
        pattern_count = len(get_patterns(self.model_type, self.word_type))
        
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_mask = torch.tensor(inputs['attention_mask']).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids, seq_mask) 
        option_scores = np.zeros((pattern_count, len(options_ids)))
        for pattern_no in range(pattern_count):
            for token_len_no, masked_count in enumerate(tokenization_lengths):
                input_no = pattern_no*(len(tokenization_lengths)) + token_len_no

                mask_start_ind = start_inds[pattern_no]
                target_indices = torch.tensor(range(mask_start_ind,mask_start_ind+masked_count))
                   
                softmax_probs = self.softmax(output[0][input_no,tuple(target_indices),:], dim=1)
                
                ids = np.asarray([option_ids.copy() for option_ids in options_ids if len(option_ids) == masked_count])
                inds = [i for i in range(len(options_ids)) if len(options_ids[i]) == masked_count]
     
                for no, masked_ind in enumerate(target_indices):  
                    new_probs = softmax_probs.data[no,tuple(ids[:,no])].cpu().numpy() 
                    option_scores[pattern_no, inds] += np.log(new_probs) # take average log probabilities instead of probabilities  
                
                option_scores[pattern_no, inds] = option_scores[pattern_no, inds] / masked_count
        return option_scores

    def _write_to_file(self, f, context_scores):
        for context_score in context_scores:
            for token_score in context_score:                   
                f.write(str(token_score) + "\t")

    def _truncate(self, context_tokens: List[str]):
        # for the addition of cls and sep tokens that occurs later
        while len(context_tokens) > (self.max_seq_len-2):
            mask_idx = context_tokens.index(self.mask_token)
            if mask_idx > (len(context_tokens) - 1) / 2:
                del context_tokens[0]
            else:
                del context_tokens[-1]


    def _get_prediction_rank(self, probs, target_id):
        sort_inds = torch.argsort(probs, descending=True)
        ranks = torch.argsort(sort_inds)
        return 1/(ranks[target_id].item() + 1)


    def _mask_context(self, word, context):
        tokenization_length = len(self.tokenizer.tokenize(word))
 
        masked_context = []
        for context_word in context.split():
            if context_word == word:
                masked_context.extend([self.mask_token] * tokenization_length)
            else:
                masked_context.append(context_word)
        return " ".join(masked_context)
    
if __name__ == "__main__":
    main(sys.argv[1:])
    


