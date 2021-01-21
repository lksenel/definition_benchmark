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
    'Transformer-XL': (TransfoXLTokenizer, TransfoXLLMHeadModel),
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
                f'definition of {MASK_TOKEN} is to {DEFINITION_TOKEN}',
                f'to {DEFINITION_TOKEN} is the definition of {MASK_TOKEN}'
            ]
        elif model_type == 'lm':
            return [
                f'to {DEFINITION_TOKEN} is the definition of'
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
                terms = line.split()
#                 for ind, term in enumerate(terms):
#                     try:
#                         float(term)
#                         break
#                     except:
#                         pass
                # Ugly but faster and better solution
                ind = -768
                word = ' '.join(terms[:ind])
                vec = np.array([np.float(entry) for entry in terms[ind:]])
                yield word, vec

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
            
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")       

        self.mask_token = self.tokenizer.mask_token if self.model_type == 'mlm' else None
        self.mask_id =  self.tokenizer.mask_token_id if self.model_type == 'mlm' else None
        self.softmax = torch.nn.functional.softmax

        self.word_type = word_type
        
        self.gpu_id = gpu_id
        self.max_def_len = max_def_len
    
        self.model.to(self.device)
#         print(f"Model created!. available GPU memory {get_gpu_memory(self.gpu_id)} MB")
    
    
    def process(self, word, option_defs, embedding):
        if embedding is not None:
            # INFERRED_TOKEN is the last token in the vocabulary
            self.model.get_input_embeddings().weight[-1] = torch.tensor(embedding).to(self.device)
            
        
        if self.model_type == 'mlm':
            inputs, start_inds, target_ids = self._process_context_batch_for_mlm(word, option_defs, embedding) 
                
            context_scores = self._evaluate_contexts_for_mlm(inputs, start_inds, target_ids)       
            return context_scores, self.tokenizer.convert_ids_to_tokens(target_ids[0])
        
        elif self.model_type == 'lm':
            inputs, start_inds, target_ids = self._process_context_batch_for_lm(word, option_defs, embedding)  
            context_scores = self._evaluate_contexts_for_lm(inputs, start_inds, target_ids) 
            return context_scores, self.tokenizer.convert_ids_to_tokens(target_ids)

        
    def _process_context_batch_for_mlm(self, word, option_defs, embedding):     
        target_ids = [] # len = pattern_count
        start_inds = [] # len = input_count
        contexts = []
        for pattern_no, pattern in enumerate(get_patterns(self.model_type, self.word_type)):
            word_ind = pattern.split().index(MASK_TOKEN)
            if word_ind == 0:
                process_word = word.capitalize()
            else:
                process_word = " " + word
            target_word_ids = self.tokenizer.encode(process_word, add_special_tokens=False)
            target_ids.append(target_word_ids)
            
            for option_def in option_defs:
                defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(option_def)[:self.max_def_len])
                context = pattern.replace(DEFINITION_TOKEN, defin)
                masked_context = context.replace(MASK_TOKEN, ' '.join([self.mask_token]*len(target_word_ids)))
                contexts.append(masked_context) 
                
                start_ind = self.tokenizer.encode(masked_context, add_special_tokens=True).index(self.mask_id)
                start_inds.append(start_ind)
          
        inputs = self.tokenizer(contexts, padding=True)
  
        return inputs, start_inds, target_ids
  

    def _process_context_batch_for_lm(self, word, option_defs, embedding): 
        target_ids = self.tokenizer.encode( " " + word, add_special_tokens=False)
         
        start_inds = []
        contexts = []
        for pattern_no, pattern in enumerate(get_patterns(self.model_type, self.word_type)):
            def_ind = pattern.split().index(DEFINITION_TOKEN)     
            for option_def in option_defs: 
                if def_ind == 0:
                    defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(option_def.capitalize())[:self.max_def_len])
                else:
                    defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(option_def)[:self.max_def_len])
                context = pattern.replace(DEFINITION_TOKEN, defin)
                contexts.append(context)
                start_inds.append(len(self.tokenizer.encode(context))-1)
                
        inputs = self.tokenizer(contexts, padding=True)                
        return inputs, start_inds, target_ids
    
    
    def _evaluate_contexts_for_mlm(self, inputs, start_inds, target_ids):  
        pattern_count = len(get_patterns(self.model_type, self.word_type))
        def_count = len(start_inds) // pattern_count
        
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_mask = torch.tensor(inputs['attention_mask']).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=seq_mask)[0]
        
        option_scores = np.ones((pattern_count, def_count))
        for pattern_no in range(pattern_count):
            pattern_target_ids = target_ids[pattern_no]
            mask_count = len(pattern_target_ids)
            for def_no in range(def_count):
                input_no = pattern_no*(def_count) + def_no    
                
                mask_start_ind = start_inds[input_no]
                target_indices = torch.tensor(range(mask_start_ind, mask_start_ind+mask_count))
                
                softmax_probs = self.softmax(output[input_no,tuple(target_indices),:], dim=1)
                
                for no, target_id in enumerate(pattern_target_ids):
                    token_prediction_prob = float(softmax_probs.data[no, target_id].item())                
                    option_scores[pattern_no, def_no] = option_scores[pattern_no, def_no]*token_prediction_prob
    
        return option_scores

    
    def _evaluate_contexts_for_lm(self, inputs, start_inds, target_ids):  
        pattern_count = len(get_patterns(self.model_type, self.word_type))
        def_count = len(start_inds) // pattern_count
        
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_mask = torch.tensor(inputs['attention_mask']).to(self.device)
        
        pattern_count = len(get_patterns(self.model_type, self.word_type))
        
        option_scores = np.ones((pattern_count, def_count))
        for gen_no, target_id in enumerate(target_ids):
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=seq_mask)[0]
               
            for pattern_no in range(pattern_count):
                for def_no in range(def_count):
                    input_no = pattern_no*(def_count) + def_no    
                    
                    target_location = sum(seq_mask[input_no]) - 1
                    
                    scores = output[input_no, target_location, :] # get the scores for predicted words
                    prob = self.softmax(scores,dim=0)[target_id].cpu().numpy()
                    option_scores[pattern_no, def_no] = option_scores[pattern_no, def_no]*(prob)
                    
                    if target_location == (len(input_ids[0]) - 1):
                        input_ids[input_no] = torch.cat((input_ids[input_no, 1:], torch.tensor(target_id).unsqueeze(0).to(self.device)),0) 
                    else:
                        input_ids[input_no, target_location + 1] = target_id
                        seq_mask[input_no, target_location + 1] = 1
           
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
    


