from transformers import GPT2Tokenizer, GPT2LMHeadModel

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
pattern = f'{DEFINITION_TOKEN} is the definition of'
              

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
                 max_batch_size: int, inferred_embs: bool):
            
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)
        self.model = GPT2LMHeadModel.from_pretrained(pretrained_model)
        self.model.eval()
        
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.pad_id = self.tokenizer.encode(self.tokenizer.pad_token)[0]
            
        if inferred_embs:
            self.inferred_token = INFERRED_TOKEN
            self.tokenizer.add_tokens([self.inferred_token]) 
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")       

        self.softmax = torch.nn.functional.softmax

        self.gpu_id = gpu_id
    
        self.max_batch_size = max_batch_size
        
        self.model.to(self.device)
#         print(f"Model created!. available GPU memory {get_gpu_memory(self.gpu_id)} MB")
    
    
    def process(self, word, option_defs, embedding, contexts_to_pad):
        if embedding is not None:
            # INFERRED_TOKEN is the last token in the vocabulary
            self.model.get_input_embeddings().weight[-1] = torch.tensor(embedding).to(self.device)
            
       
        inputs, start_inds, target_ids = self._process_context_batch_for_lm(word, option_defs, embedding, contexts_to_pad)  
        context_scores = self._evaluate_contexts_for_lm(inputs, start_inds, target_ids) 
        return context_scores, self.tokenizer.convert_ids_to_tokens(target_ids)


    def _process_context_batch_for_lm(self, word, option_defs, embedding, contexts_to_pad): 
        target_ids = self.tokenizer.encode( " " + word, add_special_tokens=False)
         
        start_inds = []
        contexts = []
        def_ind = pattern.split().index(DEFINITION_TOKEN)     
        for option_def in option_defs: 
            if def_ind == 0:
                defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(option_def.capitalize()))
            else:
                defin = self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(option_def))
            context = contexts_to_pad + pattern.replace(DEFINITION_TOKEN, defin)
            contexts.append(context)
            start_inds.append(len(self.tokenizer.encode(context))-1)
                
        inputs = self.tokenizer(contexts, padding=True)  
        seq_len = len(inputs['input_ids'][0])
        return inputs, start_inds, target_ids
    
    
    def _evaluate_contexts_for_lm(self, inputs, start_inds, target_ids):  
        option_count = len(start_inds)
        
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_mask = torch.tensor(inputs['attention_mask']).to(self.device)

        input_pad = (torch.ones(len(input_ids), len(target_ids))*self.pad_id).type(torch.LongTensor).to(self.device)
        mask_pad = torch.zeros(len(input_ids), len(target_ids)).type(torch.LongTensor).to(self.device)
        
        input_ids = torch.cat((input_ids, input_pad),1) 
        seq_mask = torch.cat((seq_mask, mask_pad),1)  
        
        seq_len = len(input_ids[0])
#         logger.info(f'Input sequence length is {seq_len} tokens')
        
        split_inds = np.array_split(list(range(option_count)), np.ceil(option_count/self.max_batch_size))
        
        prediction_probs = np.ones([option_count])
        for gen_no, target_id in enumerate(target_ids):  
            sample_no = 0
            for inds in split_inds:
                input_ids_split = input_ids[inds].to(self.device)
                seq_mask_split = seq_mask[inds].to(self.device)
            
                with torch.no_grad():
                    output = self.model(input_ids=input_ids_split, attention_mask=seq_mask_split)[0]

                for sample_no_split in range(len(input_ids_split)):
                    target_ind = sum(seq_mask_split[sample_no_split])-1
                   
                    
                    prob = self.softmax(output[sample_no_split, target_ind,:], dim=0)[target_id].cpu().numpy()   
#                     print(f'{sample_no}) {self.tokenizer.decode(input_ids_split[sample_no_split][:target_ind+2])}')
#                     print(prob)
                    prediction_probs[sample_no] *= prob

                    input_ids[sample_no, target_ind + 1] = target_id
                    seq_mask[sample_no, target_ind + 1] = 1

                    sample_no += 1
           
        return prediction_probs
    

if __name__ == "__main__":
    main(sys.argv[1:])
    


