# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# from transformers import TransfoXLTokenizer, TransfoXLLMHeadModel
# from transformers import XLNetTokenizer, XLNetLMHeadModel
# from transformers import BertTokenizer, BertForMaskedLM
# from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import AutoModelWithLMHead, AutoTokenizer

from itertools import permutations
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

logger = log.get_logger('root')
logger.propagate = False

def find_best_allignment(scores: np.array):
    # scores should be n x n  array where n is number of candidates in a synset group
    syn_count = scores.shape[0]
    all_permutations = list(permutations(range(syn_count))) 
    # print(scores.shape)
    # print(scores)

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

class SG_ops:
    def get_first_contexts(candidates: [dict], mask=None):
        """
        return the first context. if mask is given, mask the target word using the mask
        """
        contexts = []
        for candidate in candidates:
            context = candidate['contexts'][0]
            if mask:
                context = context.replace(candidate['words_in_contexts'][0], mask)
            contexts.append(context)
        return contexts

    def get_definitions(candidates: [dict]):
        return [ " " + C['definition'] for C in candidates]


class Aligner:    
    def __init__(self, model_cls: str, pretrained_model: str, gpu_id: int,
                 max_def_len: int, max_batch_size: int, nonce_word: str, word_type: str):
    
        self.context_token = "<CONTEXT>"
        self.nonce_token = "<NONCE>"

        self.model_type = 'mlm' if model_cls in ['bert', 'roberta'] else 'lm'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelWithLMHead.from_pretrained(pretrained_model)
        self.model.eval()
        
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
                
        self.nonce_word = nonce_word
        self.patterns = {
            'n': f"{self.context_token} Definition of {self.nonce_token} is",
            # 'n': f"{self.context_token} {self.nonce_token} is defined as", # Pattern 2
            # 'n': f"{self.context_token} {self.nonce_token} is", # Pattern 3
            'v': f"{self.context_token} Definition of {self.nonce_token} is to"
        }

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")       
        
        self.mask_token = self.tokenizer.mask_token if self.model_type == 'mlm' else None
        self.mask_id = self.tokenizer.mask_token_id if self.model_type == 'mlm' else None
        self.softmax = torch.nn.functional.softmax

        self.word_type = word_type
        self.nonce_word = nonce_word
        self.gpu_id = gpu_id
        self.max_def_len = max_def_len
        self.max_batch_size = max_batch_size
        
        self.model.to(self.device)
        self.verbose = False
        
    def align(self, sample_group: list):
        
        if self.verbose:
            candidate = sample_group['candidates'][0]
            print_str = ''
            print_str += f"Synset: {candidate['synset_name']} ({candidate['words_in_contexts'][0]})\n"
            print_str += f"Definition: {candidate['definition']}\n"
            print_str += f"Context: {candidate['contexts'][0]}\n"
            print(print_str)

        if self.model_type == 'mlm':
            inputs, def_start_inds, def_lengths = self._create_batch_for_mlm(sample_group) 
            word_prediction_scores = self._evaluate_contexts_for_mlm(inputs, def_start_inds, def_lengths)      
            predicted_alignment, aligment_score = find_best_allignment(word_prediction_scores)

            return word_prediction_scores, predicted_alignment, aligment_score
        
        elif self.model_type == 'lm':
            inputs, all_target_ids = self._create_batch_for_lm(sample_group) 
            word_prediction_scores = self._evaluate_contexts_for_lm(inputs, all_target_ids)   
            predicted_alignment, aligment_score = find_best_allignment(word_prediction_scores)

            return word_prediction_scores, predicted_alignment, aligment_score


    def _create_batch_for_mlm(self, sample_group):  

        contexts = SG_ops.get_first_contexts(sample_group['candidates'], self.nonce_word)
        definitions = SG_ops.get_definitions(sample_group['candidates'])
        
        def_start_inds = []
        def_lengths = []
        inputs = []
        for context_no, context in enumerate(contexts):
            inp = self.patterns[self.word_type].replace(self.context_token, context).replace(self.nonce_token, self.nonce_word)
            inp_len = len(self.tokenizer.tokenize(inp))

            for def_no, definition in enumerate(definitions):
                # Below operation removes the space from the beginning of the definition so it is manually added again
                definition = " " + self.tokenizer.convert_tokens_to_string(self.tokenizer.tokenize(definition)[:self.max_def_len])
                def_len = len(self.tokenizer.tokenize(definition))
                def_lengths.append(def_len)

                def_start_ind = inp_len + 1 # assumes one special token at the beginning
                def_start_inds.append(def_start_ind)

                inputs.append(inp + definition)
            
        inputs = self.tokenizer(inputs, padding=True)
        return inputs, np.array(def_start_inds), np.array(def_lengths)
        
        
    def _evaluate_contexts_for_mlm(self, inputs, def_start_inds, def_lengths):       
        unmasked_input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_mask = torch.tensor(inputs['attention_mask']).to(self.device)
        
        sample_count = len(def_start_inds)
        syn_count = int(np.sqrt(sample_count))

        split_inds = np.array_split(list(range(sample_count)), np.ceil(sample_count/self.max_batch_size))
        
        pred_probs = np.zeros([sample_count])
        for mask_iter in range(max(def_lengths)):
            
            input_ids = unmasked_input_ids.clone()  
            target_indices = def_start_inds + mask_iter
            target_ids = []
            for sample_no, target_ind in enumerate(target_indices):
                target_ids.append(input_ids[sample_no, target_ind].clone())
                input_ids[sample_no, target_ind] = self.mask_id
                
            sample_no = 0
            for inds in split_inds:
                input_ids_split = input_ids[inds].to(self.device)
                seq_mask_split = seq_mask[inds].to(self.device)
                target_ids_split = [target_ids[i] for i in inds]
                
                with torch.no_grad():
                    output = self.model(input_ids=input_ids_split, attention_mask=seq_mask_split)[0]

                for sample_no_split in range(len(input_ids_split)):
                    if mask_iter >= def_lengths[sample_no]:
                        sample_no += 1
                        continue
                        
                    target_ind = target_indices[sample_no]
                    target_id = target_ids[sample_no]
                    
                    if self.verbose:
                        print(f'\nMask no: {mask_iter+1}/{max(def_lengths)}')
                        print(f'Sample: {sample_no+1}/{sample_count}')
                        print(self.tokenizer.convert_ids_to_tokens(input_ids_split[sample_no_split]))
                        print(target_ind)
                        print(self.tokenizer.convert_ids_to_tokens([target_id]))
                        sys.exit()

                    prob = self.softmax(output[sample_no_split, target_ind,:], dim=0)[target_id].cpu().numpy()
                    pred_probs[sample_no] += np.log(prob)

                    sample_no += 1
        
        # Normalize by the definition length (skipped because of Big bench)
        # pred_probs = np.divide(pred_probs, def_lengths)
     
        return np.reshape(pred_probs, (syn_count, syn_count))
    
    
    def _create_batch_for_lm(self, sample_group):
        contexts = SG_ops.get_first_contexts(sample_group['candidates'], self.nonce_word)
        definitions = SG_ops.get_definitions(sample_group['candidates'])
        
        all_target_ids = []
        inputs = []
        for context_no, context in enumerate(contexts):
            inp = self.patterns[self.word_type].replace(self.context_token, context).replace(self.nonce_token, self.nonce_word)

            for def_no, definition in enumerate(definitions):
                target_ids = self.tokenizer.encode(definition, add_special_tokens=False)[:self.max_def_len]
                # d = self.tokenizer.decode(target_ids)
                inputs.append(inp)
                all_target_ids.append(target_ids)
            
        inputs = self.tokenizer(inputs, padding=True)
        return inputs, all_target_ids 
    
    def _evaluate_contexts_for_lm(self, inputs, all_target_ids):  
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_mask = torch.tensor(inputs['attention_mask']).to(self.device)
        
        def_lengths = np.array([len(ids) for ids in all_target_ids])
        
        sample_count = len(all_target_ids)
        syn_count = int(np.sqrt(sample_count))
        
        max_gen_count = max([len(ids) for ids in all_target_ids])
        
        split_inds = np.array_split(list(range(sample_count)), np.ceil(sample_count/self.max_batch_size))

        pred_probs = np.zeros([sample_count])
        for gen_no in range(max(def_lengths)):
            target_indices = torch.sum(seq_mask,dim=1)-1
            target_ids = np.ones([sample_count])
            for sample_no in range(sample_count):
                try: # target id is 1 if no target for this generation
                    target_ids[sample_no] = all_target_ids[sample_no][gen_no]
                except:
                    pass
                
            sample_no = 0
            for inds in split_inds:
                input_ids_split = torch.tensor(input_ids)[inds].to(self.device)
                seq_mask_split = torch.tensor(seq_mask)[inds].to(self.device)

                with torch.no_grad():
                    output = self.model(input_ids=input_ids_split, attention_mask=seq_mask_split)[0]

                for sample_no_split in range(len(input_ids_split)):
                    if gen_no >= def_lengths[sample_no]:
                        sample_no += 1
                        continue
                        
                    target_ind = target_indices[sample_no]
                    target_id = int(target_ids[sample_no])
                    
                    if self.verbose:
                        print(f'\nGeneration no: {gen_no+1}/{max(def_lengths)}')
                        print(f'Sample: {sample_no+1}/{sample_count}')
                        print(self.tokenizer.convert_ids_to_tokens(input_ids_split[sample_no_split]))
                        print(target_ind.data)
                        print(self.tokenizer.convert_ids_to_tokens([target_id]))
                        sys.exit()

                    prob = self.softmax(output[sample_no_split, target_ind,:], dim=0)[target_id].cpu().numpy()
                    pred_probs[sample_no] += np.log(prob)

                    # if sample_no == 0:
                        # print(f"\nInput: {self.tokenizer.decode(input_ids_split[0, :(target_ind+1)])}")
                        # print(f"Input ids: {input_ids_split[0, :(target_ind+1)]}")
                        # print(f"Target token: {self.tokenizer.decode([target_ids[0]])}")
                        # print(f"Target token id: {target_ids[0]}")
                        
                        # print(f"Target token score: {np.log(prob)}")
                    sample_no += 1
            
            # Update the input
            input_ids = torch.cat( (input_ids, torch.ones(sample_count,1, dtype=torch.long).to(self.device)*self.tokenizer.pad_token_id), 1)
            for no, ind in enumerate(target_indices):
                input_ids[no,ind+1] = target_ids[no]
            seq_mask = torch.cat( (torch.ones(seq_mask.shape[0],1, dtype=torch.long).to(self.device), seq_mask), 1)
         
        # print(f"\nTotal target score: {pred_probs[0]}")

        # print('All Target Scores:')
        # print(np.reshape(pred_probs, (syn_count, syn_count)))

        # Normalize by the definition length (skipped because of Big bench)
        # pred_probs = np.divide(pred_probs, def_lengths)

        return np.reshape(pred_probs, (syn_count, syn_count))


    def _get_prediction_rank(self, probs, target_id):
        sort_inds = torch.argsort(probs, descending=True)
        ranks = torch.argsort(sort_inds)
        return 1/(ranks[target_id].item() + 1)


    
if __name__ == "__main__":
    main(sys.argv[1:])
    


