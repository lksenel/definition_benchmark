from transformers import AutoModelWithLMHead, AutoTokenizer
import numpy as np
import torch

import sys
import log

logger = log.get_logger('root')
logger.propagate = False


class SG_ops:
    def get_first_contexts(candidates: list(), mask=None, max_len = None):
        """
        return the first context. if mask is given, mask the target word using the mask
        """
        contexts = []
        for candidate in candidates:
            context = candidate['contexts'][0]
            target = candidate['words_in_contexts'][0]
            if mask:
                context = context.replace(target, mask)
                target = mask
            if max_len: # this does not exactly match token count but should be good to provide a limit
                context_words = context.split(' ')
                if len(context_words) > max_len:
                    target_ind = context_words.index(target)
                    right_count = len(context_words) - target_ind - 1
                    m = min(target_ind, right_count)
                    if m >= (max_len//2):
                        context_words = context_words[target_ind-max_len//2:target_ind+max_len//2]
                    else:
                        if target_ind < right_count:
                            context_words = context_words[:max_len]
                        else:
                            context_words = context_words[-max_len:]
                        # print(context_words)
                        # sys.exit()
                    context = ' '.join(context_words)

            contexts.append(context)
        return contexts

    def get_definitions(candidates: list()):
        return [ " " + C['definition'] for C in candidates]


class Evaluator:    
    def __init__(self, model_cls: str, pretrained_model: str, gpu_id: int,
                 max_batch_size: int, nonce_word: str, word_type: str):

        self.definition_token = "<DEFINITION>"
        self.context_token = "<CONTEXT>"
        self.nonce_token = "<NONCE>"

        self.model_type = 'mlm' if model_cls in ['bert', 'roberta'] else 'lm'
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = AutoModelWithLMHead.from_pretrained(pretrained_model)
        self.model.eval()
        
        if self.tokenizer.pad_token == None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
                
        self.nonce_word = nonce_word
        patterns = {
            'n': [f"{self.context_token} Is {self.nonce_token} defined as {self.definition_token}?",
                  f"{self.context_token} Can {self.nonce_token} be defined as {self.definition_token}?",
                  f"{self.context_token} Is {self.nonce_token} {self.definition_token}?"],
            'v': [f"{self.context_token} Is {self.nonce_token} defined as to {self.definition_token}?",
                  f"{self.context_token} Can {self.nonce_token} defined as to {self.definition_token}?",
                  f"{self.context_token} Is {self.nonce_token} to {self.definition_token}?"]
        }
        

        
        self.target_ids = torch.tensor([self.tokenizer.encode(" Yes", add_special_tokens=False)[0], self.tokenizer.encode(" No", add_special_tokens=False)[0]])

        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")       
        
        self.softmax = torch.nn.functional.softmax

        self.max_context_len = 72
        self.word_type = word_type
        self.nonce_word = nonce_word
        self.gpu_id = gpu_id
        self.max_batch_size = max_batch_size
        self.patterns = patterns[self.word_type]

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

        inputs = self._create_batch(sample_group) 
        prediction_scores = self._evaluate_cd_pairs(inputs)      
        
        return prediction_scores


    def _create_batch(self, sample_group):  

        contexts = SG_ops.get_first_contexts(sample_group['candidates'], self.nonce_word, self.max_context_len)
        definitions = SG_ops.get_definitions(sample_group['candidates'])
        
        inputs = []
        for context_no, context in enumerate(contexts):

            for def_no, definition in enumerate(definitions):
                for pattern in self.patterns:
                    inp = pattern.replace(self.context_token, context).replace(self.nonce_token, self.nonce_word).replace(self.definition_token, definition)
                    if self.model_type == "mlm": 
                        inp += f" {self.tokenizer.mask_token}"

                    inputs.append(inp)
            
        inputs = self.tokenizer(inputs, padding=True)
        return inputs
        
        
    def _evaluate_cd_pairs(self, inputs):       
        input_ids = torch.tensor(inputs['input_ids']).to(self.device)
        seq_masks = torch.tensor(inputs['attention_mask']).to(self.device)
        
        target_inds = [sum(seq_mask)-2 if self.model_type == "mlm" else sum(seq_mask)- 1 for seq_mask in seq_masks]
        sample_count = len(target_inds)
        pattern_count = len(self.patterns)
        syn_count = int(np.sqrt(sample_count//pattern_count))

        split_inds = np.array_split(list(range(sample_count)), np.ceil(sample_count/self.max_batch_size))
        
        pred_probs = np.zeros([sample_count])

        sample_no = 0
        for inds in split_inds:
            input_ids_split = input_ids[inds].to(self.device)
            seq_mask_split = seq_masks[inds].to(self.device)
            
            with torch.no_grad():
                output = self.model(input_ids=input_ids_split, attention_mask=seq_mask_split)[0] 
            
            for sample_no_split in range(len(inds)):
                target_ind = target_inds[sample_no]

                logits = output[sample_no_split, target_ind, self.target_ids]
                yes_prob = self.softmax(logits, dim=0)[0].cpu().numpy()

                pred_probs[sample_no] = yes_prob
                sample_no += 1

                # probs_all = self.softmax(output[sample_no_split, target_ind, :], dim=0).cpu()
                # print(f'Sample: {sample_no+1}/{sample_count}')
                # print(self.tokenizer.convert_ids_to_tokens(input_ids_split[sample_no_split]))
                # print(target_ind)
                # print(self.tokenizer.convert_ids_to_tokens(input_ids_split[sample_no_split][target_ind:target_ind+1]))
                # print(self.tokenizer.convert_ids_to_tokens(self.target_ids))


                # print(f"'Yes', 'No' logits: {logits}")
                # print('Only yes or no are accepted answers')
                # print(f"Prob for 'Yes', 'No': {probs[0]}, {probs[1]}")

                # k = 10
                # print(f"If all vocab is accepted")
                # top_ids = torch.argsort(probs_all, dim=0, descending=True)
                # top_probs = probs_all[top_ids[:k]]
                # top_words = self.tokenizer.convert_ids_to_tokens(top_ids[:k])
                # for i in range(k):
                #     print(f"{top_words[i]}: {top_probs[i]:.4f}")
                
                # print(f"Prob for 'Yes', 'No': {probs_all[self.target_ids[0]]}, {probs_all[self.target_ids[1]]}")
                # print(f"Rank for 'Yes', 'No': {(top_ids==self.target_ids[0]).nonzero()}, {(top_ids==self.target_ids[1]).nonzero()}")


        return np.reshape(pred_probs, (syn_count, syn_count, len(self.patterns)))
    
    
if __name__ == "__main__":
    main(sys.argv[1:])
    


