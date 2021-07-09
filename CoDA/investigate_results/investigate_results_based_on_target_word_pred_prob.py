import pickle
import json 
import os


import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer


datasets = [
    # 'CoDA-clean-hard',
    'CoDA-clean-easy',
    # 'CoDA-noisy-hard',
    # 'CoDA-noisy-easy',
]
word_types = ['n']#,'v']
nonce = 'bkatuhla'

gpu_id = '1'

results_dir = "/mounts/work/kerem/Projects/CoDA/evaluation_results"
data_dir = "/mounts/work/kerem/Projects/CoDA/dataset"


# test_model = 'gpt2-xl'
# model_type = 'lm'
test_model = 'roberta-large'
model_type = 'mlm'

softmax = torch.nn.functional.softmax

tokenizer = AutoTokenizer.from_pretrained(test_model)
model = AutoModelWithLMHead.from_pretrained(test_model)
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")     
model.to(device)

all_probs = dict()

for dataset in datasets:
    all_probs[dataset]  = dict()
    # load dataset
    with open(os.path.join(data_dir, dataset + '.json'), 'r') as handle:
        CoDA = json.load(handle)

    for word_type in word_types:
        all_probs[dataset][word_type] = [[],[]]
        sample_groups = CoDA[word_type]
        
        print(f"{dataset} {word_type}")
        # load results
        with open(os.path.join(results_dir, test_model, f"{test_model}_on_{dataset}_{word_type}_nonce_{nonce}_results.pickle"), 'rb') as handle:
            results = pickle.load(handle)


        for group_no, sample_group in enumerate(sample_groups):
            predicted_alignment = results[group_no]['predicted_alignment']

            for candidate_no, candidate in enumerate(sample_group['candidates']):
                correctness = int(predicted_alignment[candidate_no] == candidate_no)
                context = candidate['contexts'][0]
                target_word = candidate['words_in_contexts'][0]
                target_ids = tokenizer.encode(' ' + target_word, add_special_tokens=False)

                if model_type == 'mlm':
                    context = context.replace(target_word, ' '.join([tokenizer.mask_token]*len(target_ids)))
                    inputs = tokenizer([context])

                    input_ids = torch.tensor(inputs['input_ids']).to(device)
                    seq_mask = torch.tensor(inputs['attention_mask']).to(device)

                    target_inds = (input_ids[0] == tokenizer.mask_token_id).nonzero()

                    with torch.no_grad():
                        output = model(input_ids=input_ids, attention_mask=seq_mask)[0][0]
                    
                    prob = 0
                    for no, target_id in enumerate(target_ids):
                        prob += np.log(softmax(output[target_inds[no],:], dim=1)[0,target_id].cpu().numpy())

                    prob = prob/len(target_inds)                

                elif model_type == 'lm':
                    context = context.split(target_word)[0]
                    inputs = tokenizer([context])

                    input_ids = torch.tensor(inputs['input_ids']).to(device)
                    seq_mask = torch.tensor(inputs['attention_mask']).to(device)

                    with torch.no_grad():
                        output = model(input_ids=input_ids, attention_mask=seq_mask)[0]
                    
                    prob = softmax(output[0,-1,:], dim=0)[target_ids[0]]

                all_probs[dataset][word_type][correctness].append(prob)

            if (group_no+1)%20 == 0:
                print(f"Sample group {group_no+1}/{len(sample_groups)}")

        print(f"Average prob for correctly aligned samples: {np.mean(all_probs[dataset][word_type][1])} ({len(all_probs[dataset][word_type][1])})")
        print(f"Average prob for incorrectly aligned samples: {np.mean(all_probs[dataset][word_type][0])} ({len(all_probs[dataset][word_type][0])})")

with open(f"/mounts/Users/cisintern/lksenel/Projects/CoDA/investigate_results/{test_model}_prob_vs_alignment.pickle", 'wb') as handle:
    pickle.dump(all_probs, handle)