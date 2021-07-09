import pickle
import json
import nltk

with open('/mounts/work/kerem/datasets/WordNet/wordnet_word_counts_in_WWC.pickle', 'rb') as handle:
    word_counts = pickle.load(handle)

results_folder = "/mounts/work/kerem/Projects/CoDA/evaluation_results"
data_folder = "/mounts/work/kerem/Projects/CoDA/dataset"

datasets = [
    "CoDA-clean-hard",
    "CoDA-clean-easy",
    "CoDA-noisy-hard",
    "CoDA-noisy-easy"
]
word_types = ["n", "v"]
model = "gpt2-xl"

count_limits = [100, 1000]

freq_results = [[],[],[]]

for dataset in datasets:
    with open(f"{data_folder}/{dataset}.json") as handle:
        data = json.load(handle)
    for word_type in word_types:
        with open(f"{results_folder}/{model}/{model}_on_{dataset}_{word_type}_nonce_bkatuhla_results.pickle", 'rb') as handle:
            results = pickle.load(handle)
        
        for sample, result in zip(data[word_type], results):
            for candidate_no, (candidate, alignment) in enumerate(zip(sample['candidates'], result['predicted_alignment'])):
                name = ' '.join(nltk.word_tokenize(candidate['synset_name'].split('.')[0].replace('_', ' ')))
                count = word_counts[name]

                is_correct = int(alignment == candidate_no)
                ind = sum([int(count>=lim) for lim in count_limits])

                freq_results[ind].append(is_correct)

        print(f"\nDataset: {dataset} {word_type}")
        print(f"Rare: {sum(freq_results[0])/len(freq_results[0]):.2f} ({len(freq_results[0])})")
        print(f"Medium: {sum(freq_results[1])/len(freq_results[1]):.2f} ({len(freq_results[1])})")
        print(f"Frequent: {sum(freq_results[2])/len(freq_results[2]):.2f} ({len(freq_results[2])})")


