import json

data_path = "/mounts/work/kerem/Projects/CoDA/dataset/"
dataset = "CoDA-clean-easy_20_samples.json"
word_type = 'n'

with open(data_path+dataset) as f_in:
    data = json.load(f_in)

out_file = "/mounts/Users/cisintern/lksenel/Projects/CoDA/investigate_results/heatmaps/samples.txt"
with open(out_file, 'w') as f_out:
    for sample_no, sample in enumerate(data[word_type]):
        f_out.write(f"=========== Sample {sample_no} ===========\n\n")
        candidates = sample['candidates']

        defs = [candidate['definition'] for candidate in candidates]
        contexts = [candidate['contexts'][0].replace(candidate['words_in_contexts'][0], "<XXX>") for candidate in candidates]
        syn_names = [candidate['synset_name'] for candidate in candidates]

        f_out.write("Contexts\n")
        for no, c in enumerate(contexts):
            f_out.write(f"{no+1}) {c}\n")

        f_out.write("\nDefinitions\n")
        for no, (d, name) in enumerate(zip(defs, syn_names)):
            f_out.write(f"{no+1}) {d} ({name})\n")

        f_out.write("\n\n")
