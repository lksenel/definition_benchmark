OUTPUT_FOLDER=/mounts/work/kerem/gitlab_projects/definition_benchmark/W2D/with_context_prepad/results/random_results
GPU_ID=1

MODEL_CLS=gpt-2
MODEL_NAME=gpt2-xl

# EMB_FILE=/mounts/work/kerem/embedding_algorithms/bert_ota/embeddings-wiki-100-diff-bbu-all.txt
CONTEXTS_FILE=/mounts/work/kerem/datasets/WordNet/wordnet_words_max_count_100_contexts.txt
CONTEXT_EVAL_MODEL=/mounts/work/kerem/data/contextevaluator_data/trained_models/without_finetuning_sentence_transformer/with_distillation_loss/rare_words_not_masked/12_layers_12_heads_rl_scores/
# CONTEXT_EVAL_MODEL=/mounts/work/kerem/data/contextevaluator_data/trained_models/without_finetuning_sentence_transformer/with_distillation_loss/older_models/rare_words_not_masked/

MAX_BATCH_SIZE=5

# SYNSETS_FILE=/mounts/work/kerem/gitlab_projects/definition_benchmark/W2D/with_context_prepad/synsets.txt

python evaluate.py \
--model_cls $MODEL_CLS \
--contexts_file $CONTEXTS_FILE \
--context_eval_model $CONTEXT_EVAL_MODEL \
--model_name $MODEL_NAME \
--output_folder $OUTPUT_FOLDER \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \
# --synsets_file $SYNSETS_FILE \
# --gpu_id $GPU_ID \
# --emb_file $EMB_FILE

