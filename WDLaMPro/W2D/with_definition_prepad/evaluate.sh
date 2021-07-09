OUTPUT_FOLDER=/mounts/work/kerem/gitlab_projects/definition_benchmark/W2D/with_definition_prepad/results/selected_results
GPU_ID=0

MODEL_CLS=gpt-2
MODEL_NAME=gpt2

MAX_BATCH_SIZE=5

SYNSETS_FILE=/mounts/work/kerem/gitlab_projects/definition_benchmark/W2D/with_definition_prepad/synsets.txt

python evaluate.py \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--output_folder $OUTPUT_FOLDER \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \
--synsets_file $SYNSETS_FILE \

