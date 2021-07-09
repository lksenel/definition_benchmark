DATA_FOLDER=/mounts/work/kerem/Projects/CoDA/dataset/
NONCE_WORD=bkatuhla

# -------------------------------------- DATASET 1 nouns -------------------------------------s
FILE_NAME=CoDA-clean-easy.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=7

MODEL_CLS=roberta
MODEL_NAME=roberta-large
MAX_BATCH_SIZE=50
WORD_TYPE=n
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results_for_asking/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \
