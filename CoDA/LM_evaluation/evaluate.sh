DATA_FOLDER=/mounts/work/kerem/Projects/CoDA/dataset/
MAX_DEF_LEN=48

NONCE_WORD=bkatuhla
# NONCE_WORD=x
# NONCE_WORD=opyatzel
# NONCE_WORD=orange
# NONCE_WORD=cloud

# FILE_NAME=CoDA-clean-easy.json
FILE_NAME=CoDA-clean-easy_20_samples.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=7

MODEL_CLS=roberta
MODEL_NAME=roberta-large
MAX_BATCH_SIZE=24
WORD_TYPE=n
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

# echo $FILE_NAME
# echo $WORD_TYPE
python evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \
