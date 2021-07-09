DATA_FOLDER=/mounts/work/kerem/Projects/CoDA/dataset/
MAX_DEF_LEN=48
NONCE_WORD=bkatuhla



# -------------------------------------- DATASET 1 nouns -------------------------------------s
FILE_NAME=CoDA-clean-hard.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=n
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \

# -------------------------------------- DATASET 1 verbs -------------------------------------s
FILE_NAME=CoDA-clean-hard.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=v
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \


# -------------------------------------- DATASET 2 nouns -------------------------------------s
FILE_NAME=CoDA-clean-easy.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=n
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \

# -------------------------------------- DATASET 2 verbs -------------------------------------s
FILE_NAME=CoDA-clean-easy.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=v
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \


# -------------------------------------- DATASET 3 nouns -------------------------------------s
FILE_NAME=CoDA-noisy-hard.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=n
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \

# -------------------------------------- DATASET 3 verbs -------------------------------------s
FILE_NAME=CoDA-noisy-hard.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=v
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \


# -------------------------------------- DATASET 4 nouns -------------------------------------s
FILE_NAME=CoDA-noisy-easy.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=n
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \

# -------------------------------------- DATASET 4 verbs -------------------------------------s
FILE_NAME=CoDA-noisy-easy.json
DATA_FILE=$DATA_FOLDER$FILE_NAME

GPU_ID=4

MODEL_CLS=bert
MODEL_NAME=bert-base-uncased
MAX_BATCH_SIZE=81
WORD_TYPE=v
OUTPUT_FOLDER=/mounts/work/kerem/Projects/CoDA/evaluation_results/$MODEL_NAME

echo $FILE_NAME
echo $WORD_TYPE
python -W ignore evaluate.py \
--output_folder $OUTPUT_FOLDER \
--data_file $DATA_FILE \
--model_cls $MODEL_CLS \
--model_name $MODEL_NAME \
--word_type $WORD_TYPE \
--nonce_word $NONCE_WORD \
--max_def_len $MAX_DEF_LEN \
--max_batch_size $MAX_BATCH_SIZE \
--gpu_id $GPU_ID \