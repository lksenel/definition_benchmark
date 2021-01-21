
for MAX_SIM in 0.5 0.6 0.7 0.8 0.9 1
do
    OUTPUT_FOLDER=/mounts/work/kerem/data/definition_benchmark_data/W2D/W2D_on_WDLAMPro_easy/max_sim_$MAX_SIM
    DATA_FILE=/mounts/work/kerem/data/definition_benchmark_data/WDNLAMPro_max_sim_$MAX_SIM.json
    GPU_ID=5

    MODEL_CLS=roberta
    MODEL_NAME=roberta-large

    # EMB_FILE=/mounts/work/kerem/embedding_algorithms/bert_ota/embeddings-wiki-100-diff-bbu-all.txt

    MAX_DEF_LEN=48
    MAX_BATCH_SIZE=64

    WORD_TYPE=n
    python -W ignore evaluate.py \
    --model_cls $MODEL_CLS \
    --model_name $MODEL_NAME \
    --output_folder $OUTPUT_FOLDER \
    --data_file $DATA_FILE \
    --word_type $WORD_TYPE \
    --max_def_len $MAX_DEF_LEN \
    --max_batch_size $MAX_BATCH_SIZE \
    --gpu_id $GPU_ID \
    # --emb_file $EMB_FILE

    WORD_TYPE=v
    python -W ignore evaluate.py \
    --model_cls $MODEL_CLS \
    --model_name $MODEL_NAME \
    --output_folder $OUTPUT_FOLDER \
    --data_file $DATA_FILE \
    --word_type $WORD_TYPE \
    --max_def_len $MAX_DEF_LEN \
    --max_batch_size $MAX_BATCH_SIZE \
    --gpu_id $GPU_ID \
    # --emb_file $EMB_FILE
done