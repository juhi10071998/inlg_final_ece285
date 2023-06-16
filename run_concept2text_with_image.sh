TASK=concept2text
DATASET=commongen_inhouse_with_image_ofa
LOG_DIR=log/commongen

MODEL=bart-base
MODE=contraclipcap
PORTION=1
TRAIN_EPOCH=1
TRAIN_BATCH_SIZE=3
GRAD_ACC=1
WARM_UP=200
LR=2e-5  
MAX_INPUT_LENGTH=32
MAX_OUTPUT_LENGTH=64
RANDOM_SEEDS=41,42,43
TEST_SET=test

MAPPER_TYPE=transformer
PREFIX_LENGTH=20
RANDOM_INIT_MAPPER=0
MAPPER_PRETRAIN_DATASET=vist_sis-tunelm
TUNE_MAPPER=1
RANDOM_INIT_PROJECTION=0
TUNE_PROJECTION=1
LAMBDA_TEXT=0.
LAMBDA_CONTRA_LOSS=1.5
START_CONTRA_EPOCH=4

equal_batch_size=$(($TRAIN_BATCH_SIZE*$GRAD_ACC))
echo "batch_size = ${equal_batch_size}"

LABEL=${DATASET}_td${PORTION}_ep${TRAIN_EPOCH}_bs${equal_batch_size}_lr${LR}_lt${LAMBDA_TEXT}_lcontra${LAMBDA_CONTRA_LOSS}_cep${START_CONTRA_EPOCH}_warm${WARM_UP}_mapper-${MAPPER_TYPE}${PREFIX_LENGTH}_randinitmapper${RANDOM_INIT_MAPPER}_tunemapper${TUNE_MAPPER}_randinitproj${RANDOM_INIT_PROJECTION}_tuneproj${TUNE_PROJECTION}

CUDA_VISIBLE_DEVICES=0 python code/main.py \
--model_type $MODEL \
--mode $MODE \
--task $TASK \
--dataset $DATASET \
--train_ratio $PORTION \
--train_epoch ${TRAIN_EPOCH} \
--train_batch_size ${TRAIN_BATCH_SIZE} \
--gradient_accumulation_steps ${GRAD_ACC} \
--max_input_length ${MAX_INPUT_LENGTH} \
--max_output_length ${MAX_OUTPUT_LENGTH} \
--lr $LR \
--mapper_type ${MAPPER_TYPE} \
--visual_prefix_length ${PREFIX_LENGTH} \
--random_init_mapper ${RANDOM_INIT_MAPPER} \
--tune_clip_mapper ${TUNE_MAPPER} \
--mapper_pretrain_dataset ${MAPPER_PRETRAIN_DATASET} \
--random_init_projection ${RANDOM_INIT_PROJECTION} \
--tune_projection ${TUNE_PROJECTION} \
--warmup_steps ${WARM_UP} \
--lambda_text ${LAMBDA_TEXT} \
--lambda_contrastive_loss ${LAMBDA_CONTRA_LOSS} \
--start_contrastive_at_epoch ${START_CONTRA_EPOCH} \
--test_set ${TEST_SET} \
--random_seeds ${RANDOM_SEEDS} \
--label $LABEL \
--verbose 0 --best_imagination --single_imagination --multi_imagination|& tee ${LOG_DIR}/$MODEL-$MODE-$LABEL.out 
