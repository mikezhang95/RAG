
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

python train_generator.py \
    --seed 42 \
    --passages_per_question 5 \
    --passages_per_question_predict 5 \
    --max_answer_length 400 \
    --min_answer_length 10 \
    --learning_rate 1e-5 \
    --eval_step 10000 \
    --warmup_steps 10000 \
    --encoder_model_type hf_rag \
    --pretrained_model_cfg facebook/rag-token-nq \
    --do_lower_case \
    --dev_file data/dr_data/reader/dev.json \
    --train_file data/dr_data/reader/train.json \
    --sequence_length 512 \
    --num_train_epochs 10 \
    --batch_size 16 \
    --dev_batch_size 16 \
    --output_dir data/dr_exp/generator \
    --gradient_accumulation_steps 1 \
    > data/dr_exp/generator/train.log 
