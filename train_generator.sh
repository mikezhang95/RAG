
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch \
    --nproc_per_node=2 train_generator.py \
    --seed 42 \
    --passages_per_question 2 \
    --passages_per_question_predict 2 \
    --max_answer_length 200 \
    --min_answer_length 10 \
    --learning_rate 1e-5 \
    --eval_step 10000 \
    --warmup_steps 10000 \
    --encoder_model_type hf_rag \
    --pretrained_model_cfg congcongwang/bart-base-en-zh \
    --do_lower_case \
    --dev_file data/dr_data/reader/dev.json \
    --train_file data/dr_data/reader/train.json \
    --sequence_length 300 \
    --num_train_epochs 10 \
    --batch_size 8 \
    --dev_batch_size 8 \
    --output_dir data/dr_exp/generator \
    --gradient_accumulation_steps 1 \
    # > data/dr_exp/generator/train.log 
