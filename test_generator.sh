
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

python train_generator.py \
    --prediction_results_file ./data/dr_data/reader/dev_results.json.rag \
    --dev_file ./data/dr_data/reader/dev.json \
    --passages_per_question_predict 2 \
    --model_file ./data/dr_exp/generator/rag_generator.7.211 \
    --dev_batch_size 32 \
    --sequence_length 300 \
    --max_answer_length 200 \
    --min_answer_length 10 \
    # --test_only 
