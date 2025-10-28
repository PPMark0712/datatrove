export CUDA_VISIBLE_DEVICES=4,5
model_path=/mnt/data/kw/models/Qwen/Qwen2.5-0.5B
model_path=/mnt/data/kw/models/NousResearch/Llama-3.2-1B
python calc_ppl.py \
    --input_path /mnt/data/kw/yyz/downloads/datasets/Henrychur/MMedC/English/processed \
    --output_path /mnt/data/kw/yyz/data/datatrove_output/MMedC_en_ppl_Llama-3.2-1B \
    --model_path $model_path \
    --tasks 32 \
    --encoder_workers 32 \
    --ppl_workers 2
