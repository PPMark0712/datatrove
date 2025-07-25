export CUDA_VISIBLE_DEVICES=4,5

python cdf_gc.py \
    --input_path /data/yingyizhou/data/fincorpus_cleaned/output \
    --output_path /data/yingyizhou/data/datatrove_output/cdf_gc_new \
    --tasks 64 \
    --workers 32 \
    --ltp_model_path /data/yingyizhou/downloads/models/LTP/small \
    --tokenizer_path /data/downloads/models/NousResearch/Llama-3.2-1B/tokenizer.json \
    --dependency_parsing_workers_per_gpu 4 \
    --limit 10 \
    --sample_rate 0.2