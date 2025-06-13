import os
import json
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random

def write_samples(data_by_buckets, key, output_fn):
    """
    将已经分桶的数据写入文件
    data_by_buckets: Dict[float, List[str]] 桶索引到文本列表的映射
    """
    if not data_by_buckets:
        return
        
    # 找到数据范围
    min_val = min(data_by_buckets.keys())
    max_val = max(data_by_buckets.keys())
    
    # 写入文件
    with open(output_fn, 'w') as f:
        f.write(f"Samples for {key}\n")
        f.write("=" * 50 + "\n\n")
        
        for bucket_start in np.arange(min_val, max_val + 0.1, 0.1):
            bucket_end = bucket_start + 0.1
            f.write(f"\nBucket [{bucket_start:.1f}, {bucket_end:.1f})\n")
            f.write("-" * 30 + "\n")
            
            if bucket_start in data_by_buckets:
                samples = data_by_buckets[bucket_start]
                for i, text in enumerate(samples, 1):
                    f.write(f"\nSample {i}:\n{text}\n")
                    f.write("=" * 30 + "\n")
            else:
                f.write("No samples in this bucket\n")
                f.write("=" * 30 + "\n")

def read_file(merged_args):
    data_path, file_path, keys = merged_args
    # 为每个key维护一个按桶划分的数据字典
    data_by_buckets = {key: defaultdict(list) for key in keys}
    
    with open(os.path.join(data_path, file_path), "r") as f:
        for line in f:
            item = json.loads(line)
            text = item.get("text", "")
            for key in keys:
                try:
                    value = item['metadata']['repetition'][key]
                    # 计算桶索引
                    bucket_idx = np.floor(value * 10) / 10
                    # 如果该桶的样本数还不到10个，则添加
                    if len(data_by_buckets[key][bucket_idx]) < 10:
                        data_by_buckets[key][bucket_idx].append(text)
                except KeyError:
                    continue
    return data_by_buckets

if __name__ == "__main__":
    data_path = "/data1/yyz/projects/data/datatrove_output/rep_test/2_quality_filter/zh/output"
    output_path = "/data1/yyz/projects/data/datatrove_output/rep_test/samples"
    keys = ["dup_line_frac"] + [f"top_{n}_gram" for n in range(2, 5)] + [f"duplicated_{n}_n_grams" for n in range(5, 11)]
    
    file_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith(".jsonl"):
                file_list.append(os.path.join(root, file))
    
    merged_args_list = [(data_path, file, keys) for file in file_list]
    # 为每个key维护一个按桶划分的数据字典
    result_by_buckets = {key: defaultdict(list) for key in keys}
    
    with mp.Pool(processes=32) as pool:
        for file_data_by_buckets in pool.imap_unordered(read_file, merged_args_list):
            # 合并各个文件的结果，但每个桶最多保留10个样本
            for key in keys:
                for bucket_idx, texts in file_data_by_buckets[key].items():
                    current_bucket = result_by_buckets[key][bucket_idx]
                    remaining_space = 10 - len(current_bucket)
                    if remaining_space > 0:
                        # 随机选择要添加的样本数量
                        n_samples = min(remaining_space, len(texts))
                        if n_samples < len(texts):
                            texts = random.sample(texts, n_samples)
                        current_bucket.extend(texts)
    
    os.makedirs(output_path, exist_ok=True)
    
    top_n_gram_keys = [f"top_{n}_gram" for n in range(2, 5)]
    dup_n_gram_keys = [f"duplicated_{n}_n_grams" for n in range(5, 11)]
    
    # 处理top_n_gram数据
    for key in top_n_gram_keys:
        output_fn = os.path.join(output_path, f"{key}.txt")
        write_samples(result_by_buckets[key], key, output_fn)
    
    # 处理dup_n_gram数据
    for key in dup_n_gram_keys:
        output_fn = os.path.join(output_path, f"{key}.txt")
        write_samples(result_by_buckets[key], key, output_fn)

    # 处理dup_line_frac
    output_fn = os.path.join(output_path, "dup_line_frac.txt")
    write_samples(result_by_buckets["dup_line_frac"], "dup_line_frac", output_fn)
