import os
import json
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import random

def get_samples(remove_path):
    data = []
    for file in os.listdir(remove_path):
        with open(os.path.join(remove_path, file), "r") as f:
            for i, line in enumerate(f):
                item = json.loads(line)
                data.append(item["text"])
                if i >= 3:
                    break
    return data
                

if __name__ == "__main__":
    data_path = "/data1/yyz/data/datatrove_output/0615"
    output_path = "/data1/yyz/data/datatrove_output/0615/test_rules"

    quality_filter_paths = [
        "2_quality_filter/zh/removed/1_gopher_repetition_filter",
        "2_quality_filter/zh/removed/2_gopher_quality_filter",
        "2_quality_filter/zh/removed/4_fineweb_quality_filter",
    ]

    remove_paths = [
        "1_language_filter/removed/1_other_languages",
        "3_minhash_deduplication/zh/4_result/removed",
    ]
    for quality_filter_path in quality_filter_paths:
        for folder in os.listdir(os.path.join(data_path, quality_filter_path)):
            remove_paths.append(os.path.join(quality_filter_path, folder))

    remove_paths = [os.path.join(data_path, path) for path in remove_paths]

    for remove_path in remove_paths:
        print(remove_path)
        data = get_samples(remove_path)
        rel_path = os.path.relpath(remove_path, data_path)
        output_fn = os.path.join(output_path, rel_path) + ".txt"
        print(output_fn)
        os.makedirs(os.path.dirname(output_fn), exist_ok=True)
        with open(output_fn, "w") as f:
            for text in data:
                f.write(text + "\n")
                f.write("=" * 30 + "\n")
