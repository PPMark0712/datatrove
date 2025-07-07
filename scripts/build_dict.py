import argparse
import os
import itertools

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.cl import DictBuilder, DictMerger
from datatrove.pipeline.readers import JsonlReader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--ltp_model_path", type=str, default="/data1/yyz/downloads/models/LTP/small")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_path = os.path.join(args.output_path, "output")
    log_path = os.path.join(args.output_path, "logs")

    levels = os.listdir(args.input_path)
    levels.sort()
    for level in levels:
        if level == "university":
            continue
        leval_input_path = os.path.join(args.input_path, level)
        level_output_path = os.path.join(result_path, level)
        extracted_path = os.path.join(level_output_path, "extracted")
        merged_path = os.path.join(level_output_path, "merged")
        level_log_path = os.path.join(log_path, level)

        file_cnt = len(os.listdir(leval_input_path))
        extract_executor = LocalPipelineExecutor(
            pipeline=[
                JsonlReader(
                    leval_input_path,
                    # limit=1
                ),
                DictBuilder(
                    language=args.language,
                    output_folder=extracted_path,
                    ltp_model_path=args.ltp_model_path,
                )
            ],
            tasks=file_cnt,
            workers=file_cnt,
            skip_completed=not args.rerun,
            logging_dir=os.path.join(level_log_path, "extract"),
        )
        extract_executor.run()
        
        merge_executor = LocalPipelineExecutor(
            pipeline=[
                DictMerger(
                    input_folder=extracted_path,
                    output_folder=merged_path,
                )
            ],
            tasks=1,
            workers=1,
            skip_completed=not args.rerun,
            logging_dir=os.path.join(level_log_path, "merge"),
        )
        merge_executor.run()

    levels = os.listdir(result_path)
    level_words = {}
    for level in levels:
        with open(os.path.join(result_path, level, "merged", "merged_dict.txt"), "r") as f:
            words = f.read().splitlines()
        level_words[level] = set(words)
    
    levels = ["primary", "junior_high", "senior_high"]

    # 按顺序去除前面 level 已经出现过的词
    previous_words = set()
    for level in levels:
        words = level_words.get(level, set())
        level_words[level] = words - previous_words
        previous_words.update(words)
    
    final_path = os.path.join(args.output_path, "final")
    os.makedirs(final_path, exist_ok=True)
    for level in levels:
        words = level_words.get(level, set())
        words = list(words)
        words.sort()
        with open(os.path.join(final_path, f"{level}.txt"), "w") as f:
            for word in words:
                f.write(word + "\n")