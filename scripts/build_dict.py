import argparse
import os

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
    log_path = os.path.join(args.output_path, "logs")
    levels = os.listdir(args.input_path)
    levels.sort()
    for level in levels:
        if level == "university":
            continue
        leval_input_path = os.path.join(args.input_path, level)
        level_output_path = os.path.join(args.output_path, level)
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