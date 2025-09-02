import os
import argparse
# import nltk
# nltk_path = "/data1/yyz/downloads/models/nltk_data"
# nltk.data.path.append(nltk_path)
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.cl_wordnet.lexical_difficulty_calculator import (
    HypernymDepthCalculator,
    PosHypernymDepthNormalizer,
    WeightSorter
)
from datatrove.pipeline.writers.jsonl import JsonlWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--glob_pattern", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    MAIN_OUTPUT_PATH = args.output_path
    hypernym_depth_path = os.path.join(MAIN_OUTPUT_PATH, "1_hypernym_depth")
    min_max_path = os.path.join(MAIN_OUTPUT_PATH, "2_min_max_data")
    normalized_hypernym_depth_path = os.path.join(MAIN_OUTPUT_PATH, "3_normalized_hypernym_depth")
    result_path = os.path.join(MAIN_OUTPUT_PATH, "4_result")
    LOG_PATH = os.path.join(MAIN_OUTPUT_PATH, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                limit=args.limit,
            ),
            HypernymDepthCalculator(
                output_folder=hypernym_depth_path,
                min_max_output_folder=min_max_path,
                nltk_path="/data1/yyz/downloads/models/nltk_data"
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH, "1_calculate_hypernym_depth"),
        skip_completed=not args.rerun
    )
    executor.run()

    normalizer_executor = LocalPipelineExecutor(
        pipeline=[
            PosHypernymDepthNormalizer(
                input_folder=hypernym_depth_path,
                min_max_folder=min_max_path,
                output_folder=normalized_hypernym_depth_path,
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH, "2_normalize_hypernym_depth"),
        skip_completed=not args.rerun
    )
    normalizer_executor.run()

    sorter_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                limit=args.limit,
            ),
            WeightSorter(
                normalized_hypernym_depth_folder=normalized_hypernym_depth_path,
            ),
            JsonlWriter(
                output_folder=result_path,
                compression=None
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH, "3_sort_by_lexical_difficulty"),
        skip_completed=not args.rerun
    )
    sorter_executor.run()


if __name__ == "__main__":
    main()
