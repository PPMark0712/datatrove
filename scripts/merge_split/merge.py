import argparse
import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.merge_split import FileMerger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--output_file_count", type=int, default=None)
    parser.add_argument("--rows_per_file", type=int, default=None)
    parser.add_argument("--input_glob_pattern", type=str, default="*.jsonl")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    result_path = os.path.join(main_output_path, "result")
    log_path = os.path.join(main_output_path, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            FileMerger(
                input_folder=args.input_path,
                output_folder=result_path,
                output_file_count=args.output_file_count,
                rows_per_file=args.rows_per_file,
                input_glob_pattern=args.input_glob_pattern,
            ),
        ],
        tasks=1,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=log_path,
    )
    executor.run()


if __name__ == "__main__":
    main()
