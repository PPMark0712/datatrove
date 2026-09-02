import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.merge_split import FileSplitter
from datatrove.pipeline.readers import JsonlReader
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--output_file_count", type=int, default=None)
    parser.add_argument("--max_rows_per_file", type=int, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    result_path = os.path.join(main_output_path, "result")
    log_path = os.path.join(main_output_path, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            FileSplitter(
                output_folder=result_path,
                output_file_count=args.output_file_count,
                max_rows_per_file=args.max_rows_per_file,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=log_path,
    )
    executor.run()


if __name__ == "__main__":
    main()
