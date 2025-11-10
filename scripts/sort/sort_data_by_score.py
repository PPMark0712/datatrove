import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.sorter import DocumentSplitSorter, DocumentSplitMerger
from datatrove.pipeline.readers import JsonlReader
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    score_path = args.score_path
    split_path = os.path.join(main_output_path, "split_cache")
    result_path = os.path.join(main_output_path, "result")
    result_score_path = os.path.join(main_output_path, "result_scores")
    log_path = os.path.join(main_output_path, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit
            ),
            DocumentSplitSorter(
                score_folder=score_path,
                split_folder=split_path,
                reverse=args.reverse,
                batch_size_mb=1024
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "split")
    )
    executor.run()

    merge_executor = LocalPipelineExecutor(
        pipeline=[
            DocumentSplitMerger(
                split_folder=split_path,
                output_folder=result_path,
                output_score_folder=result_score_path,
                reverse=args.reverse
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "merge")
    )
    merge_executor.run()



if __name__ == "__main__":
    main()
