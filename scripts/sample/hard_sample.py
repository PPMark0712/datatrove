import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.eta_dacp import HardSampler
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter, output_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--score_path", type=str, required=True)
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
                limit=args.limit
            ),
            HardSampler(
                score_folder=args.score_path,
                top_p=1/3
            ),
            JsonlWriter(
                output_folder=result_path,
                adapter=output_adapter,
                compression=None
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=log_path
    )
    executor.run()


if __name__ == "__main__":
    main()
