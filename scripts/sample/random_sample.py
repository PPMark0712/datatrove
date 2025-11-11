import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.samplers import RandomSampler, DocumentCounter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter, output_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--unit", type=str, default="doc", choices=["doc", "token"])
    parser.add_argument("--token_count_path", type=str, default=None)
    args = parser.parse_args()
    assert 0 <= args.sample_rate <= 1, "sample_rate must be between 0 and 1"
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    count_path = os.path.join(main_output_path, "count")
    result_path = os.path.join(main_output_path, "result")
    log_path = os.path.join(main_output_path, "logs")

    counter_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit
            ),
            DocumentCounter(
                output_folder=count_path
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "counter")
    )
    counter_executor.run()

    sample_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit
            ),
            RandomSampler(
                sample_rate=args.sample_rate,
                count_folder=count_path,
                unit=args.unit,
                token_count_folder=args.token_count_path
            ),
            JsonlWriter(
                output_folder=result_path,
                adapter=output_adapter,
                compression=None
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=log_path
    )
    sample_executor.run()


if __name__ == "__main__":
    main()
