import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.samplers import CdfSampler
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter, output_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--unit", type=str, default="doc", choices=["doc", "token"])
    parser.add_argument("--token_count_path", type=str, default=None)
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
            CdfSampler(
                score_folder=args.score_path,
                sample_rate=args.sample_rate,
                unit=args.unit,
                token_count_folder=args.token_count_path
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
