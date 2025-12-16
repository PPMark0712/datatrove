import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.fineweb_edu import FinewebEduScoreCalculator
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter, output_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    score_path = os.path.join(main_output_path, "fineweb_edu_score") 
    log_path = os.path.join(main_output_path, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            FinewebEduScoreCalculator(
                output_folder=score_path,
                model_path=args.model_path,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "fineweb_edu")
    )
    executor.run()


if __name__ == "__main__":
    main()
