import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.readers import JsonlReader
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--tokenizer_path", type=str, required=True, help="path to tokenizer.json for counting tokens")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    token_count_path = os.path.join(main_output_path, "token_count")
    log_path = os.path.join(main_output_path, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit
            ),
            TokensCounter(
                args.tokenizer_path,
                output_folder=token_count_path
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "count_tokens")
    )
    executor.run()


if __name__ == "__main__":
    main()
