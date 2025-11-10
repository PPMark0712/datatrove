import os
import argparse
import dataclasses

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.samplers import RandomSampler, DocumentCounter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.data import Document
from datatrove.utils.common_argparser import get_common_argparser


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--unit", type=str, default="document", choices=["document", "token"])
    parser.add_argument("--token_count_path", type=str, default=None)
    args = parser.parse_args()
    return args


def input_adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "text": data.pop("text", ""),
        "id": data.pop("id", f"{path}/{id_in_file}"),
        "metadata": {
            **data.pop("metadata", {}),
            **data
        },
    }


def output_adapter(self, document: Document) -> dict:
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    return data


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
            RandomSampler(
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
