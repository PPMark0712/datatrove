import os
import argparse
import dataclasses

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.perplexity import PerplexityCalculator, PPLModel
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.data import Document


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--glob_pattern", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    assert args.tensor_parallel_size * args.workers <= len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
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
    only_ppl_path = os.path.join(main_output_path, "only_ppl")
    data_with_ppl_path = os.path.join(main_output_path, "data_with_ppl")
    log_path = os.path.join(main_output_path, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            PerplexityCalculator(
                output_folder=only_ppl_path,
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size
            ),
            JsonlWriter(
                output_folder=data_with_ppl_path,
                adapter=output_adapter,
                compression=None
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
