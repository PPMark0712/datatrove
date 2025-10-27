"""
Run ETA-DACP pipeline, select top-p documents with the highest part-of-speech entropy.
if args.tokenizer_path is None, sample top-p docs.
else, sample top docs with token-level budget.
"""
import os
import argparse
import dataclasses

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.eta_dacp import PosEntCalculator, HardSampler
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
    parser.add_argument("--language", type=str, default="en", choices=["zh", "en"])
    parser.add_argument("--tokenizer_path", type=str, default=None, help="path to tokenizer.json for counting tokens")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--nltk_data_path", type=str, default=None)
    args = parser.parse_args()
    return args


def input_adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "text": data.pop("text", ""),
        "id": data.pop("id", f"{path}/{id_in_file}"),
        "metadata": {
            **data.pop("metadata", {}),
            **data
        },  # remaining data goes into metadata
    }


def output_adapter(self, document: Document) -> dict:
    data = {key: val for key, val in dataclasses.asdict(document).items() if val}
    return data


def main():
    args = get_args()
    main_output_path = args.output_path
    pos_ent_path = os.path.join(main_output_path, "part_of_speech_entropy")
    result_path = os.path.join(main_output_path, "sample_result")
    log_path = os.path.join(main_output_path, "logs")

    pos_ent_calculate_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            *([TokensCounter(args.tokenizer_path)] if args.tokenizer_path is not None else []),
            PosEntCalculator(
                language=args.language,
                output_folder=pos_ent_path,
                nltk_data_path=args.nltk_data_path,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "part_of_speech_entropy_calculator"),
    )
    pos_ent_calculate_executor.run()

    sample_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            HardSampler(
                score_folder=pos_ent_path,
                top_p=args.sample_rate,
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
        logging_dir=os.path.join(log_path, "result"),
    )
    sample_executor.run()


if __name__ == "__main__":
    main()
