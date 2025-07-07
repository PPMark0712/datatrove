import os
import argparse
from functools import partial

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.cl.lexical_difficulty_calculator import LexicalDifficultyLabeler
from datatrove.pipeline.writers.jsonl import JsonlWriter

def wudao_adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "text": data.pop("title") + "\n" + data.pop("content"),
        "id": data.pop("id", f"{path}/{id_in_file}"),
        "media": data.pop("media", []),
        "metadata": data,
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--glob_pattern", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    MAIN_OUTPUT_PATH = args.output_path
    output_path = os.path.join(MAIN_OUTPUT_PATH, "output")
    LOG_PATH = os.path.join(MAIN_OUTPUT_PATH, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=wudao_adapter,
                limit=args.limit,
            ),
            LexicalDifficultyLabeler(
                dict_files={
                    "primary": "/data1/yyz/data/ChinaTextBook_processed/words/final/primary.txt",
                    "junior_high": "/data1/yyz/data/ChinaTextBook_processed/words/final/junior_high.txt",
                    "senior_high": "/data1/yyz/data/ChinaTextBook_processed/words/final/senior_high.txt",
                    # "stop_words": "/data1/yyz/data/ChinaTextBook_processed/words/final/stop_words.txt",
                },
            ),
            JsonlWriter(
                output_path,
                compression=None,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH),
        skip_completed=not args.rerun
    )
    executor.run()


if __name__ == "__main__":
    main()
