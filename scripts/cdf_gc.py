import os
import argparse
import multiprocessing
import torch

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.cdf_gc import (
    DocumentPartOfSpeechPredictor,
    LexicalDiversityCalculator,
    DocumentDependencyParser,
    SyntacticComplexityCalculator
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--glob_pattern", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--language", type=str, default="zh")
    parser.add_argument("--tokenizer_path", type=str, default="/data1/yyz/downloads/models/NousResearch/Llama-3.2-1B/tokenizer.json")
    parser.add_argument("--only_dependency_parsing", action="store_true")
    parser.add_argument("--dependency_parsing_workers_per_gpu", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    gc_path = os.path.join(main_output_path, "gc")
    part_of_speech_predicting_path = os.path.join(gc_path, "part_of_speech_predicting")
    lexical_diversity_path = os.path.join(gc_path, "lexical_diversity")
    dependency_parsing_path = os.path.join(gc_path, "dependency_parsing")
    syntactic_complexity_path = os.path.join(gc_path, "syntactic_complexity")
    log_path = os.path.join(main_output_path, "logs")

    denpendency_parsing_excecutor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                limit=args.limit,
            ),
            DocumentDependencyParser(
                language=args.language,
                output_folder=dependency_parsing_path,
                n_gpus=args.n_gpus,
                workers_per_gpu=args.dependency_parsing_workers_per_gpu,
                ltp_model_path="/data1/yyz/downloads/models/LTP/small"
            ),
        ],
        tasks=args.tasks,
        workers=args.n_gpus * args.dependency_parsing_workers_per_gpu,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "gc", "dependency_parsing"),
    )
    denpendency_parsing_excecutor.run()
    if args.only_dependency_parsing:
        return
    
    part_of_speech_predicting_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                limit=args.limit,
            ),
            TokensCounter(args.tokenizer_path),
            DocumentPartOfSpeechPredictor(
                language=args.language,
                output_folder=part_of_speech_predicting_path,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "gc", "part_of_speech_predicting"),
    )
    part_of_speech_predicting_executor.run()

    # information_merging_executor = LocalPipelineExecutor(
    #     pipeline=[
    #         LexicalDiversityCalculator(
    #             input_folder=part_of_speech_predicting_path,
    #             output_folder=lexical_diversity_path,
    #         ),
    #         SyntacticComplexityCalculator(
    #             input_folder=dependency_parsing_path,
    #             output_folder=syntactic_complexity_path,
    #         ),
    #         GCCombiner(
    #             lexical_diversity_folder=lexical_diversity_path,
    #             syntactic_complexity_folder=syntactic_complexity_path,
    #             output_folder=gc_path,
    #         ),
    #     ],
    #     tasks=args.tasks,
    #     workers=args.workers,
    #     skip_completed=not args.rerun,
    #     logging_dir=os.path.join(log_path, "gc", "gc_calculator"),
    # )
    # information_merging_executor.run()
    

if __name__ == "__main__":
    main()

