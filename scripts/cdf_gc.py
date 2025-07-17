import os
import argparse

from datatrove.executor import LocalPipelineExecutor, MpPipelineExecutor
from datatrove.pipeline.cdf_gc import (
    DocumentPartOfSpeechPredictor,
    LexicalDiversityCalculator,
    DocumentDependencyParser,
    SyntacticComplexityCalculator,
    GcCombiner,
    ProbabilityCalculator,
    ProbabilitySampler,
)
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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--language", type=str, default="zh", choices=["zh"])
    parser.add_argument("--tokenizer_path", type=str, required=True, help="path to tokenizer.json for counting tokens")
    parser.add_argument("--ltp_model_path", type=str, default=None, help="path to huggingface model: LTP/small")
    parser.add_argument("--dependency_parsing_workers_per_gpu", type=int, default=1)
    parser.add_argument("--n_gpus", type=int, default=0)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--rate_for_hard_sample", type=float, default=0.4)
    args = parser.parse_args()
    assert args.n_gpus == len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    gc_path = os.path.join(main_output_path, "gc")
    part_of_speech_predicting_path = os.path.join(gc_path, "part_of_speech_predicting")
    lexical_diversity_path = os.path.join(gc_path, "lexical_diversity")
    dependency_parsing_path = os.path.join(gc_path, "dependency_parsing")
    syntactic_complexity_path = os.path.join(gc_path, "syntactic_complexity")
    gc_result_path = os.path.join(gc_path, "result")
    sampling_path = os.path.join(main_output_path, "sampling")
    probability_path = os.path.join(sampling_path, "probability")
    sample_result_path = os.path.join(sampling_path, "sample_result")

    log_path = os.path.join(main_output_path, "logs")

    denpendency_parsing_excecutor = MpPipelineExecutor(
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
                ltp_model_path=args.ltp_model_path
            ),
        ],
        tasks=args.tasks,
        workers=args.n_gpus * args.dependency_parsing_workers_per_gpu,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "gc", "dependency_parsing"),
    )
    denpendency_parsing_excecutor.run()
    
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

    information_merging_executor = LocalPipelineExecutor(
        pipeline=[
            LexicalDiversityCalculator(
                input_folder=part_of_speech_predicting_path,
                output_folder=lexical_diversity_path,
            ),
            SyntacticComplexityCalculator(
                input_folder=dependency_parsing_path,
                output_folder=syntactic_complexity_path,
            ),
            GcCombiner(
                lexical_diversity_folder=lexical_diversity_path,
                syntactic_complexity_folder=syntactic_complexity_path,
                output_folder=gc_result_path,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "gc", "gc_calculator"),
    )
    information_merging_executor.run()
    
    sample_probability_calculator_executor = LocalPipelineExecutor(
        pipeline=[
            ProbabilityCalculator(
                input_folder=gc_result_path,
                output_folder=probability_path,
                sample_token_rate=args.sample_rate,
                rate_for_hard_sample=args.rate_for_hard_sample
            ),
        ],
        tasks=1,
        workers=1,
        skip_completed=False,
        logging_dir=os.path.join(log_path, "sampling", "probability_calculator"),
    )
    sample_probability_calculator_executor.run()
    
    sample_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                limit=args.limit,
            ),
            ProbabilitySampler(
                prob_folder=probability_path
            ),
            JsonlWriter(
                output_folder=sample_result_path,
                compression=None
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=False,
        logging_dir=os.path.join(log_path, "sampling", "sample_result"),
    )
    sample_executor.run()


if __name__ == "__main__":
    main()
