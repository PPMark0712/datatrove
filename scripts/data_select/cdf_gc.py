import os
import argparse
import dataclasses

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.cdf_gc import (
    DocumentPartOfSpeechPredictor,
    LexicalDiversityCalculator,
    DocumentDependencyParser,
    SyntacticComplexityCalculator,
    GcCombiner,
    GcNormalizer,
    ProbabilityCalculator,
    ProbabilitySampler,
)
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.data import Document
from datatrove.utils.common_argparser import get_common_argparser


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--language", type=str, default="zh", choices=["zh"])
    parser.add_argument("--tokenizer_path", type=str, required=True, help="path to tokenizer.json for counting tokens")
    parser.add_argument("--ltp_model_path", type=str, default=None, help="path to huggingface model: LTP/small")
    parser.add_argument("--workers_per_gpu", type=int, default=1)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--rate_for_hard_sample", type=float, default=0.4)
    args = parser.parse_args()
    args.n_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(","))
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
    gc_path = os.path.join(main_output_path, "1_gc_data")
    dependency_parsing_path = os.path.join(gc_path, "1_dependency_parsing")
    part_of_speech_predicting_path = os.path.join(gc_path, "2_part_of_speech_predicting")
    lexical_diversity_path = os.path.join(gc_path, "3_lexical_diversity")
    syntactic_complexity_path = os.path.join(gc_path, "4_syntactic_complexity")
    gc_result_path = os.path.join(gc_path, "5_combined_gc")
    normalized_gc_path = os.path.join(gc_path, "6_normalized_gc")
    sampling_path = os.path.join(main_output_path, "2_sampling")
    probability_path = os.path.join(sampling_path, "1_probability")
    sample_result_path = os.path.join(sampling_path, "2_sample_result")

    log_path = os.path.join(main_output_path, "logs")

    denpendency_parsing_excecutor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            DocumentDependencyParser(
                language=args.language,
                output_folder=dependency_parsing_path,
                n_gpus=args.n_gpus,
                workers_per_gpu=args.workers_per_gpu,
                ltp_model_path=args.ltp_model_path
            ),
        ],
        tasks=args.tasks,
        workers=args.n_gpus * args.workers_per_gpu,
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
                adapter=input_adapter
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
            GcNormalizer(
                input_folder=gc_result_path,
                output_folder=normalized_gc_path,
            ),
            ProbabilityCalculator(
                input_folder=normalized_gc_path,
                output_folder=probability_path,
                sample_token_rate=args.sample_rate,
                rate_for_hard_sample=args.rate_for_hard_sample
            ),
        ],
        tasks=1,
        workers=1,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "sampling", "probability_calculator"),
    )
    sample_probability_calculator_executor.run()
    
    sample_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            ProbabilitySampler(
                prob_folder=probability_path,
                gc_folder=normalized_gc_path
            ),
            JsonlWriter(
                output_folder=sample_result_path,
                adapter=output_adapter,
                compression=None
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "sampling", "sample_result"),
    )
    sample_executor.run()


if __name__ == "__main__":
    main()
