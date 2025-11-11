"""
Run ETA-DACP pipeline, select top-p documents with the highest part-of-speech entropy.
if args.tokenizer_path is None, sample top-p docs.
else, sample top docs with token-level budget.
"""
import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.eta_dacp import PosEntCalculator
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.samplers import HardSampler
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter, output_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--language", type=str, default="en", choices=["zh", "en"])
    parser.add_argument("--token_count_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--sample_rate", type=float, required=True)
    parser.add_argument("--sample_unit", type=str, choices=["doc", "token"], default="doc")
    parser.add_argument("--nltk_data_path", type=str, default=None)
    args = parser.parse_args()
    if args.sample_unit == "token":
        assert args.tokenizer_path or args.token_count_path
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    pos_ent_path = os.path.join(main_output_path, "part_of_speech_entropy")
    token_count_path = args.token_count_path or os.path.join(main_output_path, "token_count")
    result_path = os.path.join(main_output_path, "eta_dacp_result")
    log_path = os.path.join(main_output_path, "logs")

    pos_ent_calculate_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            *([TokensCounter(
                args.tokenizer_path,
                output_folder=token_count_path,
            )] if args.sample_unit == "token" and not args.token_count_path else []),
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
                sample_rate=args.sample_rate,
                unit=args.sample_unit,
                token_count_folder=token_count_path if args.sample_unit == "token" else None,
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
        logging_dir=os.path.join(log_path, "eta_dacp_result"),
    )
    sample_executor.run()


if __name__ == "__main__":
    main()
