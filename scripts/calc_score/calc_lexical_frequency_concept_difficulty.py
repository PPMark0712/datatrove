import os

import nltk

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.cl_wordnet.lexical_difficulty_calculator import LexicalDifficultyCalculator
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--nltk_path", type=str, default=None)
    args = parser.parse_args()
    return args


def check_nltk_dependencies(nltk_path: str = None):
    if nltk_path:
        nltk.data.path.append(nltk_path)
    nltk_dependencies = [
        "wordnet",
        "stopwords",
        "punkt_tab",
        "averaged_perceptron_tagger_eng",
    ]
    for package in nltk_dependencies:
        nltk.download(package, download_dir=nltk_path)


def main():
    args = get_args()
    check_nltk_dependencies(args.nltk_path)

    MAIN_OUTPUT_PATH = args.output_path
    difficulty_path = os.path.join(MAIN_OUTPUT_PATH, "lexical_difficulty")
    LOG_PATH = os.path.join(MAIN_OUTPUT_PATH, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit
            ),
            LexicalDifficultyCalculator(
                output_folder=difficulty_path,
                nltk_path=args.nltk_path
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH, "calc_freq_conc"),
        skip_completed=not args.rerun
    )
    executor.run()


if __name__ == "__main__":
    main()
