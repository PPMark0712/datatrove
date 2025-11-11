import os

import nltk

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.freq_conc import FcdCalculator
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
    nltk_dependencies = {
        "corpora": [
            "wordnet",
            "stopwords"
        ],
        "tokenizers": [
            "punkt_tab"
        ],
        "taggers": [
            "averaged_perceptron_tagger_eng"
        ]
    }
    for path, packages in nltk_dependencies.items():
        for package in packages:
            try:
                nltk.data.find(f"{path}/{package}")
            except LookupError:
                nltk.download(package, download_dir=nltk_path)


def main():
    args = get_args()
    check_nltk_dependencies(args.nltk_path)

    MAIN_OUTPUT_PATH = args.output_path
    difficulty_path = os.path.join(MAIN_OUTPUT_PATH, "freq_conc_difficulty")
    LOG_PATH = os.path.join(MAIN_OUTPUT_PATH, "logs")

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            FcdCalculator(
                output_folder=difficulty_path,
                nltk_path=args.nltk_path,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH, "calc_freq_conc_difficulty"),
        skip_completed=not args.rerun,
    )
    executor.run()


if __name__ == "__main__":
    main()
