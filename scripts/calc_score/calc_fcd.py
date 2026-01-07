import os

import nltk

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.fcd import FcdCalculator
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--nltk_path", type=str, default=None)
    parser.add_argument("--freq_scaling_factor", type=float, default=0.7)
    parser.add_argument("--w_f", type=float, default=0.5)
    parser.add_argument("--power_mean_alpha", type=float, default=1.5)
    parser.add_argument("--agg_top_quantile", type=float, default=0.9)
    parser.add_argument("--agg_top_weight", type=float, default=0.7)
    parser.add_argument("--noun_weight", type=float, default=0.7)
    args = parser.parse_args()
    return args


def check_nltk_dependencies(nltk_path: str = None):
    print("checking nltk dependencies")
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
                print(f"looking up {path}/{package}")
                nltk.data.find(f"{path}/{package}")
            except LookupError:
                print(f"nltk package {package} not found, downloading...")
                nltk.download(package, download_dir=nltk_path)
    print("all nltk dependencies are installed")


def main():
    args = get_args()
    # check_nltk_dependencies(args.nltk_path)

    MAIN_OUTPUT_PATH = args.output_path
    difficulty_path = os.path.join(MAIN_OUTPUT_PATH, "fcd_score")
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
                freq_scaling_factor=args.freq_scaling_factor,
                w_f=args.w_f,
                power_mean_alpha=args.power_mean_alpha,
                agg_top_quantile=args.agg_top_quantile,
                agg_top_weight=args.agg_top_weight,
                noun_weight=args.noun_weight,
            ),
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH, "calc_fcd"),
        skip_completed=not args.rerun,
    )
    executor.run()


if __name__ == "__main__":
    main()
