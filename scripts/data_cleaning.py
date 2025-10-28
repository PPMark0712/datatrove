import os
import argparse
from functools import partial

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.dedup import MinhashDedupCluster, MinhashDedupFilter, MinhashDedupSignature
from datatrove.pipeline.dedup.minhash import MinhashConfig, MinhashDedupBuckets
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    LambdaFilter
)
from datatrove.pipeline.formatters import PIIFormatter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import TokensCounter
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.hashing import HashConfig
from datatrove.data import Document


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data1/yyz/downloads/datasets/FinCorpus/processed")
    parser.add_argument("--glob_pattern", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="/data1/yyz/projects/data/datatrove_output/debug")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--languages", nargs="+", default=["zh"])
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    MAIN_OUTPUT_PATH = args.output_path
    LOG_PATH = os.path.join(MAIN_OUTPUT_PATH, "logs")
    LANGUAGE_FILTER_PATH = os.path.join(MAIN_OUTPUT_PATH, "1_language_filter")
    LANGUAGE_FILTER_REMOVE_PATH = os.path.join(LANGUAGE_FILTER_PATH, "removed")
    LANGUAGE_FILTER_OUTPUT_PATH = os.path.join(LANGUAGE_FILTER_PATH, "output")

    QUALITY_FILTERING_PATH = os.path.join(MAIN_OUTPUT_PATH, "2_quality_filter")

    MINHASH_PATH = os.path.join(MAIN_OUTPUT_PATH, "3_minhash_deduplication")
    
    """
    ==============================
    1. language filter
    ==============================
    """
    def above_language_threshold(doc, language_threshold=0.65):
        return doc.metadata["language_score"] >= language_threshold
    
    language_filter_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                args.input_path,
                glob_pattern=args.glob_pattern,
                limit=args.limit,
            ),
            LanguageFilter(
                languages=args.languages,
                exclusion_writer=JsonlWriter(
                    os.path.join(LANGUAGE_FILTER_REMOVE_PATH, "1_other_languages"),
                    compression=None
                )
            ),
            LambdaFilter(
                filter_function=partial(above_language_threshold, language_threshold=0.65),
                exclusion_writer=JsonlWriter(
                    os.path.join(LANGUAGE_FILTER_REMOVE_PATH, "2_below_language_score_threshold"),
                    compression=None
                )
            ),
            JsonlWriter(
                LANGUAGE_FILTER_OUTPUT_PATH,
                output_filename="${language}/${rank}.jsonl",
                compression=None,
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=os.path.join(LOG_PATH, "1_language_filter"),
        skip_completed=not args.rerun
    )
    language_filter_executor.run()

    """
    ==============================
    2. quality filter
    ==============================
    """

    language_folders = os.listdir(LANGUAGE_FILTER_OUTPUT_PATH)
    for language in language_folders:
        QUALITY_FILTERING_REMOVE_PATH = os.path.join(QUALITY_FILTERING_PATH, language, "removed")
        QUALITY_FILTERING_OUTPUT_PATH = os.path.join(QUALITY_FILTERING_PATH, language, "output")
        quality_filter_executor = LocalPipelineExecutor(
            pipeline=[         
                JsonlReader(
                    os.path.join(LANGUAGE_FILTER_OUTPUT_PATH, language)
                ),
                GopherRepetitionFilter(
                    language=language,
                    # top_n_grams=((2, 0.083799), (3, 0.092784), (4, 0.106796)),
                    dup_n_grams=((5, 0.25), (6, 0.23), (7, 0.20), (8, 0.19), (9, 0.18), (10, 0.17)),
                    exclusion_writer=JsonlWriter(
                        os.path.join(QUALITY_FILTERING_REMOVE_PATH, "1_gopher_repetition_filter"),
                        output_filename="${filter_reason}/${rank}.jsonl",
                        compression=None
                    )
                ),
                GopherQualityFilter(
                    language=language,
                    min_stop_words=None,
                    max_non_alpha_words_ratio=None,
                    min_avg_word_length=None,
                    max_avg_word_length=None,
                    exclusion_writer=JsonlWriter(
                        os.path.join(QUALITY_FILTERING_REMOVE_PATH, "2_gopher_quality_filter"),
                        output_filename="${filter_reason}/${rank}.jsonl",
                        compression=None
                    )
                ),
                FineWebQualityFilter(
                    language=language,
                    exclusion_writer=JsonlWriter(
                        os.path.join(QUALITY_FILTERING_REMOVE_PATH, "4_fineweb_quality_filter"),
                        output_filename="${filter_reason}/${rank}.jsonl",
                        compression=None
                    )
                ),
                JsonlWriter(
                    QUALITY_FILTERING_OUTPUT_PATH,
                    compression=None
                )
            ],
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=os.path.join(LOG_PATH, "2_quality_filter"),
            skip_completed=not args.rerun
        )
        quality_filter_executor.run()
    
        """
        ==============================
        3. minhash deduplication
        ==============================
        """
        MINHASH_SIGNATURE_PATH = os.path.join(MINHASH_PATH, language, "1_signatures")
        MINHASH_BUCKETS_PATH = os.path.join(MINHASH_PATH, language, "2_buckets")
        MINHASH_REMOVE_IDS_PATH = os.path.join(MINHASH_PATH, language, "3_remove_ids")
        MINHASH_RESULT_PATH = os.path.join(MINHASH_PATH, language, "4_result")

        minhash_config = MinhashConfig(
            hash_config=HashConfig(
                hash_fc="sha1",  # better precision -> fewer false positives (collisions)
                precision=64,
            ),
            num_buckets=14,
            hashes_per_bucket=8,
            n_grams=5,
        )

        MINHASH_INPUT_READER = JsonlReader(
            QUALITY_FILTERING_OUTPUT_PATH
        )

        minhash_signature_executor = LocalPipelineExecutor(
            pipeline=[
                MINHASH_INPUT_READER,
                MinhashDedupSignature(
                    language=language,
                    output_folder=MINHASH_SIGNATURE_PATH,
                    config=minhash_config
                ),
            ],
            tasks=args.tasks,
            workers=args.workers,
            logging_dir=os.path.join(LOG_PATH, "3_minhash_deduplication", language, "1_signatures"),
            skip_completed=not args.rerun
        )

        minhash_buckets_executor = LocalPipelineExecutor(
            pipeline=[
                MinhashDedupBuckets(
                    input_folder=MINHASH_SIGNATURE_PATH,
                    output_folder=MINHASH_BUCKETS_PATH,
                    config=minhash_config
                ),
            ],
            tasks=minhash_config.num_buckets,
            workers=args.workers,
            logging_dir=os.path.join(LOG_PATH, "3_minhash_deduplication", language, "2_buckets"),
            skip_completed=not args.rerun,
            depends=minhash_signature_executor,
        )

        minhash_cluster_executor = LocalPipelineExecutor(
            pipeline=[
                MinhashDedupCluster(
                    input_folder=MINHASH_BUCKETS_PATH,
                    output_folder=MINHASH_REMOVE_IDS_PATH,
                    config=minhash_config,
                ),
            ],
            tasks=1,
            logging_dir=os.path.join(LOG_PATH, "3_minhash_deduplication", language, "3_cluster"),
            skip_completed=not args.rerun,
            depends=minhash_buckets_executor,
        )

        minhash_filter_executor = LocalPipelineExecutor(
            pipeline=[
                MINHASH_INPUT_READER,
                TokensCounter("/data1/yyz/downloads/models/NousResearch/Llama-3.2-1B/tokenizer.json"),
                MinhashDedupFilter(
                    input_folder=MINHASH_REMOVE_IDS_PATH,
                    exclusion_writer=JsonlWriter(
                        os.path.join(MINHASH_RESULT_PATH, "removed"),
                        compression=None
                    )
                ),
                PIIFormatter(),
                JsonlWriter(
                    os.path.join(MINHASH_RESULT_PATH, "output"),
                    compression=None
                )
            ],
            tasks=args.tasks,
            logging_dir=os.path.join(LOG_PATH, "3_minhash_deduplication", language, "4_filter"),
            skip_completed=not args.rerun,
            depends=minhash_cluster_executor,
        )
        minhash_filter_executor.run()


if __name__ == "__main__":
    main()

