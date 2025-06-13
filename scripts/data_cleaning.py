import os
import json
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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data1/yyz/downloads/datasets/FinCorpus/processed")
    parser.add_argument("--glob_pattern", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="/data1/yyz/projects/data/datatrove_output/debug")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--tasks", type=int, default=64)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--languages", nargs="+", default=["zh"])
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

    MINHASH_OUTPUT_PATH = os.path.join(MAIN_OUTPUT_PATH, "3_minhash_deduplication")
    
    def above_language_threshold(doc, language_threshold=0.65):
        return doc.metadata["language_score"] >= language_threshold
    
    language_filter_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                args.input_path,
                glob_pattern=args.glob_pattern,
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
                    dup_n_grams=((5, 0.20), (6, 0.18), (7, 0.16), (8, 0.145), (9, 0.13), (10, 0.12)),
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
    

if __name__ == "__main__":
    main()

# DUMP_TO_PROCESS = "CC-MAIN-2023-50"  # example

# MAIN_OUTPUT_PATH = "s3://some_s3_bucket"
# FILTERING_OUTPUT_PATH = f"{MAIN_OUTPUT_PATH}/base_processing"

# main_processing_executor = LocalPipelineExecutor(
#     job_name=f"cc_{DUMP_TO_PROCESS}",
#     pipeline=[
#         WarcReader(
#             f"s3://commoncrawl/crawl-data/{DUMP_TO_PROCESS}/segments/",
#             glob_pattern="*/warc/*",  # we want the warc files
#             default_metadata={"dump": DUMP_TO_PROCESS},
#         ),
#         URLFilter(exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/1_url/{DUMP_TO_PROCESS}")),
#         Trafilatura(favour_precision=True),
#         LanguageFilter(
#             exclusion_writer=JsonlWriter(
#                 f"{FILTERING_OUTPUT_PATH}/2_non_english/",
#                 output_filename="${language}/" + DUMP_TO_PROCESS + "/${rank}.jsonl.gz",
#                 # folder structure: language/dump/file
#             )
#         ),
#         GopherRepetitionFilter(
#             exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/3_gopher_rep/{DUMP_TO_PROCESS}")
#         ),
#         GopherQualityFilter(
#             exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/4_gopher_qual/{DUMP_TO_PROCESS}")
#         ),
#         C4QualityFilter(
#             filter_no_terminal_punct=False,
#             exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/5_c4/{DUMP_TO_PROCESS}"),
#         ),
#         FineWebQualityFilter(
#             exclusion_writer=JsonlWriter(f"{FILTERING_OUTPUT_PATH}/removed/6_fineweb_qual/{DUMP_TO_PROCESS}")
#         ),
#         JsonlWriter(f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"),
#     ],
#     tasks=8000,
#     time="10:00:00",
#     logging_dir=f"{MAIN_OUTPUT_PATH}/logs/base_processing/{DUMP_TO_PROCESS}",
#     slurm_logs_folder=f"logs/base_processing/{DUMP_TO_PROCESS}/slurm_logs",  # must be local
#     randomize_start_duration=180,  # don't hit the bucket all at once with the list requests
#     mem_per_cpu_gb=2,
#     partition="hopper-cpu",
# )
# main_processing_executor.run()

# """
#     we then applied minhash deduplication to each individual dump,
# """

# # you can also change ngrams or the number of buckets and their size here
# minhash_config = MinhashConfig(
#     hash_config=HashConfig(
#         hash_fc="sha1",  # better precision -> fewer false positives (collisions)
#         precision=64,
#     ),
#     num_buckets=14,
#     hashes_per_bucket=8,
#     n_grams=5,
# )

# S3_MINHASH_BASE_PATH = f"{MAIN_OUTPUT_PATH}/minhash"

# S3_LOGS_FOLDER = f"{MAIN_OUTPUT_PATH}/logs/minhash"
# LOCAL_LOGS_FOLDER = "logs/minhash"

# TOTAL_TASKS = 1000

# # this is the original data that we want to deduplicate
# INPUT_READER = JsonlReader(
#     f"{FILTERING_OUTPUT_PATH}/output/{DUMP_TO_PROCESS}"
# )  # this is the output from the first part

# # stage 1 computes minhash signatures for each task (each task gets a set of files)
# stage1 = SlurmPipelineExecutor(
#     job_name=f"mh1_{DUMP_TO_PROCESS}",
#     pipeline=[
#         INPUT_READER,
#         MinhashDedupSignature(
#             output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures", config=minhash_config
#         ),
#     ],
#     tasks=TOTAL_TASKS,
#     time="5:00:00",
#     partition="hopper-cpu",
#     logging_dir=f"{S3_LOGS_FOLDER}/signatures",
#     slurm_logs_folder=f"{LOCAL_LOGS_FOLDER}/signatures/slurm_logs",
#     randomize_start_duration=180,
#     depends=main_processing_executor,  # only start after the first one completes
# )

# stage2 = SlurmPipelineExecutor(
#     job_name=f"mh2_{DUMP_TO_PROCESS}",
#     pipeline=[
#         MinhashDedupBuckets(
#             input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/signatures",
#             output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
#             config=MinhashConfig(hash_config=minhash_config.hash_config),
#         ),
#     ],
#     tasks=minhash_config.num_buckets * 50,  # the code supports parallelizing each bucket. here we run 50
#     # workers per bucket
#     randomize_start_duration=180,
#     logging_dir=f"{S3_LOGS_FOLDER}/buckets",
#     partition="hopper-cpu",
#     time="02:00:00",
#     mem_per_cpu_gb=4,
#     cpus_per_task=3,  # you can add run more (smaller) tasks if you do not have a lot of memory
#     depends=stage1,
# )


# stage3 = SlurmPipelineExecutor(
#     job_name=f"mh3_{DUMP_TO_PROCESS}",
#     pipeline=[
#         MinhashDedupCluster(
#             input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/buckets",
#             output_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids",
#             config=minhash_config,
#         ),
#     ],
#     tasks=1,  # this step runs on a single task
#     logging_dir=f"{S3_LOGS_FOLDER}/clustering",
#     partition="hopper-cpu",
#     time="30:00:00",  # and can also be quite slow. Usually not this slow though
#     mem_per_cpu_gb=25,
#     cpus_per_task=8,  # if you dedup a full dump, you do need a lot of memory for this one
#     depends=stage2,
# )


# stage4 = SlurmPipelineExecutor(
#     job_name=f"mh4_{DUMP_TO_PROCESS}",
#     pipeline=[
#         INPUT_READER,
#         TokensCounter(),  # you can remove this one, it's just a nice way to know how many tokens we have
#         # before and after dedup
#         MinhashDedupFilter(input_folder=f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/remove_ids"),
#         # run the PII removal
#         PIIFormatter(),
#         JsonlWriter(f"{S3_MINHASH_BASE_PATH}/{DUMP_TO_PROCESS}/deduped_output"),
#     ],
#     tasks=TOTAL_TASKS,
#     logging_dir=f"{S3_LOGS_FOLDER}/filtering",
#     partition="hopper-cpu",
#     time="5:00:00",
#     mem_per_cpu_gb=4,
#     depends=stage3,
# )

# # launch dedup pipelines
# stage4.run()
