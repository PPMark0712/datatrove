"""
CUDA_VISIBLE_DEVICES=4,5 python calc_ppl.py --input_path /mnt/data/kw/yyz/data/datatrove_output/MMedC_en_wordnet_1011/2_sorted_data --output_path /mnt/data/kw/yyz/data/datatrove_output/debug --limit 10 --model_path "/mnt/data/kw/models/Qwen/Qwen2.5-0.5B" --tasks 2 --workers 2
"""
import os

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.perplexity import Encoder, PerplexityCalculator
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.utils.common_argparser import get_common_argparser
from datatrove.utils.io_adapters import input_adapter, output_adapter


def get_args():
    parser = get_common_argparser()
    parser.add_argument("--encoder_workers", type=int, default=16)
    parser.add_argument("--ppl_workers", type=int, default=16)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()
    assert args.tensor_parallel_size * args.ppl_workers <= len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    return args


def main():
    args = get_args()
    main_output_path = args.output_path
    token_ids_path = os.path.join(main_output_path, "token_ids") 
    only_ppl_path = os.path.join(main_output_path, "only_ppl")
    data_with_ppl_path = os.path.join(main_output_path, "data_with_ppl")
    log_path = os.path.join(main_output_path, "logs")

    encode_executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            Encoder(
                output_folder=token_ids_path,
                model_path=args.model_path,
                max_model_len=args.max_model_len,
            )
        ],
        tasks=args.tasks,
        workers=args.encoder_workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "encode")
    )
    encode_executor.run()

    executor = LocalPipelineExecutor(
        pipeline=[
            JsonlReader(
                data_folder=args.input_path,
                glob_pattern=args.glob_pattern,
                adapter=input_adapter,
                limit=args.limit,
            ),
            PerplexityCalculator(
                token_ids_folder=token_ids_path,
                output_folder=only_ppl_path,
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size
            ),
            JsonlWriter(
                output_folder=data_with_ppl_path,
                adapter=output_adapter,
                compression=None
            ),
        ],
        tasks=args.tasks,
        workers=args.ppl_workers,
        skip_completed=not args.rerun,
        logging_dir=os.path.join(log_path, "ppl")
    )
    executor.run()


if __name__ == "__main__":
    main()
