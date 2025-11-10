# HF_ENDPOINT="https://hf-mirror.com" python download_fineweb.py --output_path ...
import os
import time
import argparse
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tasks', type=int, default=64)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=1000, help="line limit per task, -1 = download all")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用解析的参数配置路径和执行器
    base_output_path = args.output_path
    data_path = os.path.join(base_output_path, "data")
    log_path = os.path.join(base_output_path, "logs")

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                "hf://datasets/HuggingFaceFW/fineweb/data",
                limit=args.limit
            ),
            JsonlWriter(
                data_path,
                compression=None
            )
        ],
        tasks=args.tasks,
        workers=args.workers,
        logging_dir=log_path
    )
    
    # 执行数据处理流程
    while True:
        try:
            stats = pipeline_exec.run()
            if stats is None:
                break
        except Exception as e:
            print(e)
            time.sleep(60)
