# HF_ENDPOINT="https://hf-mirror.com" python download_fineweb.py
import os
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    base_output_path = "/data1/yyz/downloads/datasets/fineweb"
    data_path = os.path.join(base_output_path, "data")
    log_path = os.path.join(base_output_path, "logs")

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                "hf://datasets/HuggingFaceFW/fineweb/data",
                limit=100000
            ),
            JsonlWriter(
                data_path,
                compression=None
            )
        ],
        tasks=32,
        workers=4,
        logging_dir=log_path
    )
    pipeline_exec.run()
