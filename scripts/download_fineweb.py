# HF_ENDPOINT="https://hf-mirror.com" python download_fineweb.py
import os
import time
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import JsonlWriter

if __name__ == "__main__":
    base_output_path = "/data1/yyz/downloads/datasets/fineweb_2e7"
    data_path = os.path.join(base_output_path, "data")
    log_path = os.path.join(base_output_path, "logs")

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                "hf://datasets/HuggingFaceFW/fineweb/data",
                limit=300000
            ),
            JsonlWriter(
                data_path,
                compression=None
            )
        ],
        tasks=64,
        workers=8,
        logging_dir=log_path
    )
    while True:
        try:
            stats = pipeline_exec.run()
            if stats is None:
                break
        except Exception as e:
            print(e)
            time.sleep(60)
