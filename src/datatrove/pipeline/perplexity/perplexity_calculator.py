import os
import json
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.logging import logger
from .ppl_model import PPLModel


class PerplexityCalculator(PipelineStep):
    name = "Perplexity Calculator"
    type = "Perplexity"

    def __init__(
        self,
        token_ids_folder: DataFolderLike,
        output_folder: DataFolderLike,
        model_path: str,
        tensor_parallel_size: int = 1
    ):
        super().__init__()
        self.token_ids_folder = get_datafolder(token_ids_folder)
        self.output_folder = get_datafolder(output_folder)
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.visible_gpus = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            gpu_start_idx = self.tensor_parallel_size * self._local_rank
            gpu_end_idx = self.tensor_parallel_size * (self._local_rank + 1)
            use_gpus = self.visible_gpus[gpu_start_idx: gpu_end_idx]
            logger.info(f"process {self._local_rank} using GPUs {use_gpus} for Perplexity Calculator")

            inputs = []
            with self.token_ids_folder.open(f"{rank:05d}.jsonl", mode="r") as f:
                for line in f:
                    inputs.append({"prompt_token_ids": json.loads(line)})
            try:
                ppl_model = PPLModel(
                    self.model_path,
                    self.tensor_parallel_size,
                    use_gpu_ids=use_gpus
                )
                ppl_data = ppl_model.calc_ppl(inputs)
                del ppl_model
            except Exception as e:
                logger.error(e)
                del ppl_model  # ensure GPU is released
                raise e
            with self.output_folder.open(f"{rank:05d}.json", mode="w") as f:
                json.dump(ppl_data, f)
            for doc, ppl in zip(data, ppl_data):
                doc.metadata["perplexity"] = ppl
                yield doc