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
        output_folder: DataFolderLike,
        model_path: str,
        tensor_parallel_size: int = 1
    ):
        super().__init__()
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
            try:
                ppl_model = PPLModel(
                    self.model_path,
                    self.tensor_parallel_size,
                    use_gpu_ids=use_gpus,
                )
                all_docs = [doc for doc in data]
                texts = [doc.text for doc in all_docs]
                ppls = ppl_model.calc_ppl(texts)
                with self.output_folder.open(f"{rank:05d}.json", mode="w") as f:
                    json.dump(ppls, f)
                for doc, ppl in zip(all_docs, ppls):
                    doc.metadata["perplexity"] = ppl
                    yield doc
            except Exception as e:
                logger.error(e)
                del ppl_model  # ensure GPU is released
