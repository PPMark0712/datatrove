import os
import json
from transformers import AutoTokenizer

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.logging import logger
from .ppl_model import PPLModel


class Encoder(PipelineStep):
    name = "Encoder"
    type = "Perplexity"

    def __init__(
        self,
        output_folder: DataFolderLike,
        model_path: str,
        max_model_len: int = 4096,
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.model_path = model_path
        self.max_model_len = max_model_len

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for doc in data:
                token_ids = tokenizer(doc.text)["input_ids"]
                if len(token_ids) > self.max_model_len - 1:
                    token_ids = token_ids[:self.max_model_len - 1]
                output_file.write(json.dumps(token_ids) + "\n")