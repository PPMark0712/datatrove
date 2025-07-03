from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger
from datatrove.utils.word_tokenizers import load_word_tokenizer


class LexicalDifficultyLabeler(PipelineStep):

    def __init__(
        self,
        language: str,
        input_folder: DataFolderLike,
    ):
        super().__init__()
        self.language = language
        self.input_folder = input_folder

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            pass