from abc import abstractmethod
from typing import List, Literal

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


class BaseSampler(PipelineStep):
    """Base module for Samplers. Sample data by score."""
    type = "Sampler"

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_sampled_indexes(self, rank: int = 0, world_size: int = 1) -> List[int]:
        """Get sampled document indexes (sorting not required)"""
        raise NotImplementedError

    def _normalize_indexes(self, indexes: List[int]) -> List[int]:
        sorted_indexes = sorted(set(indexes))
        if len(sorted_indexes) != len(indexes):
            raise ValueError("duplicate indexes in sampled_indexes")
        return sorted_indexes

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.track_time():
            sampled_indexes = self.get_sampled_indexes(rank, world_size)
            sorted_indexes = self._normalize_indexes(sampled_indexes)
            p = 0
            origin_doc_count = 0
            sampled_doc_count = 0
            for i, doc in enumerate(data):
                origin_doc_count += 1
                if p < len(sorted_indexes) and i == sorted_indexes[p]:
                    yield doc
                    p += 1
                    sampled_doc_count += 1
            sample_rate = sampled_doc_count / origin_doc_count if origin_doc_count else 0
            logger.info(f"sampled {sampled_doc_count}/{origin_doc_count} ({sample_rate * 100:.2f}%) docs.")
