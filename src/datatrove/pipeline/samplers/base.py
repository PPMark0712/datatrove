from abc import ABC, abstractmethod
from typing import List

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


class BaseIndexSampler(ABC):
    @abstractmethod
    def sample_by_doc_count(self, indexes: List[int]) -> List[int]:
        """Sample documents by specified count constraint (preserves complete documents).

        Args:
            indexes: List of document indexes to sample from (1D list of integers)

        Returns:
            List[int]: Sampled document indexes (sorted or in original order based on subclass implementation)
        """
        raise NotImplementedError

    @abstractmethod
    def sample_by_token_limit(self, indexes: List[int], token_counts: List[int]) -> List[int]:
        """Sample documents by accumulated token limit (preserves complete documents, no splitting).
        Critical note: `token_counts` is a global mapping array where `token_counts[i]` returns the token count
        of the document with original index `i` — it is NOT a list of the same length as `indexes`.

        Args:
            indexes: List of document indexes to sample from (1D list of integers)
            token_counts: Global token count mapping array (indexed by original document index). 
                For any index `i` in `indexes`, `token_counts[i]` gives the corresponding document's token count.

        Returns:
            List[int]: Sampled document indexes (complete documents only, accumulated tokens ≤ limit)
        """
        raise NotImplementedError


class BaseSampler(PipelineStep, ABC):
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
