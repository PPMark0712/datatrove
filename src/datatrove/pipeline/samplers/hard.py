from typing import List, Literal

from datatrove.io import DataFolderLike
from .base import BaseSampler, BaseIndexSampler
from .utils import read_score_file


class IndexHardSampler(BaseIndexSampler):
    def __init__(self, sample_rate: float, reverse: bool = False) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.reverse = reverse

    def sample_by_doc_count(self, indexes: List[int]) -> List[int]:
        if self.reverse:
            indexes = indexes[::-1]
        sample_size = int(len(indexes) * self.sample_rate)
        return indexes[:sample_size]

    def sample_by_token_limit(self, indexes: List[int], token_counts: List[int]) -> List[int]:
        if self.reverse:
            indexes = indexes[::-1]
        total_tokens = sum(token_counts[i] for i in indexes)
        sample_tokens = int(total_tokens * self.sample_rate)
        sampled_indexes = []
        current_tokens = 0
        for i in indexes:
            current_tokens += token_counts[i]
            sampled_indexes.append(i)
            if current_tokens >= sample_tokens:
                break
        return sampled_indexes


class HardSampler(BaseSampler):
    """Sample data with highest score."""
    type = "Sampler"
    name = "Hard Sampler"

    def __init__(
        self,
        score_folder: DataFolderLike,
        sample_rate: float,
        higher_is_better: bool = True,
        unit: Literal["doc", "token"] = "doc",
        token_count_folder: DataFolderLike = None
    ):
        super().__init__()
        self.score_folder = get_data_folder(score_folder)
        self.sample_rate = sample_rate
        self.higher_is_better = higher_is_better
        self.unit = unit
        self.token_count_folder = get_data_folder(token_count_folder) if token_count_folder else None

    def get_sampled_indexes(self, rank: int = 0, world_size: int = 1) -> List[int]:
        score = read_score_file(self.score_folder, rank)
        indexes = list(range(len(score)))
        indexes.sort(key=lambda x: score[x], reverse=self.higher_is_better)
        index_hard_sampler = IndexHardSampler(self.sample_rate)
        if self.unit == "doc":
            return index_hard_sampler.sample_by_doc_count(indexes)
        elif self.unit == "token":
            token_counts = read_score_file(self.token_count_folder, rank)
            return index_hard_sampler.sample_by_token_limit(indexes, token_counts)
