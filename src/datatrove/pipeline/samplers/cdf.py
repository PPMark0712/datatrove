import random
from typing import List, Literal

from datatrove.io import DataFolderLike
from .base import BaseSampler, BaseIndexSampler
from .random import IndexRandomSampler
from .hard import IndexHardSampler
from .utils import read_score_file


class CDFBalanceIndexSampler(BaseIndexSampler):
    def __init__(self, sample_rate: float, reverse: bool = False) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.reverse = reverse


class CDFSampler(BaseSampler):
    """Sample data with highest score."""
    type = "Sampler"
    name = "CDF Sampler"

    def __init__(
        self,
        score_folder: DataFolderLike,
        sample_rate: float,
        hard_sample_ratio: float = 0.4,
        reverse: bool = False,
        unit: Literal["doc", "token"] = "doc",
        token_count_folder: DataFolderLike = None,
        seed: int = 42
    ):
        super().__init__()
        self.score_folder = get_data_folder(score_folder)
        self.sample_rate = sample_rate
        self.hard_sample_ratio = hard_sample_ratio
        self.reverse = reverse
        self.unit = unit
        self.token_count_folder = get_data_folder(token_count_folder) if token_count_folder else None

    def get_sampled_indexes(self, rank: int = 0, world_size: int = 1) -> List[int]:
        random.seed(seed)
        score = read_score_file(self.score_folder, rank)
        indexes = list(range(len(score)))
        indexes.sort(key=lambda x: score[x], reverse=self.reverse)

        if self.unit == "doc":
            sample_count = int(len(indexes) * self.sample_rate)
            hard_sample_count = int(sample_count * self.hard_sample_ratio)
            soft_sample_count = sample_count - hard_sample_count

            hard_sample_indexes = indexes[-hard_sample_count:] if hard_sample_count else []
            remain_indexes = indexes[:-hard_sample_count] if hard_sample_count else indexes

            expected_soft_sample_count = 0
            for i in remain_indexes:
                if random.random() < expected_soft_sample_count / len(remain_indexes):
                    soft_sample_indexes.append(i)
                    expected_soft_sample_count -= 1


        elif self.unit == "token":
            token_counts = read_score_file(self.token_count_folder, rank)
            current_tokens = 0
            total_tokens = sum(token_counts)
            sample_tokens = int(total_tokens * self.sample_rate)
            sampled_indexes = []
            for i in indexes:
                current_tokens += token_counts[i]
                sampled_indexes.append(i)
                if current_tokens >= sample_tokens:
                    break
            return sampled_indexes
