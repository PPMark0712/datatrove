from typing import List, Literal

from datatrove.io import DataFolderLike
from .base import BaseSampler
from .utils import read_score_file


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

        if self.unit == "doc":
            sample_size = int(len(indexes) * self.sample_rate)
            return indexes[:sample_size]
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
