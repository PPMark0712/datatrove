import random
from typing import List, Literal

from datatrove.io import DataFolderLike
from .base import BaseSampler
from .utils import read_score_file


class DocumentCounter(PipelineStep):
    """Count the number of documents in the pipeline."""
    type = "Document Counter"

    def __init__(self, output_folder: DataFolderLike):
        super().__init__()
        self.output_folder = get_data_folder(output_folder)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        count = sum(1 for _ in data)
        with self.output_folder.open(f"{rank:05d}.txt", "w") as f:
            f.write(str(count))


class RandomSampler(BaseSampler):
    """Random sample data."""
    type = "Sampler"
    name = "Random Sampler"

    def __init__(
        self,
        sample_rate: float,
        count_folder: DataFolderLike,
        seed: int = 42,
        unit: Literal["doc", "token"] = "doc",
        token_count_folder: DataFolderLike = None
    ):
        super().__init__(score_folder)
        self.sample_rate = sample_rate
        self.count_folder = get_data_folder(count_folder)
        self.seed = seed
        self.higher_is_better = higher_is_better
        self.unit = unit
        self.token_count_folder = get_data_folder(token_count_folder) if token_count_folder else None

    def sample_indexes(self, rank: int = 0, world_size: int = 1) -> List[int]:
        random.seed(self.seed)
        with self.count_folder.open(f"{rank:05d}.txt", "r") as f:
            total_count = int(f.read().strip())
        indexes = list(range(total_count))
        random.shuffle(indexes)
        sample_size = int(total_count * self.sample_rate)

        if self.unit == "doc":
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
