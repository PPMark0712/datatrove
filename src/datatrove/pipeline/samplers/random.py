import random
from typing import List, Literal

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.io import DataFolderLike
from .base import BaseSampler, BaseIndexSampler
from .utils import read_score_file


class IndexRandomSampler(BaseIndexSampler):
    def __init__(self, sample_rate: float) -> None:
        super().__init__()
        self.sample_rate = sample_rate

    def sample_by_doc_count(self, indexes: List[int]) -> List[int]:
        sample_size = int(len(indexes) * self.sample_rate)
        return random.sample(indexes, sample_size)

    def sample_by_token_limit(self, indexes: List[int], token_counts: List[int]) -> List[int]:
        shuffled_indexes = indexes.copy()
        random.shuffle(shuffled_indexes)
        total_tokens = sum(token_counts[i] for i in indexes)
        sample_tokens = int(total_tokens * self.sample_rate)
        sampled_indexes = []
        current_tokens = 0
        for i in shuffled_indexes:
            current_tokens += token_counts[i]
            sampled_indexes.append(i)
            if current_tokens >= sample_tokens:
                break
        return sampled_indexes


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
            doc_count = int(f.read())
        indexes = list(range(doc_count))
        random_index_sampler = IndexRandomSampler(self.sample_rate)
        if self.unit == "doc":
            return random_index_sampler.sample_by_doc_count(indexes)
        elif self.unit == "token":
            token_counts = read_score_file(self.token_count_folder, rank)
            return random_index_sampler.sample_by_token_limit(indexes, token_counts)
