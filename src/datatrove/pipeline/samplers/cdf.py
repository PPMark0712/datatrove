import random
from typing import List, Literal

from datatrove.io import DataFolderLike
from .base import BaseSampler, BaseIndexSampler
from .random import IndexRandomSampler
from .hard import IndexHardSampler
from .utils import read_score_file


class IndexCdfSampler(BaseIndexSampler):
    def __init__(self, sample_rate: float, reverse: bool = False) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.reverse = reverse
        assert self.sample_rate <= 0.5

    def sample_by_doc_count(self, indexes: List[int]) -> List[int]:
        if self.reverse:
            indexes = indexes[::-1]
        n = len(indexes)
        sample_count = int(n * self.sample_rate)
        expected_count = (1 + n) / 2
        r = sample_count / expected_count
        sampled_indexes = []
        for i, idx in enumerate(indexes):
            prob = r * i / n
            if random.uniform(0, 1) <= prob:
                sampled_indexes.append(idx)
        return sampled_indexes

    def sample_by_token_limit(self, indexes: List[int], token_counts: List[int]) -> List[int]:
        if self.reverse:
            indexes = indexes[::-1]
        n = len(indexes)
        total_tokens = sum(token_counts[i] for i in indexes)
        sample_tokens =  int(n * self.sample_rate)
        expected_tokens = 0
        accumulated_tokens = 0
        for i in indexes:
            accumulated_tokens += tokens[i]
            prob = accumulated_tokens / total_tokens
            expected_tokens += prob * tokens[i]

        r = sample_tokens / expected_tokens
        sampled_indexes = []
        accumulated_tokens = 0
        for i in indexes:
            accumulated_tokens += tokens[i]
            prob = r * accumulated_tokens / total_tokens
            if random.uniform(0, 1) <= prob:
                sampled_indexes.append(idx)
        return sampled_indexes


class IndexCdfBalancedSampler(BaseIndexSampler):
    def __init__(
        self,
        sample_rate: float,
        rate_for_hard_sample: float = 0.4,
        hard_sample_range: float = 2.0,
        reverse: bool = False
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.rate_for_hard_sample = rate_for_hard_sample
        self.hard_sample_range = hard_sample_range
        self.reverse = reverse

    def sample_by_doc_count(self, indexes: List[int]) -> List[int]:
        if self.reverse:
            indexes = indexes[::-1]
        n = len(indexes)
        sample_count = int(n * self.sample_rate)
        hard_sample_count = int(n * self.rate_for_hard_sample)
        cdf_sample_count = sample_count - hard_sample_count
        hard_sample_split_index = n - hard_sample_count * self.hard_sample_range

        if hard_sample_split_index < n:
            hard_sample_split = indexes[hard_sample_split_index:]
            random_index_sampler = IndexRandomSampler(hard_sample_count / len(hard_sample_split), reverse)
            hard_sample_result = random_index_sampler.sample_by_doc_count(hard_sample_split)
        else:
            hard_sample_result = []

        if hard_sample_split_index > 0:
            cdf_sample_split = indexes[:hard_sample_split_index]
            cdf_sampler = IndexCdfSampler(cdf_sample_count / len(cdf_sample_split), reverse)
            cdf_sample_result = cdf_sampler.sample_by_doc_count(cdf_sample_split)
        else:
            cdf_sample_result = []

        return cdf_sample_result + hard_sample_result

    def sample_by_token_limit(self, indexes: List[int], token_counts: List[int]) -> List[int]:
        if self.reverse:
            indexes = indexes[::-1]
        n = len(indexes)
        total_tokens = sum(token_counts[i] for i in indexes)
        hard_sample_tokens = int(n * self.rate_for_hard_sample)
        cdf_sample_tokens = total_tokens - hard_sample_tokens

        accumulated_tokens = 0
        hard_sample_split_index = n
        for i in reversed(range(n)):
            accumulated_tokens += token_counts[i]
            if accumulated_tokens >= hard_sample_tokens:
                break
            hard_sample_split_index = i

        if hard_sample_split_index < n:
            hard_sample_split = indexes[hard_sample_split_index:]
            sum_tokens = sum(token_counts[i] for i in hard_sample_split)
            random_index_sampler = IndexRandomSampler(hard_sample_tokens / sum_tokens, reverse)
            hard_sample_result = random_index_sampler.sample_by_token_limit(hard_sample_split, token_counts)
        else:
            hard_sample_result = []

        if hard_sample_split_index > 0:
            cdf_sample_split = indexes[:hard_sample_split_index]
            sum_tokens = sum(token_counts[i] for i in cdf_sample_split)
            cdf_sampler = IndexCdfSampler(cdf_sample_tokens / sum_tokens, reverse)
            cdf_sample_result = cdf_sampler.sample_by_doc_count(cdf_sample_split)
        else:
            cdf_sample_result = []

        return cdf_sample_result + hard_sample_result


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
