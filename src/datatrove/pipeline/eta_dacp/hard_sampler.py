import json

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.io import DataFolderLike, get_datafolder


class HardSampler(PipelineStep):
    name = "Hard Sampler"

    def __init__(
        self,
        score_folder: DataFolderLike,
        top_p: float,
        highest: bool = True,
        unit="doc"  # doc or token
    ):
        super().__init__()
        self.score_folder = get_datafolder(score_folder)
        self.top_p = top_p
        self.highest = highest
        self.unit = unit

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.track_time():
            with self.score_folder.open(f"{rank:05d}.json", mode="r") as score_file:
                scores = json.load(score_file)
            all_docs = [doc for doc in data]
            indexes = sorted(range(len(all_docs)), key=lambda i: scores[i], reverse=self.highest)

            if self.unit == "doc":
                top_p_index = int(self.top_p * len(indexes))
                sampled_indexes = indexes[:top_p_index]
            elif self.unit == "token":
                # run TokensCounter in Pipeline before HardSampler
                total_tokens = sum(len(doc["tokens"]) for doc in all_docs)
                token_budget = int(self.top_p * total_tokens)
                current_token_count = 0
                sampled_indexes = []
                for i in indexes:
                    sampled_indexes.append(i)
                    current_token_count += len(all_docs[i]["tokens"])
                    if current_token_count >= token_budget:
                        break
            
            for i in sampled_indexes:
                yield all_docs[i]
