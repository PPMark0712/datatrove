import json
import random

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.logging import logger

class ProbabilityCalculator(PipelineStep):
    name = "Probability Calculator"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        sample_token_rate: float = 0.2,
        rate_for_hard_sample: float = 0.4,
        gc_components: list[str] = ["pos_ent", "con_ent", "dep_ent", "avg_dep_height", "avg_dep_dis"],
        weights: list[float] = [0.2, 0.2, 0.2, 0.2, 0.2],
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.sample_token_rate = sample_token_rate
        self.rate_for_hard_sample = rate_for_hard_sample
        self.gc_components = gc_components
        self.weights = weights

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.track_time():
            world_size = len(self.input_folder.glob("*.jsonl"))
            gc_data = []
            for rank in range(world_size):
                input_file = self.input_folder.open(f"{rank:05d}.jsonl", mode="r")
                for doc_id, line in enumerate(input_file):
                    item = json.loads(line)
                    gc_data.append({
                        "rank": rank,
                        "doc_id": doc_id,
                        "token_count": item["token_count"],
                        "gc_score": sum(item["normalized_gc"][gc] * w for gc, w in zip(self.gc_components, self.weights))
                    })
            idxs = list(range(len(gc_data)))
            idxs.sort(key=lambda x: gc_data[x]["gc_score"])
            total_tokens = sum(x["token_count"] for x in gc_data)
            sample_tokens = int(total_tokens * self.sample_token_rate)
            logger.info(f"total_tokens: {total_tokens}, expected sample_tokens: {sample_tokens}")
            hard_sample_tokens = int(sample_tokens * self.rate_for_hard_sample)
            cdf_sample_tokens = sample_tokens - hard_sample_tokens

            # hard sample
            hard_sample_idx = len(idxs)
            cur_hard_sample_tokens = 0
            for i in range(len(idxs) - 1, -1, -1):
                if cur_hard_sample_tokens + gc_data[idxs[i]]["token_count"] > hard_sample_tokens:
                    break
                hard_sample_idx = i
                cur_hard_sample_tokens += gc_data[idxs[i]]["token_count"]

            hard_sample_probs = [{
                **gc_data[i],
                "prob": 1.0,
            } for i in idxs[hard_sample_idx:]] if hard_sample_idx < len(idxs) else []

            # cdf sample
            base_expected_tokens = 0
            total_tokens = sum(gc_data[idxs[i]]["token_count"] for i in range(hard_sample_idx))
            accumulated_tokens = 0
            for i in range(hard_sample_idx):
                accumulated_tokens += gc_data[idxs[i]]["token_count"]
                base_expected_tokens += accumulated_tokens / total_tokens * gc_data[idxs[i]]["token_count"]

            r = cdf_sample_tokens / base_expected_tokens if base_expected_tokens > 0 else 0
            logger.info(f"r={r}")
            cdf_sample_probs = []
            accumulated_tokens = 0
            for i in range(hard_sample_idx):
                accumulated_tokens += gc_data[idxs[i]]["token_count"]
                cdf = accumulated_tokens / total_tokens
                cdf_sample_probs.append({
                    **gc_data[idxs[i]],
                    "prob": min(1, r * cdf),
                })

            prob_result = hard_sample_probs + cdf_sample_probs
            prob_result.sort(key=lambda x: (x["rank"], x["doc_id"]))
            rank_dict = {rank: [] for rank in range(world_size)}
            for item in prob_result:
                rank_dict[item["rank"]].append(item)
            for rank in rank_dict:
                rank_dict[rank].sort(key=lambda x: x["doc_id"])

            for rank, result in rank_dict.items():
                output_file = self.output_folder.open(f"{rank:05d}.json", mode="w")
                probs = [item["prob"] for item in result]
                output_file.write(json.dumps(probs))

            
class ProbabilitySampler(PipelineStep):
    name = "Probability Sampler"

    def __init__(
        self,
        prob_folder: DataFolderLike,
        gc_folder: DataFolderLike,
        seed: int = 42,
    ):
        super().__init__()
        self.prob_folder = get_datafolder(prob_folder)
        self.gc_folder = get_datafolder(gc_folder)
        random.seed(seed)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.track_time():
            prob_file = self.prob_folder.open(f"{rank:05d}.json", mode="r")
            probs = json.load(prob_file)
            gc_file = self.gc_folder.open(f"{rank:05d}.json", mode="r")
            
            for doc, prob, gc_line in zip(data, probs, gc_file):
                rand_val = random.uniform(0, 1)
                if rand_val <= prob:
                    gc_item = json.load(gc_line)
                    doc.metadata["org_gc"] = gc_item["org_gc"]
                    doc.metadata["normalized_gc"] = gc_item["normalized_gc"]
                    yield doc