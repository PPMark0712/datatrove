import math
import json

from collections import Counter
from multiprocessing import current_process

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.cdf_gc import PartOfSpeechPredictor, DependencyParser
from datatrove.utils.logging import logger
from datatrove.io import DataFolderLike, get_datafolder

def preprocess_text(text: str) -> str:
    new_lines = []
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("|") and line.endswith("|"):
            line = line.replace("|", " ")
        new_lines.append(line)
    return "\n".join(new_lines)


def calc_counter_entropy(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


class DocumentPartOfSpeechPredictor(PipelineStep):
    name = "Document Part of Speech Predictor"
    type = "CDF-GC"

    def __init__(
        self,
        language: str,
        output_folder: DataFolderLike,
        **kwargs
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.pos_predictor = PartOfSpeechPredictor(language, **kwargs)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for doc_idx, doc in enumerate(data):
                words, pos_tags = self.pos_predictor.predict(preprocess_text(doc.text))
                content_words = self.pos_predictor.get_content_words(words, pos_tags)
                content_word_counter = Counter(content_words)
                output_file.write(json.dumps({
                    "words": words,
                    "pos_tags": pos_tags,
                    "content_words_counter": content_word_counter,
                    "token_count": doc.metadata["token_count"],
                }, ensure_ascii=False) + "\n")


class LexicalDiversityCalculator(PipelineStep):
    name = "Lexical Diversity Calculator"
    type = "CDF-GC"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        **kwargs
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            input_file = self.input_folder.open(f"{rank:05d}.jsonl", mode="r")
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for line in input_file:
                item = json.loads(line)
                pos_counter = Counter(item["pos_tags"])
                pos_ent = calc_counter_entropy(pos_counter)
                content_word_counter = item["content_words_counter"]
                con_ent = calc_counter_entropy(content_word_counter)
                output_file.write(json.dumps({
                    "pos_ent": pos_ent,
                    "con_ent": con_ent,
                    "token_count": item["token_count"],
                }) + "\n")


class DocumentDependencyParser(PipelineStep):
    name = "Document Dependency Parser"
    type = "CDF-GC"

    def __init__(
        self,
        language: str,
        output_folder: DataFolderLike,
        n_gpus: int,
        workers_per_gpu: int,
        **kwargs
    ):
        super().__init__()
        self.language = language
        self.output_folder = get_datafolder(output_folder)
        self.n_gpus = n_gpus
        self.workers_per_gpu = workers_per_gpu
        self.kwargs = kwargs

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            identity = current_process()._identity
            if identity:
                local_rank = identity[0] - 1
            else:
                local_rank = 0
            model = DependencyParser(self.language, local_rank % self.n_gpus, **self.kwargs)
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for doc_id, doc in enumerate(data):
                parsed_sentences = model.predict(doc.text)
                output_file.write(json.dumps({
                    # "doc_id": doc_id,
                    "parsed_sentences": parsed_sentences
                }, ensure_ascii=False) + "\n")


def calc_tree_height(parents: list[int]) -> int:
    n = len(parents)
    heights = [-1] * n

    def get_height(i: int) -> int:
        if parents[i] == -1:
            return 0
        if heights[i] != -1:
            return heights[i]
        heights[i] = get_height(parents[i]) + 1
        return heights[i]

    return max(get_height(i) for i in range(n))


class SyntacticComplexityCalculator(PipelineStep):
    name = "Syntactic Complexity Calculator"
    type = "CDF-GC"

    def __init__(
        self,
        input_folder: DataFolderLike,  # dependency parser output
        output_folder: DataFolderLike,  # syntactic complexity output
        **kwargs
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.track_time():
            input_file = self.input_folder.open(f"{rank:05d}.jsonl", mode="r")
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for doc_id, line in enumerate(input_file):
                item = json.loads(line)
                parsed_sentences = item["parsed_sentences"]
                dep_label_counter = Counter(label for sentence in parsed_sentences for label in sentence["dep_labels"])
                dep_ent = calc_counter_entropy(dep_label_counter)

                total_tree_cnt = len(parsed_sentences)
                total_tree_height = 0
                total_dependency_distance = 0
                total_edge_cnt = 0
                for sentence in parsed_sentences:
                    parents = sentence["parents"]
                    total_edge_cnt += len(parents) - 1
                    total_tree_height += calc_tree_height(parents)
                    for i, parent in enumerate(parents):
                        if parent == -1:
                            continue
                        total_dependency_distance += abs(i - parent)

                avg_dep_height = total_tree_height / total_tree_cnt
                avg_dep_dis = total_dependency_distance / total_edge_cnt

                output_file.write(json.dumps({
                    # "doc_id": doc_id,
                    "dep_ent": dep_ent,
                    "avg_dep_height": avg_dep_height,
                    "avg_dep_dis": avg_dep_dis
                }) + "\n")


class GcCombiner(PipelineStep):
    name = "GC Combiner"
    type = "CDF-GC"

    def __init__(
        self,
        lexical_diversity_folder: DataFolderLike,
        syntactic_complexity_folder: DataFolderLike,
        output_folder: DataFolderLike,
    ):
        super().__init__()
        self.lexical_diversity_folder = get_datafolder(lexical_diversity_folder)
        self.syntactic_complexity_folder = get_datafolder(syntactic_complexity_folder)
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.track_time():
            lexical_diversity_file = self.lexical_diversity_folder.open(f"{rank:05d}.jsonl", mode="r")
            syntactic_complexity_file = self.syntactic_complexity_folder.open(f"{rank:05d}.jsonl", mode="r")
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")

            def lexical_diversity_generator():
                for line in lexical_diversity_file:
                    item = json.loads(line)
                    yield {
                        "pos_ent": item["pos_ent"],
                        "con_ent": item["con_ent"],
                        "token_count": item["token_count"]
                    }

            def syntactic_complexity_generator():
                for line in syntactic_complexity_file:
                    item = json.loads(line)
                    yield {
                        "dep_ent": item["dep_ent"],
                        "avg_dep_height": item["avg_dep_height"],
                        "avg_dep_dis": item["avg_dep_dis"]
                    }

            lexical_diversity_items = lexical_diversity_generator()
            syntactic_complexity_items = syntactic_complexity_generator()
            for doc_id, (lexical_diversity_item, syntactic_complexity_item) in enumerate(zip(lexical_diversity_items, syntactic_complexity_items)):
                output_file.write(json.dumps({
                    # "doc_id": doc_id,
                    **lexical_diversity_item,
                    **syntactic_complexity_item,
                }) + "\n")


class GcNormalizer(PipelineStep):
    name = "GC Normalizer"
    type = "CDF-GC"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        gc_components: list[str] = ["pos_ent", "con_ent", "dep_ent", "avg_dep_height", "avg_dep_dis"],
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        self.gc_components = gc_components

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.track_time():
            world_size = len(self.input_folder.glob("*.jsonl"))

            # gather all GC data and get min max dict
            gc_data = []
            min_max_values_dict = {}
            for gc_component in self.gc_components:
                min_max_values_dict[gc_component] = [float("inf"), float("-inf")]
            for rank in range(world_size):
                input_file = self.input_folder.open(f"{rank:05d}.jsonl", mode="r")
                for line in input_file:
                    item = json.loads(line)
                    gc_data.append({
                        "rank": rank,
                        "doc_id": item["doc_id"],
                        "token_count": item["token_count"],
                        "org_gc": {
                            gc_component: item[gc_component] for gc_component in self.gc_components
                        }
                    })
                    for gc_component in self.gc_components:
                        min_max_values_dict[gc_component][0] = min(min_max_values_dict[gc_component][0], item[gc_component])
                        min_max_values_dict[gc_component][1] = max(min_max_values_dict[gc_component][1], item[gc_component])

            # min-max normalize
            for item in gc_data:
                normalized_gc = {}
                for gc_component in self.gc_components:
                    min_value, max_value = min_max_values_dict[gc_component]
                    val = 0
                    if max_value != min_value:
                        val = (item["org_gc"][gc_component] - min_value) / (max_value - min_value)
                    normalized_gc[gc_component] = val
                item["normalized_gc"] = normalized_gc

            # write to output
            gc_data.sort(key=lambda x: (x["rank"], x["doc_id"]))
            rank_dict = {rank: [] for rank in range(world_size)}
            for item in gc_data:
                rank_dict[item["rank"]].append(item)
            for rank, rank_data in rank_dict.items():
                output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
                for item in rank_data:
                    item.pop("rank")
                    output_file.write(json.dumps(item) + "\n")