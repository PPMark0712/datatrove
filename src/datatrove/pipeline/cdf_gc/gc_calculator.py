import math
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import Counter
import jieba
import nltk
from ltp import LTP

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.cdf_gc import PartOfSpeechPredictor, DependencyParser
from datatrove.utils.logging import logger
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.text import split_into_sentences

def calc_counter_entropy(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


@dataclass
class GCItem:
    """GC Item for a given document

    Args:
        file_id: file id
        doc_id: document id
        reader_id: reader id. Used to know from where the next signature should be requested
    """
    file_id: int
    doc_id: int
    gc_data: dict

    def to_dict(self):
        return {
            "file_id": self.file_id,
            "doc_id": self.doc_id,
            "gc_data": self.gc_data,
        }


class LexicalDiversityCalculator(PipelineStep):
    name = "Lexical Diversity Calculator"
    _requires_dependencies = ["jieba", "nltk"]

    def __init__(
        self,
        language: str,
        output_folder: DataFolderLike,
        **kwargs
    ):
        super().__init__()
        self.language = language
        self.output_folder = output_folder
        self.pos_predictor = PartOfSpeechPredictor(language, **kwargs)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for doc_idx, doc in enumerate(data):
                word_with_pos_tags = self.pos_predictor.predict(doc.text)
                content_words = self.pos_predictor.get_content_words(word_with_pos_tags)
                pos_tag_counter = Counter(pos for word, pos in word_with_pos_tags)
                pos_tag_entropy = calc_counter_entropy(pos_tag_counter)
                content_word_counter = Counter(content_words)
                content_word_entropy = calc_counter_entropy(content_word_counter)
                output_file.write(json.dumps({
                    "doc_id": doc_idx,
                    "pos_tag_entropy": pos_tag_entropy,
                    "content_word_entropy": content_word_entropy
                }) + "\n")


class DocumentDependencyParser(PipelineStep):
    name = "Document Dependency Parser"
    _requires_dependencies = ["ltp"]

    def __init__(
        self,
        language: str,
        output_folder: DataFolderLike,
        **kwargs
    ):
        super().__init__()
        self.language = language
        self.output_folder = output_folder
        self.dependency_parser = DependencyParser(**kwargs)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for doc_idx, doc in enumerate(data):
                parsed_sentences = self.dependency_parser.predict(doc.text, rank)
                output_file.write(json.dumps({
                    "doc_id": doc_idx,
                    "parsed_sentences": parsed_sentences
                }) + "\n")


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
    _requires_dependencies = ["ltp"]

    def __init__(
        self,
        language: str,
        input_folder: DataFolderLike,  # dependency parser output
        output_folder: DataFolderLike,  # syntactic complexity output
        **kwargs
    ):
        super().__init__()
        self.language = language
        self.input_folder = input_folder
        self.output_folder = output_folder

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        with self.track_time():
            input_file = self.input_folder.open(f"{rank:05d}.jsonl", mode="r")
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", mode="w")
            for doc_id, line in input_file:
                item = json.loads(line)
                parsed_sentences = item["parsed_sentences"]
                dependency_label_counter = Counter(label for sentence in parsed_sentences for label in sentence["dependency_labels"])
                dependency_label_entropy = calc_counter_entropy(dependency_label_counter)

                total_tree_cnt = len(parsed_sentences)
                total_tree_height = 0
                total_dependency_distance = 0
                for sentence in parsed_sentences:
                    parents = sentence["parents"]
                    total_tree_height += calc_tree_height(parents)
                    for i, parent in enumerate(parents):
                        if parent == -1:
                            continue
                        total_dependency_distance += abs(i - parent)

                avg_tree_height = total_tree_height / total_tree_cnt
                avg_dependency_distance = total_dependency_distance / total_tree_cnt

                output_file.write(json.dumps({
                    "doc_id": doc_id,
                    "dependency_label_entropy": dependency_label_entropy,
                    "avg_tree_height": avg_tree_height,
                    "avg_dependency_distance": avg_dependency_distance
                }) + "\n")