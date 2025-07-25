import os
import json
import math
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger

def calc_counter_entropy(counter: dict):
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


def get_wordnet_pos(nltk_pos_tag:str):
    d = {
        "J": wn.ADJ,
        "V": wn.VERB,
        "N": wn.NOUN,
        "R": wn.ADV
    }
    return d.get(nltk_pos_tag[0], None)


def get_lemmatize_pos(nltk_pos_tag: str):
    d = {
        "N": "n",
        "V": "v",
        "J": "a",
        "R": "r",
    }
    return d.get(nltk_pos_tag[0], "n")


class LexicalDifficultySorter(PipelineStep):
    name = "🔤 - Lexical difficulty sorter"
    type = "cl"

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        self._check_nltk_dependencies()
        self.stop_words = set(stopwords.words("english"))
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "concreteness_dict.json"), "r") as f:
            self.concrectness_dict = json.load(f)

    def _check_nltk_dependencies(self):
        if "nltk_path" in self.kwargs:
            nltk.data.path.append(self.kwargs["nltk_path"])
        nltk_dependencies = [
            "wordnet",
            "stopwords",
            "punkt_tab",
            "averaged_perceptron_tagger_eng",
        ]
        for package in nltk_dependencies:
            try:
                nltk.data.find(package)
            except:
                nltk.download(package, download_dir=self.kwargs.get("nltk_path", None))

    def is_valid_word(self, word: str) -> bool:
        return len(word) > 1 and any(c.isalpha() for c in word) and word not in self.stop_words

    def calc_scores(self, difficulty_data: list):
        """Normalize all difficulty dimensions using min-max scaling"""
        dimensions = difficulty_data[0].keys()
        min_vals = {dim: float('inf') for dim in dimensions}
        max_vals = {dim: float('-inf') for dim in dimensions}

        for doc in difficulty_data:
            for dim in dimensions:
                if doc[dim] < min_vals[dim]:
                    min_vals[dim] = doc[dim]
                if doc[dim] > max_vals[dim]:
                    max_vals[dim] = doc[dim]

        normalized_data = []
        for doc in difficulty_data:
            normalized_doc = {}
            for dim in dimensions:
                if max_vals[dim] == min_vals[dim]:
                    normalized_doc[dim] = 0.5
                else:
                    normalized_doc[dim] = (doc[dim] - min_vals[dim]) / (max_vals[dim] - min_vals[dim])

            score = sum(normalized_doc.values()) / len(dimensions)
            normalized_doc["difficulty_score"] = score
            normalized_data.append(normalized_doc)

        return normalized_data

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            if "nltk_path" in self.kwargs:
                nltk.data.path.append(self.kwargs["nltk_path"])
            lemmatizer = WordNetLemmatizer()
            difficulty_data = []
            all_docs = []
            # 需要用all_docs保存完整的文件，占用大量内存
            for doc in data:
                all_docs.append(doc)
                text = doc.text
                sentences = sent_tokenize(text)
                total_valid_words = 0
                sum_hypernym_depth = 0
                sum_synonym_count = 0
                sum_concreteness_score = 0
                total_words_in_concreteness_dict = 0
                hypernym_counter = defaultdict(int)

                for sentence in sentences:
                    words = word_tokenize(sentence)
                    words_with_pos = pos_tag(words)
                    for word, pos in words_with_pos:
                        if not self.is_valid_word(word):
                            continue
                        total_valid_words += 1

                        synsets = wn.synsets(word)
                        sum_synonym_count += len(synsets)
                        wn_pos = get_wordnet_pos(pos)
                        synset = lesk(words, word, pos=wn_pos)
                        if synset is not None:
                            path = synset.hypernym_paths()[0]
                            # logger.info(f"{word}, {path}, {synset}")
                            sum_hypernym_depth += len(path)
                            for hypernym in path[-3:]:
                                hypernym_counter[hypernym.name()] += 1                            
                        else:
                            # logger.info("lesk error")
                            pass

                        org_word = lemmatizer.lemmatize(word, get_lemmatize_pos(pos))
                        concreteness_score = self.concrectness_dict.get(org_word, None)
                        if concreteness_score is not None:
                            sum_concreteness_score += concreteness_score
                            total_words_in_concreteness_dict += 1

                avg_hypernym_depth = sum_hypernym_depth / total_valid_words if total_valid_words else 0
                avg_synonym_count = sum_synonym_count / total_valid_words if total_valid_words else 0
                hypernym_entropy = calc_counter_entropy(hypernym_counter)
                avg_concreteness_score = sum_concreteness_score / total_words_in_concreteness_dict if total_words_in_concreteness_dict else 0

                difficulty_data.append({
                    "avg_hypernym_depth": avg_hypernym_depth,
                    # "synonym_richness": avg_synonym_count,
                    # "sense_dispersion": hypernym_entropy,
                    # "abstract_score": avg_concreteness_score
                })

            normalized_difficulty_data = self.calc_scores(difficulty_data)
            for doc, normalized_difficulty, org_difficulty in zip(all_docs, normalized_difficulty_data, difficulty_data):
                doc.metadata["difficulty"] = org_difficulty
                doc.metadata["normalized_difficulty"] = normalized_difficulty
            all_docs.sort(key=lambda x: x.metadata["normalized_difficulty"]["difficulty_score"])
            for doc in all_docs:
                yield doc
