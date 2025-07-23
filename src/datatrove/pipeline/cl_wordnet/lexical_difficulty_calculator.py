import json
import math
from collections import defaultdict

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.wsd import lesk

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
    if nltk_pos_tag.startswith('J'):
        return wn.ADJ
    elif nltk_pos_tag.startswith('V'):
        return wn.VERB
    elif nltk_pos_tag.startswith('N'):
        return wn.NOUN
    elif nltk_pos_tag.startswith('R'):
        return wn.ADV
    else:
        return None


class LexicalDifficultySorter(PipelineStep):
    name = "🔤 - Lexical difficulty sorter"
    type = "cl"

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.kwargs = kwargs
        nltk_dependencies = [
            "stopwords"
        ]
        if "nltk_path" in kwargs:
            nltk.data.path.append(kwargs["nltk_path"])
        for package in nltk_dependencies:
            try:
                nltk.data.load(package)
            except:
                nltk.download(package, download_dir=kwargs.get("nltk_path", None))
        self.stop_words = stopwords.words("english")
    
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
        if "nltk_path" in self.kwargs:
            nltk.data.path.append(self.kwargs["nltk_path"])
        with self.track_time():
            difficulty_data = []
            all_docs = []
            # 需要用all_docs保存完整的文件，占用大量内存
            for doc in data:
                all_docs.append(doc)
                text = doc.text
                sentences = sent_tokenize(text)
                total_words_counted = 0
                total_lesked_words_counted = 0
                sum_hypernym_depth = 0
                sum_synonym_count = 0
                abstract_word_count = 0
                hypernym_counter = defaultdict(int)

                for sentence in sentences:
                    words = word_tokenize(sentence)
                    words_with_pos = pos_tag(words)
                    for word, pos in words_with_pos:
                        if not self.is_valid_word(word):
                            continue
                        total_words_counted += 1

                        synsets = wn.synsets(word)
                        sum_synonym_count += len(synsets)
                        wn_pos = get_wordnet_pos(pos)
                        synset = lesk(words, word, pos=wn_pos)
                        if synset is not None:
                            total_lesked_words_counted += 1
                            path = synset.hypernyms()
                            # logger.info(f"{word}, {path}, {synset}")
                            sum_hypernym_depth += len(path)
                            if len(path) <= 2:
                                abstract_word_count += 1
                            for hypernym in path[-3:]:
                                hypernym_counter[hypernym.name()] += 1                            
                        else:
                            # logger.info("lesk error")
                            pass

                # if total_words_counted == 0:
                #     pass
                avg_hypernym_depth = sum_hypernym_depth / total_words_counted
                avg_synonym_count = sum_synonym_count / total_words_counted
                hypernym_entropy = calc_counter_entropy(hypernym_counter)
                abstract_word_ratio = abstract_word_count / total_lesked_words_counted
                difficulty_data.append({
                    "avg_hypernym_depth": avg_hypernym_depth,
                    "synonym_richness": avg_synonym_count,
                    "sense_dispersion": hypernym_entropy,
                    # "non_abstract_word_rate": 1 - abstract_word_ratio
                })

            normalized_difficulty_data = self.calc_scores(difficulty_data)
            for doc, normalized_difficulty, org_difficulty in zip(all_docs, normalized_difficulty_data, difficulty_data):
                doc.metadata["difficulty"] = org_difficulty
                doc.metadata["normalized_difficulty"] = normalized_difficulty
            all_docs.sort(key=lambda x: x.metadata["normalized_difficulty"]["difficulty_score"])
            for doc in all_docs:
                yield doc
