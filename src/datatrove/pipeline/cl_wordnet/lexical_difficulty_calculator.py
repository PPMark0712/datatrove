import os
import json
import math

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.wsd import lesk

# import wn
# from pywsd.utils import lemmatize
# from pywsd.similarity import sim

from datatrove.io import DataFolderLike, get_datafolder
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.text import split_into_sentences
from datatrove.utils.logging import logger

def get_wordnet_pos(nltk_pos_tag:str):
    d = {
        "J": wn.ADJ,
        "V": wn.VERB,
        "N": wn.NOUN,
        "R": wn.ADV
    }
    return d.get(nltk_pos_tag[0], None)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calc_freq_difficulty(log_freq, freq_scaling_factor=0.8, log_freq_center=10):
    return 1 - sigmoid(freq_scaling_factor * (log_freq - log_freq_center))


class LexicalDifficultyCalculator(PipelineStep):
    name = "Lexical Difficulty Calculator"
    type = "Curriculum Learning"

    def __init__(
        self,
        output_folder: DataFolderLike,
        min_max_output_folder: DataFolderLike,
        **kwargs
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.min_max_output_folder = get_datafolder(min_max_output_folder)
        self.kwargs = kwargs
        self._check_nltk_dependencies()
        self.stop_words = set(stopwords.words("english"))
        
        self.dis_to_difficlty = {
            0: 0.0,
            1: 0.3,
            2: 0.5,
            3: 0.7,
            4: 0.8,
            5: 0.9
        }
        self.dis_to_basic = {}
        dis_to_basic_path = ""
        with open(dis_to_basic_path, "r") as f:
            for line in f:
                synset_name, dis = line.strip().split(" ")
                self.dis_to_basic[synset_name] = int(dis)
        
        self.word_log_freq = {}
        word_freq_path = ""
        with open(word_freq_path, "r") as f:
            for line in f:
                word, freq = line.strip().split(" ")
                self.word_log_freq[word] = math.log(int(freq))

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
        if len(word) <= 1:
            return False
        if word in self.stop_words:
            return False
        alpha_rate = sum(1 for c in word if c.isalpha()) / len(word)
        if alpha_rate < 0.5:
            return False
        return True

    def merge_score(self, scores: list, alpha=1.5, top_p=0.1, top_weight=0.7):
        """merge list of one word difficulty into paragraph difficulty"""
        def power_mean(scores: list, alpha=1.5):
            return sum([s ** alpha for s in scores]) ** (1 / alpha)
        scores.sort(reverse=True)
        top_k = int(len(scores) * top_p)
        top_k_scores = scores[:top_k]
        other_scores = scores[top_k:]
        return power_mean(top_k_scores, alpha) * top_weight + power_mean(other_scores, alpha) * (1 - top_weight)

    def calc_score(self, text: list) -> dict:
        words = word_tokenize(text)
        words_with_pos = pos_tag(words)
        noun_scores = []
        non_noun_scores = []
        window_r = 10
        for i, (word, pos) in enumerate(words_with_pos):
            if not self.is_valid_word(word):
                continue
            pos = pos[0]
            if pos == "N":
                context = words[max(0, i - window_r): min(len(words), i + window_r + 1)]
                synset = lesk(context, word, pos=wn.NOUN)
                # logger.debug(f"{word}, {pos}, {synset}")
                if synset:
                    concept_dis = self.dis_to_basic[synset.name()]
                else:
                    concept_dis = 10
                concept_difficulty = self.dis_to_diff.get(concept_dis, 1)
                freq_difficulty = calc_freq_difficulty(self.word_log_freq.get(word, 0))
                noun_scores.append(concept_difficulty * freq_difficulty)
            else:
                freq_difficulty = calc_freq_difficulty(self.word_log_freq.get(word, 0))
                non_noun_scores.append(freq_difficulty)
        return noun_scores, non_noun_scores

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if "nltk_path" in self.kwargs:
            nltk.data.path.append(self.kwargs["nltk_path"])
        with self.track_time(), self.output_folder.open(f"{rank:05d}.jsonl", mode="w") as f:
            for doc in data:
                noun_scores, non_noun_scores = self.calc_score(doc.text)


class PosHypernymDepthNormalizer(PipelineStep):
    name = "Pos hypernym depth normalizer"
    type = "Curriculum Learning"

    def __init__(
        self,
        input_folder: DataFolderLike,
        min_max_folder: DataFolderLike,
        output_folder: DataFolderLike
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.min_max_folder = get_datafolder(min_max_folder)
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            min_max_dict = {pos: (100, 0.0) for pos in poss_to_calc}
            # gather min max
            for i in range(world_size):
                with self.min_max_folder.open(f"{i:05d}.json", mode="r") as f:
                    rank_min_max_dict = json.load(f)
                for pos, (rank_min_v, rank_max_v) in rank_min_max_dict.items():
                    min_v, max_v = min_max_dict[pos]
                    min_max_dict[pos] = (min(min_v, rank_min_v), max(max_v, rank_max_v))
            # normalize
            with self.input_folder.open(f"{rank:05d}.jsonl", mode="r") as input_fn, self.output_folder.open(f"{rank:05d}.jsonl", mode="w") as output_fn:
                for line in input_fn:
                    pos_hypernym_depth_dict = json.loads(line)
                    for pos, depth in pos_hypernym_depth_dict.items():
                        min_v, max_v = min_max_dict[pos]
                        if min_v == max_v:
                            pos_hypernym_depth_dict[pos] = 0
                        else:    
                            pos_hypernym_depth_dict[pos] = (depth - min_v) / (max_v - min_v)
                    output_fn.write(json.dumps(pos_hypernym_depth_dict) + "\n")


class WeightSorter(PipelineStep):
    name = "Weight Sorter"
    type = "Curriculum Learning"

    def __init__(
        self,
        normalized_hypernym_depth_folder: DataFolderLike,
        pos_weight: dict = {"N": 0.6, "V": 0.4}
    ):
        super().__init__()
        self.normalized_hypernym_depth_folder = get_datafolder(normalized_hypernym_depth_folder)
        self.pos_weight = pos_weight

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        all_docs = []
        def weight_iter():
            with self.normalized_hypernym_depth_folder.open(f"{rank:05d}.jsonl", mode="r") as f:
                for line in f:
                    hypernym_depth_dict = json.loads(line)
                    yield hypernym_depth_dict, sum([hypernym_depth_dict[pos] * self.pos_weight[pos] for pos in poss_to_calc])
        for doc, (hypernym_depth_dict, weight) in zip(data, weight_iter()):
            doc.metadata["hypernym_depth"] = hypernym_depth_dict
            doc.metadata["weight"] = weight
            all_docs.append(doc)
        all_docs.sort(key=lambda x: x.metadata["weight"])
        for doc in all_docs:
            yield doc
