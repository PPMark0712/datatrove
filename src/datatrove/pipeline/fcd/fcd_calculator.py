import json
import math
import os

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
from datatrove.utils.logging import logger


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calc_freq_difficulty(log_freq, freq_scaling_factor, log_freq_center):
    return 1 - sigmoid(freq_scaling_factor * (log_freq - log_freq_center))


def power_mean(scores: list, alpha=1.5):
    if len(scores) == 0:
        return 0
    return (sum([s ** alpha for s in scores]) / len(scores)) ** (1 / alpha)


def agg_scores(scores: list, alpha=1.5, top_quantile=0.9, top_weight=0.7):
    """merge list of one word difficulty into paragraph difficulty"""
    if len(scores) <= 20:
        return power_mean(scores, alpha)
    scores.sort()
    split_id = int(len(scores) * top_quantile)
    top_scores = scores[split_id:]
    other_scores = scores[:split_id]
    return power_mean(top_scores, alpha) * top_weight + power_mean(other_scores, alpha) * (1 - top_weight)


class FcdCalculator(PipelineStep):
    name = "Frequency-Concept Difficulty Calculator"
    type = "FCD"

    def __init__(
        self,
        output_folder: DataFolderLike,
        freq_scaling_factor=0.7,
        log_freq_center=None,  # math.log(36597166),
        log_freq_quantile=0.1,
        basic_words_path=os.path.join(os.path.dirname(__file__), "build_dict", "data", "basic_words.txt"),
        dis_to_basic_path=os.path.join(os.path.dirname(__file__), "build_dict", "data", "dis_to_basic.txt"),
        word_freq_path=os.path.join(os.path.dirname(__file__), "build_dict", "data", "word_freq.txt"),
        w_f=0.5,
        power_mean_alpha=1.5,
        agg_top_quantile=0.9,
        agg_top_weight=0.7,
        noun_weight=0.7,
        **kwargs
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.kwargs = kwargs
        self.w_f = w_f
        self.basic_words_path = basic_words_path
        self.dis_to_basic_path = dis_to_basic_path
        self.word_freq_path = word_freq_path
        self.freq_scaling_factor = freq_scaling_factor
        self.power_mean_alpha = power_mean_alpha
        self.agg_top_quantile = agg_top_quantile
        self.agg_top_weight = agg_top_weight
        self.noun_weight = noun_weight
        self.log_freq_center = log_freq_center
        self.log_freq_quantile = log_freq_quantile

    def init_dict(self):
        logger.info("building dict")
        if self.kwargs.get("nltk_path") is not None:
            nltk.data.path.append(self.kwargs["nltk_path"])
        self.stop_words = set(stopwords.words("english"))

        logger.info("loading dis_to_basic")
        self.dis_to_basic = {}
        with open(self.dis_to_basic_path, "r") as f:
            for line in f:
                synset_name, dis = line.strip().split(" ")
                self.dis_to_basic[synset_name] = int(dis)
        max_dis = max(self.dis_to_basic.values())
        self.dis_to_difficulty = {i: math.log(i + 1) / math.log(max_dis + 1) for i in range(max_dis + 1)}

        logger.info("loading word_freq")
        self.word_log_freq = {}
        with open(self.word_freq_path, "r") as f:
            for line in f:
                word, freq = line.strip().split(" ")
                self.word_log_freq[word] = math.log(int(freq))

        logger.info("calculating log_freq center")
        if self.log_freq_center is None:
            with open(self.basic_words_path, "r") as f:
                basic_words = [line.strip() for line in f.readlines()]
            basic_log_freqs = sorted(self.word_log_freq[word] for word in basic_words)
            self.log_freq_center = basic_log_freqs[int(len(basic_log_freqs) * self.log_freq_quantile)]

    def is_valid_word(self, word: str) -> bool:
        if len(word) <= 1:
            return False
        if word in self.stop_words:
            return False
        if not word.isalpha():
            return False
        if word.lower() not in self.word_log_freq:
            return False
        return True

    def calc_score(self, text: str) -> dict:
        words = word_tokenize(text)
        words_with_pos = pos_tag(words)
        # logger.debug(words_with_pos)
        noun_scores = []
        non_noun_scores = []
        # window_r = 10
        for i, (word, pos) in enumerate(words_with_pos):
            if not self.is_valid_word(word):
                continue
            pos = pos[0]
            if pos == "N":
                # Considering that the word sense disambiguation accuracy fails to meet expectations, we adopt the default sense difficulty of the vocabulary.

                # context = words[max(0, i - window_r): min(len(words), i + window_r + 1)]
                # synset = lesk(context, word, pos=wn.NOUN)
                synsets = wn.synsets(word, pos=wn.NOUN)
                synset = synsets[0] if len(synsets) else None
                # logger.debug(f"{word}, {pos}, {synset}")

                if synset and synset.name() in self.dis_to_basic:
                    concept_dis = self.dis_to_basic[synset.name()]
                else:
                    concept_dis = 2  # unknown word's default semantic distance
                concept_difficulty = self.dis_to_difficulty[concept_dis]
                freq_difficulty = calc_freq_difficulty(self.word_log_freq.get(word.lower(), 0), self.freq_scaling_factor, self.log_freq_center)
                score = (freq_difficulty ** self.w_f) * (concept_difficulty ** (1 - self.w_f))
                noun_scores.append((score, word))
            else:
                freq_difficulty = calc_freq_difficulty(self.word_log_freq.get(word.lower(), 0), self.freq_scaling_factor, self.log_freq_center)
                score = freq_difficulty
                non_noun_scores.append((score, word))
        return noun_scores, non_noun_scores

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        self.init_dict()
        with self.track_time():
            difficulty_list = []
            for i, doc in enumerate(data, 1):
                if i % 10000 == 0:
                    logger.debug(f"processed {i} docs")
                noun_scores_with_words, non_noun_scores_with_words = self.calc_score(doc.text)
                noun_scores_with_words.sort(key=lambda x: x[0], reverse=True)
                non_noun_scores_with_words.sort(key=lambda x: x[0], reverse=True)
                # logger.debug(noun_scores_with_words)
                # logger.debug(non_noun_scores_with_words)

                noun_scores = [score for score, word in noun_scores_with_words]
                non_noun_scores = [score for score, word in non_noun_scores_with_words]
                noun_difficulty = agg_scores(noun_scores, self.power_mean_alpha, self.agg_top_quantile, self.agg_top_weight)
                non_noun_difficulty = agg_scores(non_noun_scores, self.power_mean_alpha, self.agg_top_quantile, self.agg_top_weight)
                difficulty = self.noun_weight * noun_difficulty + (1 - self.noun_weight) * non_noun_difficulty
                difficulty_list.append(difficulty)
            with self.output_folder.open(f"{rank:05d}.json", mode="w") as f:
                json.dump(difficulty_list, f)
