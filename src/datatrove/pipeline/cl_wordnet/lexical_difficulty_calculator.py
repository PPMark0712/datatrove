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
from datatrove.utils.logging import logger


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def calc_freq_difficulty(log_freq, freq_scaling_factor=0.8, log_freq_center=math.log(2233306)):
    return 1 - sigmoid(freq_scaling_factor * (log_freq - log_freq_center))


def power_mean(scores: list, alpha=1.5):
    if len(scores) == 0:
        return 0
    return (sum([s ** alpha for s in scores]) / len(scores)) ** (1 / alpha)


def merge_score(scores: list, alpha=1.5, top_p=0.1, top_weight=0.7):
    """merge list of one word difficulty into paragraph difficulty"""
    scores.sort(reverse=True)
    top_k = int(len(scores) * top_p)
    top_k_scores = scores[:top_k]
    other_scores = scores[top_k:]
    return power_mean(top_k_scores, alpha) * top_weight + power_mean(other_scores, alpha) * (1 - top_weight)


class LexicalDifficultyCalculator(PipelineStep):
    name = "Lexical Difficulty Calculator"
    type = "Curriculum Learning"

    def __init__(
        self,
        output_folder: DataFolderLike,
        freq_scaling_factor=0.8,
        freq_center_rate=0.001,
        power_mean_alpha=1.5,
        merge_top_p=0.1,
        merge_top_weight=0.6,
        noun_weight=0.6,
        dis_to_basic_path = "/data1/yyz/projects/CurriculumLearning/build_dict/output/dict/dis_to_basic.txt",
        word_freq_path = "/data1/yyz/projects/CurriculumLearning/build_dict/output/dict/word_count.txt",
        **kwargs
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.kwargs = kwargs
        self._check_nltk_dependencies()

        logger.info("building dict")
        self.stop_words = set(stopwords.words("english"))

        # logger.info("loading dis to basic")
        self.dis_to_difficulty = {i: math.log(i + 1) / math.log(11) for i in range(11)}
        self.dis_to_basic = {}

        with open(dis_to_basic_path, "r") as f:
            for line in f:
                synset_name, dis = line.strip().split(" ")
                self.dis_to_basic[synset_name] = int(dis)

        # logger.info("loading word freq")
        self.word_log_freq = {}

        with open(word_freq_path, "r") as f:
            for line in f:
                word, freq = line.strip().split(" ")
                self.word_log_freq[word] = math.log(int(freq))

        self.freq_scaling_factor = freq_scaling_factor
        self.power_mean_alpha = power_mean_alpha
        self.merge_top_p = merge_top_p
        self.merge_top_weight = merge_top_weight
        self.noun_weight = noun_weight

        # logger.info("calculating freq center")
        total_words = len(self.word_log_freq)
        freqs = list(self.word_log_freq.values())  # already sorted (descending)
        sigmoid_center_id = int(total_words * freq_center_rate)
        self.log_freq_center = freqs[sigmoid_center_id]
        logger.info("initialized")

    def _check_nltk_dependencies(self):
        if "nltk_path" in self.kwargs:
            nltk.data.path.append(self.kwargs["nltk_path"])
        # return
        nltk_dependencies = [
            "wordnet",
            "stopwords",
            "punkt_tab",
            "averaged_perceptron_tagger_eng",
        ]
        for package in nltk_dependencies:
            # logger.debug(f"checking {package}")
            nltk.download(package, download_dir=self.kwargs.get("nltk_path", None))

    def is_valid_word(self, word: str) -> bool:
        if len(word) <= 1:
            return False
        if word in self.stop_words:
            return False
        if not word.isalpha():
            return False
        return True

    def calc_score(self, text: list) -> dict:
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
                # context = words[max(0, i - window_r): min(len(words), i + window_r + 1)]
                # synset = lesk(context, word, pos=wn.NOUN)
                synsets = wn.synsets(word, pos=wn.NOUN)
                synset = synsets[0] if len(synsets) else None
                # logger.debug(f"{word}, {pos}, {synset}")
                if synset and synset.name() in self.dis_to_basic:
                    concept_dis = self.dis_to_basic[synset.name()]
                else:
                    concept_dis = 3
                concept_difficulty = self.dis_to_difficulty[concept_dis]
                freq_difficulty = calc_freq_difficulty(self.word_log_freq.get(word.lower(), 0))
                score = (concept_difficulty * freq_difficulty) ** 0.5
                # score = freq_difficulty
                noun_scores.append((score, word))
            else:
                freq_difficulty = calc_freq_difficulty(self.word_log_freq.get(word.lower(), 0))
                score = freq_difficulty
                non_noun_scores.append((score, word))
        return noun_scores, non_noun_scores

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if "nltk_path" in self.kwargs:
            nltk.data.path.append(self.kwargs["nltk_path"])
        difficulty_list = []
        with self.track_time():
            for i, doc in enumerate(data):
                if (i + 1) % 1000 == 0:
                    logger.debug(f"processed {i + 1} docs")
                noun_scores_with_words, non_noun_scores_with_words = self.calc_score(doc.text)
                noun_scores_with_words.sort(key=lambda x: x[0], reverse=True)
                non_noun_scores_with_words.sort(key=lambda x: x[0], reverse=True)
                # logger.debug(noun_scores_with_words)
                # logger.debug(non_noun_scores_with_words)

                noun_scores = [score for score, word in noun_scores_with_words]
                non_noun_scores = [score for score, word in non_noun_scores_with_words]
                noun_difficulty = merge_score(noun_scores)
                non_noun_difficulty = merge_score(non_noun_scores)
                difficulty = self.noun_weight * noun_difficulty + (1 - self.noun_weight) * non_noun_difficulty
                difficulty_list.append(difficulty)
        with self.output_folder.open(f"{rank:05d}.json", mode="w") as f:
            json.dump(difficulty_list, f)


class WeightSorter(PipelineStep):
    name = "Weight Sorter"
    type = "Curriculum Learning"

    def __init__(
        self,
        difficulty_folder: DataFolderLike,
    ):
        super().__init__()
        self.difficulty_folder = get_datafolder(difficulty_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            all_docs = []
            for doc in data:
                all_docs.append(doc)
            with self.difficulty_folder.open(f"{rank:05d}.json", "r") as f:
                difficulty_list = json.load(f)
            idxs = [i for i in range(len(difficulty_list))]
            idxs.sort(key=lambda x: difficulty_list[x])
            # logger.debug(idxs)
            for idx in idxs:
                all_docs[idx].metadata["lexical_difficulty"] = difficulty_list[idx]
                yield all_docs[idx]
