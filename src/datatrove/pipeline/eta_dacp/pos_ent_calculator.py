import math
import json
from abc import ABC, abstractmethod
from collections import Counter
import nltk
import jieba.posseg
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.utils.logging import logger


def is_alpha_word(word: str) -> bool:
    return any(c.isalpha() for c in word)


class BasePartOfSpeechPredictor(ABC):
    @abstractmethod
    def predict(self, text: str):
        pass


class EnglishPartOfSpeechPredictor(BasePartOfSpeechPredictor):
    def __init__(self, **kwargs):
        super().__init__()
        if "nltk_data_path" in kwargs:
            nltk.data.path.append(kwargs["nltk_data_path"])

    def predict(self, text: str):
        tokens = nltk.word_tokenize(text)
        words, pos_tags = [], []
        for word, pos in nltk.pos_tag(tokens):
            if not is_alpha_word(word):
                continue
            words.append(word)
            pos_tags.append(pos)
        return words, pos_tags


class ChinesePartOfSpeechPredictor(BasePartOfSpeechPredictor):
    def __init__(self, **kwargs):
        super().__init__()

    def predict(self, text: str):
        words, pos_tags = [], []
        for word, pos in jieba.posseg.cut(text):
            if not is_alpha_word(word):
                continue
            words.append(word)
            pos_tags.append(pos)
        return words, pos_tags


class PartOfSpeechPredictor(BasePartOfSpeechPredictor):
    def __init__(self, language: str, **kwargs):
        super().__init__()
        if language == "en":
            self.predictor = EnglishPartOfSpeechPredictor(**kwargs)
        elif language == "zh":
            self.predictor = ChinesePartOfSpeechPredictor(**kwargs)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def predict(self, text: str):
        return self.predictor.predict(text)


def calc_counter_entropy(counter: Counter):
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((count / total) * math.log2(count / total) for count in counter.values())


class PosEntCalculator(PipelineStep):
    name = "Part of Speech Entropy Calculator"
    type = "ETA-DACP"

    def __init__(
        self,
        language: str,
        output_folder: DataFolderLike,
        **kwargs
    ):
        super().__init__()
        self.output_folder = get_datafolder(output_folder)
        self.language = language
        self.kwargs = kwargs

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            pos_predictor = PartOfSpeechPredictor(self.language, **self.kwargs)
            pos_ent_data = []
            for doc in data:
                words, pos_tags = pos_predictor.predict(doc.text)
                pos_ent_data.append(calc_counter_entropy(Counter(pos_tags)))
            with self.output_folder.open(f"{rank:05d}.json", mode="w") as f:
                json.dump(pos_ent_data, f)