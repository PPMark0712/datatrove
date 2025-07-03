from abc import ABC, abstractmethod

import jieba
from ltp import LTP


class BaseWordExtractor(ABC):
    @abstractmethod
    def extract_words(self, text: str) -> list[str]:
        pass


class ChineseWordExtractor(BaseWordExtractor):
    def __init__(self, **kwargs):
        super().__init__()
        jieba.initialize()
        self.jieba_words = set(jieba.dt.FREQ.keys())
        self.ltp = LTP(kwargs.get("ltp_model_path", "LTP/small"))
    
    def clean_word(self, word: str) -> str:
        return word.replace(" ", "")

    def is_valid_word(self, word: str) -> bool:
        if not word:
            return False
        if not any(c.isalpha() for c in word):
            return False
        if len(word) > 7:
            return False
        if word not in self.jieba_words:
            return False
        return True

    def extract_words(self, text: str) -> list[str]:
        words = self.ltp.pipeline(text, tasks=["cws"]).cws
        words = [self.clean_word(word) for word in words]
        words = [word for word in words if self.is_valid_word(word)]
        return words


class WordExtractor(BaseWordExtractor):
    def __init__(self, language: str, **kwargs):
        super().__init__()
        self.language = language
        if language == "zh":
            self.word_extractor = ChineseWordExtractor(**kwargs)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def extract_words(self, text: str) -> list[str]:
        return self.word_extractor.extract_words(text)