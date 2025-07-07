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

    def is_chinese_punct(self, ch: str) -> bool:
        return ch in "。！？"

    def split_into_sentences(self, text: str) -> list[str]:
        sentences = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + 300, n)
            chunk = text[i:end]
            # 从右往左找第一个终止中文标点
            split_pos = -1
            for j in range(len(chunk) - 1, -1, -1):
                if self.is_chinese_punct(chunk[j]):
                    split_pos = j
                    break
            if split_pos != -1:
                # 包含标点
                sentences.append(chunk[:split_pos + 1])
                i += split_pos + 1
            else:
                # 没有标点，直接分300个字符
                sentences.append(chunk)
                i += len(chunk)
        return sentences

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
        sentences = self.split_into_sentences(text)
        outputs = self.ltp.pipeline(sentences, tasks=["cws"])
        results = []
        for words in outputs.cws:
            words = [self.clean_word(word) for word in words]
            words = [word for word in words if self.is_valid_word(word)]
            results.extend(words)
        results = list(set(results))
        return results


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