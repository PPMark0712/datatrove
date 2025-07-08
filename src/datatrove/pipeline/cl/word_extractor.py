from abc import ABC, abstractmethod

import jieba
from ltp import LTP

from datatrove.utils.logging import logger


class BaseWordExtractor(ABC):
    @abstractmethod
    def extract_words(self, text: str) -> list[str]:
        pass


class ChineseWordExtractor(BaseWordExtractor):
    def __init__(self, **kwargs):
        super().__init__()
        jieba.initialize()
        self.jieba_words = set(jieba.dt.FREQ.keys())

        # use_ltp_clean: use ltp model to clean the error spaces, then use jieba to tokenize
        self.use_ltp_clean = kwargs.get("use_ltp_clean", False)
        if self.use_ltp_clean:
            self.ltp = LTP(kwargs.get("ltp_model_path", "LTP/small"))

    def split_into_sentences(self, text: str) -> list[str]:
        sentences = []
        i = 0
        n = len(text)
        while i < n:
            end = min(i + 300, n)
            chunk = text[i:end]
            split_pos = -1
            for j in range(len(chunk) - 1, -1, -1):
                if chunk[j] in "。！？：:；;，,|":
                    split_pos = j
                    break
            if split_pos != -1:
                sentences.append(chunk[:split_pos + 1])
                i += split_pos + 1
            else:
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
        if self.use_ltp_clean:
            sentences = self.split_into_sentences(text)
            outputs = self.ltp.pipeline(sentences, tasks=["cws"])
            results = []
            for words in outputs.cws:
                words = [word.strip() for word in words]
                results.extend(words)
            cleaned_text = ""
            p = 0
            for word in results:
                while text[p:p + len(word)] != word:
                    cleaned_text += text[p]
                    p += 1
                cleaned_text += word.replace(" ", "")
                p += len(word)
            text = cleaned_text
        words = jieba.lcut(text)
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