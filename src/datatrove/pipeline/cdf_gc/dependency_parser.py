import os
from abc import ABC, abstractmethod
from ltp import LTP
from datatrove.utils.logging import logger

def split_into_sentences(text: str, max_length: int, punctuations: str):
    # 从头开始，每当遇到punctuations时，切分一个句子，如果到达max_length也要切分
    sentences = []
    if not text:
        return []
    start = 0
    i = 0
    text_len = len(text)
    while i < text_len:
        if text[i] in punctuations or i - start + 1 >= max_length:
            sentence = text[start:i + 1]
            if sentence.strip():
                sentences.append(sentence.strip())
            start = i + 1
        i += 1
    if start < text_len:
        sentence = text[start:]
        if sentence.strip():
            sentences.append(sentence.strip())
    return sentences


class BaseDependencyParser(ABC):
    @abstractmethod
    def predict(self, text: str, rank: int):
        pass

# class EnglishSyntacticComplexityPredictor(BaseSyntacticComplexityCalculator):


class ChineseDependencyParser(BaseDependencyParser):
    _requires_dependencies = ["ltp"]
    
    def __init__(self, gpu_id: int, **kwargs):
        super().__init__()
        if "ltp_model_path" in kwargs:
            ltp_model_path = kwargs["ltp_model_path"]
        else:
            ltp_model_path = "LTP/small"
        self.ltp = LTP(ltp_model_path).to(f"cuda:{gpu_id}")
        self.batch_size = kwargs.get("batch_size", 32)
        self.max_length = kwargs.get("max_length", 128)

    def predict(self, text: str):
        """
        Returns:
            List[Dict[str, Any]]: List of dictionaries, each containing:
                - "words": List[str] - Words in the sentence
                - "dep_labels": List[str] - Dependency labels for each word
                - "parents": List[int] - Parent indices for each word
        """
        chinese_eos_puncts = "。！!？?；;：:\n\t—…"
        sentences = split_into_sentences(text, punctuations=chinese_eos_puncts, max_length=self.max_length)
        parsed_sentences = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            outputs = self.ltp.pipeline(batch, tasks=["cws", "dep"])
            for cws, dep in zip(outputs.cws, outputs.dep):
                parents = dep["head"]
                for i in range(len(parents)):
                    parents[i] -= 1  # LTP use 1-based index
                parsed_sentences.append({
                    "words": cws,
                    "dep_labels": dep["label"],
                    "parents": parents,
                })
        return parsed_sentences
    

class DependencyParser(BaseDependencyParser):
    name = "Dependency Parser"
    _requires_dependencies = ["ltp"]
    
    def __init__(self, language: str, gpu_id: int, **kwargs):
        super().__init__()
        self.language = language
        if language == "zh":
            self.dependency_parser = ChineseDependencyParser(gpu_id, **kwargs)
        else:
            raise ValueError(f"Unsupported language: {language}")
    
    def predict(self, text: str):
        return self.dependency_parser.predict(text)