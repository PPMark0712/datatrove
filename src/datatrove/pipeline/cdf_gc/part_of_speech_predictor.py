from abc import ABC, abstractmethod
import nltk
import jieba

class BasePartOfSpeechPredictor(ABC):
    @abstractmethod
    def predict(self, text: str):
        pass
    
    @abstractmethod
    def get_content_words(self, words_with_pos_tags: list[tuple[str, str]]):
        pass

class EnglishPartOfSpeechPredictor(BasePartOfSpeechPredictor):
    def __init__(self, **kwargs):
        super().__init__()
        if "nltk_data_path" in kwargs:
            nltk.data.path.append(kwargs["nltk_data_path"])
        self.content_words_pos_tags = [
            'NN', 'NNS', 'NNP', 'NNPS',  # 名词
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # 动词
            'JJ', 'JJR', 'JJS',  # 形容词
            'RB', 'RBR', 'RBS'  # 副词
        ]

    def predict(self, text: str):
        tokens = nltk.word_tokenize(text)
        word_with_pos_tags = [(word, pos) for word, pos in nltk.pos_tag(tokens)]
        return word_with_pos_tags

    def get_content_words(self, words_with_pos_tags: list[tuple[str, str]]):
        return [word for word, pos in words_with_pos_tags if pos in self.content_words_pos_tags]


class ChinesePartOfSpeechPredictor(BasePartOfSpeechPredictor):
    def __init__(self, **kwargs):
        super().__init__()
        self.content_words_pos_tags = ['n', 'v', 'a', 'm', 'q', 'd', 'b', 'r', 't', 's', 'f', 'an', 'nr', 'nrfg', 'nrt', 'ns', 'nt', 'nz', 'vn']

    def predict(self, text: str):
        words_with_pos_tags = [(word, pos) for word, pos in jieba.posseg.cut(text)]
        return words_with_pos_tags

    def get_content_words(self, words_with_pos_tags: list[tuple[str, str]]):
        return [word for word, pos in words_with_pos_tags if pos in self.content_words_pos_tags]


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