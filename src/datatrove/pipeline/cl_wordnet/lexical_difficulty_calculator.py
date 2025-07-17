import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.wsd import lesk

from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger

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


class LexicalDifficultyCalculator(PipelineStep):
    name = "🔤 - Lexical difficulty calculator"
    type = "cl"

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
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
        self.stop_words = stopwords
    
    def is_valid_word(self, word: str) -> bool:
        return len(word) > 1 and any(c.isalpha() for c in word) and word not in self.stop_words

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            for doc in data:
                text = doc.text
                sentences = sent_tokenize(text)
                total_words_counted = 0
                sum_hypernym_depth = 0
                total_synsets_count = 0
                abstract_word_count = 0
                for sentence in sentences:
                    words = word_tokenize(sentence)
                    words_with_pos = pos_tag(words)
                    for word, pos in words_with_pos:
                        if not self.is_valid_word(word):
                            continue
                        total_words_counted += 1

                        synsets = wn.synsets(word)
                        total_synsets_count += len(synsets)
                        wn_pos = get_wordnet_pos(pos)
                        synset = lesk(words, word, pos=wn_pos)
                        if synset is not None:
                            path = synset.hypernyms()
                            sum_hypernym_depth += len(path)
                            total_calculated_words += 1



class LexicalDifficultySorter(PipelineStep):
    name = "🔤 - Lexical difficulty sorter"
    type = "cl"

    def __init__(
        self,
        weights: dict[str, float],
    ):
        super().__init__()
        self.weights = weights
        self.stop_words = stop

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            """Warning: All data from a file needs to be read in at once. 
            Please ensure each file size is moderate to avoid excessive memory usage."""
            all_data = []
            scores = []
            for doc in data:
                score = 0
                level_word_counter = doc.metadata["level_word_counter"]
                total_word_cnt = sum(level_word_counter.values())
                for level, level_word_cnt in level_word_counter.items():
                    score += level_word_cnt / total_word_cnt * self.weights[level] if total_word_cnt > 0 else 0
                scores.append(score)
                doc.metadata["score"] = score
                all_data.append(doc)
            idxs = list(range(len(all_data)))
            idxs = sorted(idxs, key=lambda x: scores[x])
            for idx in idxs:
                yield all_data[idx]