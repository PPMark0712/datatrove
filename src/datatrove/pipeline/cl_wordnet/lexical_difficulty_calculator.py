import os
import json

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

poss_to_calc = "NV"  # "JNVR"

def get_wordnet_pos(nltk_pos_tag:str):
    d = {
        "J": wn.ADJ,
        "V": wn.VERB,
        "N": wn.NOUN,
        "R": wn.ADV
    }
    return d.get(nltk_pos_tag[0], None)


class HypernymDepthCalculator(PipelineStep):
    name = "Hypernym depth calculator"
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
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "concreteness_dict.json"), "r") as f:
            self.concrectness_dict = json.load(f)

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
        return len(word) > 1 and any(c.isalpha() for c in word) and word not in self.stop_words

    def calc_doc_hypernym_depth(self, text: str) -> dict:
        pos_hypernym_depth_sum = {pos: 0 for pos in poss_to_calc}
        pos_counter = {pos: 0 for pos in poss_to_calc}
        words = word_tokenize(text)
        words_with_pos = pos_tag(words)
        window_r = 10
        for i, (word, pos) in enumerate(words_with_pos):
            pos = pos[0]
            if pos not in poss_to_calc:
                continue
            context = words[max(0, i - window_r): min(len(words), i + window_r + 1)]
            synset = lesk(context, word, get_wordnet_pos(pos))
            # logger.debug(f"{word}, {pos}, {synset}")
            if synset:
                pos_counter[pos] += 1
                pos_hypernym_depth_sum[pos] += synset.max_depth()
        return {pos: pos_hypernym_depth_sum[pos] / pos_counter[pos] if pos_counter[pos] > 0 else 0 for pos in poss_to_calc}

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        if "nltk_path" in self.kwargs:
            nltk.data.path.append(self.kwargs["nltk_path"])
        with self.track_time():
            min_max_dict = {pos: (100, 0.0) for pos in poss_to_calc}
            with self.output_folder.open(f"{rank:05d}.jsonl", mode="w") as output_fn:
                for doc in data:
                    result = self.calc_doc_hypernym_depth(doc.text)
                    output_fn.write(json.dumps(result) + "\n")
                    for pos, depth in result.items():
                        min_v, max_v = min_max_dict[pos]
                        min_max_dict[pos] = (min(min_v, depth), max(max_v, depth))
            with self.min_max_output_folder.open(f"{rank:05d}.json", mode="w") as output_fn:
                output_fn.write(json.dumps(min_max_dict))


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
