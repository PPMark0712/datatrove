from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.cl_wordnet.word_extractor import WordExtractor
from datatrove.utils.logging import logger

class DictBuilder(PipelineStep):
    name = "🔤 - Dict builder"
    type = "cl"

    def __init__(
        self,
        language: str,
        output_folder: DataFolderLike,
        **kwargs
    ):
        super().__init__()
        self.language = language
        self.output_folder = get_datafolder(output_folder)
        self.kwargs = kwargs

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            words = []
            word_extractor = WordExtractor(self.language, **self.kwargs)
            for doc in data:
                current_words = word_extractor.extract_words(doc.text)
                words.extend(current_words)
                # logger.info(f"text: {doc.text}")
                # logger.info(f"words: {current_words}")
            words = list(set(words))
            words.sort()
            output_file = self.output_folder.open(f"{rank:05d}.txt", mode="w")
            for word in words:
                output_file.write(word + "\n")
            output_file.close()


class DictMerger(PipelineStep):
    name = "🔤 - Dict merger"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        with self.track_time():
            words = []
            file_list = self.input_folder.list_files()
            for file_path in file_list:
                input_file = self.input_folder.open(file_path, mode="r")
                for line in input_file:
                    words.append(line.strip())
                input_file.close()
            words = list(set(words))
            words.sort()
            output_file = self.output_folder.open(f"merged_dict.txt", mode="w")
            for word in words:
                output_file.write(word + "\n")
            output_file.close()