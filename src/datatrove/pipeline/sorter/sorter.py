import dataclasses
import heapq
import json
from typing import Iterator

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.samplers.utils import read_score_file
from datatrove.utils.logging import logger


class DocumentSplitSorter(PipelineStep):
    """Sort documents based on a score file."""
    type = "Sorter"
    name = "Document Sorter"

    def __init__(
        self,
        score_folder: DataFolderLike,
        split_folder: DataFolderLike,
        batch_size_mb: float = 256,
        reverse: bool = False
    ):
        super().__init__()
        self.score_folder = get_datafolder(score_folder)
        self.split_folder = get_datafolder(split_folder)
        self.batch_size_mb = batch_size_mb
        self.reverse = reverse

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.track_time():
            score_data = read_score_file(self.score_folder, rank)
            split_idx = 0
            split_data = []
            split_bytes = 0

            def write_data():
                nonlocal split_data, split_bytes, split_idx
                split_data.sort(key=lambda x: x[1], reverse=self.reverse)
                docs = []
                scores = []
                for doc, score in split_data:
                    docs.append(doc)
                    scores.append(score)
                with self.split_folder.open(f"{rank:05d}/{split_idx:02d}.jsonl", "w") as f:
                    for doc in docs:
                        json_dict = dataclasses.asdict(doc)
                        f.write(json.dumps(json_dict, ensure_ascii=False) + "\n")
                with self.split_folder.open(f"{rank:05d}/{split_idx:02d}.json", "w") as f:
                    json.dump(scores, f, ensure_ascii=False)
                split_idx += 1
                split_data = []
                split_bytes = 0

            for doc, score in zip(data, score_data):
                split_data.append((doc, score))
                split_bytes += len(json.dumps(dataclasses.asdict(doc), ensure_ascii=False).encode("utf-8"))
                if split_bytes >= self.batch_size_mb * 1024 * 1024:
                    write_data()
            if split_data:
                write_data()

            if split_idx == 0:
                logger.warning(f"No split files found for rank {rank} - returning empty pipeline")
                return


class DocumentSplitMerger(PipelineStep):
    """Merge sorted document split files."""
    type = "Sorter"
    name = "Document Split Merger"

    def __init__(
        self,
        split_folder: DataFolderLike,
        output_folder: DataFolderLike,
        output_score_folder: DataFolderLike,
        reverse: bool = False
    ):
        super().__init__()
        self.split_folder = get_datafolder(split_folder)
        self.output_folder = get_datafolder(output_folder)
        self.output_score_folder = get_datafolder(output_score_folder)
        self.reverse = reverse

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        with self.track_time():
            num_splits = len(list(self.split_folder.list_files(f"{rank:05d}"))) // 2
            output_file = self.output_folder.open(f"{rank:05d}.jsonl", "w")
            output_scores = []

            def get_split_iterator(split_id: int) -> Iterator[tuple[float, dict]]:
                with self.split_folder.open(f"{rank:05d}/{split_id:02d}.json", "r", encoding="utf-8") as score_f, \
                     self.split_folder.open(f"{rank:05d}/{split_id:02d}.jsonl", "r", encoding="utf-8") as doc_f:
                    scores = json.load(score_f)
                    for score, doc_line in zip(scores, doc_f):
                        yield (score, json.loads(doc_line))

            heap = []
            for split_id in range(num_splits):
                iter_obj = get_split_iterator(split_id)
                try:
                    first_score, first_doc = next(iter_obj)
                    heap_score = -first_score if self.reverse else first_score
                    heapq.heappush(heap, (heap_score, split_id, first_doc, iter_obj))
                except StopIteration:
                    continue

            while heap:
                heap_score, split_id, current_doc, iter_obj = heapq.heappop(heap)

                # output
                output_file.write(json.dumps(current_doc, ensure_ascii=False) + "\n")
                output_scores.append(-heap_score if self.reverse else heap_score)

                try:
                    next_score, next_doc = next(iter_obj)
                    next_heap_score = -next_score if self.reverse else next_score
                    heapq.heappush(heap, (next_heap_score, split_id, next_doc, iter_obj))
                except StopIteration:
                    continue

            with self.output_score_folder.open(f"{rank:05d}.json", "w") as f:
                json.dump(output_scores, f, ensure_ascii=False)

            logger.info(f"Cleaning up {num_splits} split files for rank {rank}")
            for split_id in range(num_splits):
                try:
                    self.split_folder.rm_file(f"{rank:05d}/{split_id:02d}.jsonl")
                    self.split_folder.rm_file(f"{rank:05d}/{split_id:02d}.json")
                except Exception as e:
                    logger.warning(f"Failed to delete split file {rank:05d}/{split_id:02d}: {str(e)}")
            try:
                self.split_folder.rmdir(f"{rank:05d}")
            except OSError:
                pass
