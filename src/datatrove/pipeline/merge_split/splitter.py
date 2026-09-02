import json
from typing import Callable

from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.base import PipelineStep


class FileSplitter(PipelineStep):
    name = "FileSplitter"
    type = "SPLIT"

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_file_count: int = None,
        max_rows_per_file: int = None,
        adapter: Callable = None,
    ):
        super().__init__()
        if (output_file_count is None) == (max_rows_per_file is None):
            raise ValueError("Exactly one of `output_file_count` or `max_rows_per_file` must be provided")
        if output_file_count is not None and output_file_count <= 0:
            raise ValueError("`output_file_count` must be > 0")
        if max_rows_per_file is not None and max_rows_per_file <= 0:
            raise ValueError("`max_rows_per_file` must be > 0")
        self.output_folder = get_datafolder(output_folder)
        self.output_file_count = output_file_count
        self.max_rows_per_file = max_rows_per_file
        self.adapter = adapter or self._default_adapter

    def _default_adapter(self, doc: Document) -> dict:
        result = {"text": doc.text, "id": doc.id}
        if doc.metadata:
            result["metadata"] = doc.metadata
        return result

    def _write_line(self, doc: Document, output_file):
        with self.track_time():
            line = json.dumps(self.adapter(doc), ensure_ascii=False) + "\n"
            output_file.write(line.encode("utf-8"))

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        if world_size <= 0:
            raise ValueError("`world_size` must be > 0")
        if not (0 <= rank < world_size):
            raise ValueError("`rank` must satisfy 0 <= rank < world_size")

        if self.output_file_count is not None:
            yield from self._run_by_file_count(data, rank, world_size)
        else:
            yield from self._run_by_max_rows(data, rank, world_size)

    def _run_by_file_count(self, data: DocumentsPipeline, rank: int, world_size: int):
        base_count = self.output_file_count // world_size
        remainder_count = self.output_file_count % world_size
        rank_file_count = base_count + (1 if rank < remainder_count else 0)
        if rank_file_count <= 0:
            raise ValueError(
                f"No output files assigned to rank={rank} with world_size={world_size} and "
                f"output_file_count={self.output_file_count}."
            )
        start_idx = base_count * rank + min(rank, remainder_count)
        output_filenames = [f"{file_idx:05d}.jsonl" for file_idx in range(start_idx, start_idx + rank_file_count)]
        for output_filename in output_filenames:
            self.output_folder.open(output_filename, "ab").close()
        output_files = {}
        try:
            for index, doc in enumerate(data):
                output_filename = output_filenames[index % rank_file_count]
                if output_filename not in output_files:
                    output_files[output_filename] = self.output_folder.open(output_filename, "wb")
                self._write_line(doc, output_files[output_filename])
                self.stat_update(output_filename)
                yield doc
        finally:
            for f in output_files.values():
                f.close()

    def _run_by_max_rows(self, data: DocumentsPipeline, rank: int, world_size: int):
        file_idx = rank
        rows_in_current = 0
        output_file = None
        output_filename = None
        try:
            for doc in data:
                if output_file is None or rows_in_current >= self.max_rows_per_file:
                    if output_file is not None:
                        output_file.close()
                        self.stat_update(output_filename)
                    output_filename = f"{file_idx:05d}.jsonl"
                    output_file = self.output_folder.open(output_filename, "wb")
                    rows_in_current = 0
                    file_idx += world_size
                self._write_line(doc, output_file)
                rows_in_current += 1
                yield doc
        finally:
            if output_file is not None:
                output_file.close()
                self.stat_update(output_filename)
