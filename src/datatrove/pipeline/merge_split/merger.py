from datatrove.io import DataFolderLike, get_datafolder
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.base import PipelineStep
from datatrove.utils.logging import logger


class FileMerger(PipelineStep):
    name = "FileMerger"
    type = "MERGE"

    def __init__(
        self,
        input_folder: DataFolderLike,
        output_folder: DataFolderLike,
        output_file_count: int = None,
        rows_per_file: int = None,
        input_glob_pattern: str = "*.jsonl",
    ):
        super().__init__()
        self.input_folder = get_datafolder(input_folder)
        self.output_folder = get_datafolder(output_folder)
        if output_file_count is None and rows_per_file is None:
            raise ValueError("At least one of `output_file_count` or `rows_per_file` must be provided")
        if output_file_count is not None and output_file_count <= 0:
            raise ValueError("`output_file_count` must be > 0")
        if rows_per_file is not None and rows_per_file <= 0:
            raise ValueError("`rows_per_file` must be > 0")
        if output_file_count is not None and rows_per_file is not None:
            logger.warning("Both `output_file_count` and `rows_per_file` are set; `output_file_count` takes precedence")
        self.output_file_count = output_file_count
        self.rows_per_file = rows_per_file
        self.input_glob_pattern = input_glob_pattern

    def run(self, data: DocumentsPipeline = None, rank: int = 0, world_size: int = 1):
        if data is not None:
            raise ValueError("`FileMerger` expects `data=None` and operates on input/output folders only.")
        input_files = self.input_folder.list_files(recursive=True, glob_pattern=self.input_glob_pattern)
        if not input_files:
            yield from ()
            return
        if self.output_file_count is not None:
            yield from self._run_with_output_file_count(input_files, rank, world_size)
            return
        yield from self._run_with_rows_per_file(input_files, rank, world_size)

    def _count_total_lines(self, input_files: list[str]) -> int:
        total_lines = 0
        for input_path in input_files:
            with self.input_folder.open(input_path, "rb") as input_file:
                for _ in input_file:
                    total_lines += 1
        return total_lines

    def _run_with_output_file_count(self, input_files: list[str], rank: int, world_size: int):
        total_lines = self._count_total_lines(input_files)
        if total_lines == 0:
            yield from ()
            return
        base_count = self.output_file_count // world_size
        remainder_count = self.output_file_count % world_size
        rank_output_count = base_count + (1 if rank < remainder_count else 0)
        if rank_output_count <= 0:
            raise ValueError(
                f"No output files assigned to rank={rank} with world_size={world_size} and "
                f"output_file_count={self.output_file_count}."
            )
        start_output_idx = base_count * rank + min(rank, remainder_count)
        end_output_idx = start_output_idx + rank_output_count
        base_lines = total_lines // self.output_file_count
        remainder_lines = total_lines % self.output_file_count
        output_sizes = [base_lines + (1 if output_idx < remainder_lines else 0) for output_idx in range(self.output_file_count)]
        start_line = sum(output_sizes[:start_output_idx])
        end_line = sum(output_sizes[:end_output_idx])
        for output_idx in range(start_output_idx, end_output_idx):
            self.output_folder.open(f"{output_idx:05d}.jsonl", "ab").close()
            if output_sizes[output_idx] == 0:
                with self.output_folder.open(f"{output_idx:05d}.jsonl", "wb"):
                    pass
        if start_line == end_line:
            yield from ()
            return
        current_output_idx = start_output_idx
        while current_output_idx < end_output_idx and output_sizes[current_output_idx] == 0:
            current_output_idx += 1
        remaining_in_current = output_sizes[current_output_idx]
        output_filename = f"{current_output_idx:05d}.jsonl"
        output_file = self.output_folder.open(output_filename, "wb")
        wrote_to_current = False
        input_handlers = [self.input_folder.open(path, "rb") for path in input_files]
        try:
            done = [False] * len(input_handlers)
            remaining_files = len(input_handlers)
            global_line_idx = 0
            stop = False
            while remaining_files > 0 and global_line_idx < end_line and not stop:
                wrote_any = False
                for index, input_file in enumerate(input_handlers):
                    if done[index] or global_line_idx >= end_line:
                        continue
                    line = input_file.readline()
                    if line:
                        wrote_any = True
                        if global_line_idx >= start_line:
                            with self.track_time():
                                output_file.write(line)
                            wrote_to_current = True
                            remaining_in_current -= 1
                            if remaining_in_current == 0:
                                output_file.close()
                                self.stat_update(output_filename)
                                wrote_to_current = False
                                current_output_idx += 1
                                while current_output_idx < end_output_idx and output_sizes[current_output_idx] == 0:
                                    current_output_idx += 1
                                if current_output_idx >= end_output_idx:
                                    stop = True
                                    break
                                remaining_in_current = output_sizes[current_output_idx]
                                output_filename = f"{current_output_idx:05d}.jsonl"
                                output_file = self.output_folder.open(output_filename, "wb")
                        global_line_idx += 1
                        continue
                    done[index] = True
                    remaining_files -= 1
                if not wrote_any:
                    break
        finally:
            output_file.close()
            for input_file in input_handlers:
                input_file.close()
        if wrote_to_current:
            self.stat_update(output_filename)
        yield from ()

    def _run_with_rows_per_file(self, input_files: list[str], rank: int, world_size: int):
        input_handlers = [self.input_folder.open(path, "rb") for path in input_files]
        output_file = None
        output_filename = None
        current_output_idx = None
        try:
            done = [False] * len(input_handlers)
            remaining_files = len(input_handlers)
            global_line_idx = 0
            while remaining_files > 0:
                wrote_any = False
                for index, input_file in enumerate(input_handlers):
                    if done[index]:
                        continue
                    line = input_file.readline()
                    if line:
                        wrote_any = True
                        output_idx = global_line_idx // self.rows_per_file
                        if output_idx % world_size == rank:
                            if current_output_idx != output_idx:
                                if output_file is not None:
                                    output_file.close()
                                    self.stat_update(output_filename)
                                current_output_idx = output_idx
                                output_filename = f"{output_idx:05d}.jsonl"
                                output_file = self.output_folder.open(output_filename, "wb")
                            with self.track_time():
                                output_file.write(line)
                        global_line_idx += 1
                        continue
                    done[index] = True
                    remaining_files -= 1
                if not wrote_any:
                    break
        finally:
            if output_file is not None:
                output_file.close()
                self.stat_update(output_filename)
            for input_file in input_handlers:
                input_file.close()
        yield from ()
