from typing import IO, Callable

from datatrove.io import DataFolderLike
from datatrove.pipeline.writers.disk_base import DiskWriter


class BinWriter(DiskWriter):
    """Write data to datafolder (local or remote) in BIN format

    Args:
        output_folder: a str, tuple or DataFolder where data should be saved
        output_filename: the filename to use when saving data, including extension. Can contain placeholders such as `${rank}` or metadata tags `${tag}`
        adapter: a custom function to "adapt" the Document format to the desired output format
        expand_metadata: save each metadata entry in a different column instead of as a dictionary
    """

    default_output_filename: str = "${rank}.bin"
    name = "🐿 Bin"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        output_folder: DataFolderLike,
        output_filename: str = None,
        adapter: Callable = None,
        expand_metadata: bool = False
    ):
        super().__init__(
            output_folder,
            output_filename=output_filename,
            compression=None,
            adapter=adapter,
            expand_metadata=expand_metadata,
            mode="wb",
            max_file_size=max_file_size,
        )

    def _write(self, document: dict, file_handler: IO, _filename: str):
        raise NotImplementedError  # todo