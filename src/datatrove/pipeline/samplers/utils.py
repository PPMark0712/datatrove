import json
import pickle
from datatrove.io import DataFolderLike


def read_json(file):
    return json.load(file)


def read_pickle(file):
    return pickle.load(file)


def read_text(file):
    data = []
    for line in file:
        data.append(float(line.strip()))
    return data


file_reader_mapping = {
    ".json": read_json,
    ".pkl": read_pickle,
    ".txt": read_text
}


def read_score_file(score_folder: DataFolderLike, rank: int):
    for ext, reader in file_reader_mapping.items():
        try:
            with score_folder.open(f"{rank:05d}{ext}") as f:
                return reader(f)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"Score file for rank {rank} not found in {score_folder}")
