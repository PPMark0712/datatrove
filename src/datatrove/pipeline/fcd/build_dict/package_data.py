"""Create or validate the standalone FCD Release asset using only the standard library."""

import argparse
import os
import tempfile
import zipfile
from pathlib import Path


RESOURCE_FILES = ("basic_words.txt", "dis_to_basic.txt", "word_freq.txt", "oxford_5000.csv")
DEFAULT_DATA = Path(__file__).resolve().parent / "data"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[5] / "dist/fcd_data.zip"


def validate_archive(path):
    with zipfile.ZipFile(path) as archive:
        expected = {f"data/{name}" for name in RESOURCE_FILES}
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        if set(names) != expected or len(names) != len(expected):
            raise ValueError(f"Archive must contain exactly these four files: {sorted(expected)}")
        if any(entry.file_size == 0 for entry in archive.infolist() if not entry.is_dir()):
            raise ValueError("Empty FCD resource in archive")
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC check failed: {bad}")
    return sorted(expected)


def package(data_dir=DEFAULT_DATA, output=DEFAULT_OUTPUT):
    data_dir, output = Path(data_dir), Path(output)
    for name in RESOURCE_FILES:
        path = data_dir / name
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"Missing or empty resource: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=output.parent, suffix=".zip.tmp")
    os.close(fd)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in RESOURCE_FILES:
                archive.write(data_dir / name, f"data/{name}")
        validate_archive(temporary)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate", type=Path, help="Validate an existing ZIP instead of creating one")
    args = parser.parse_args()
    if args.validate:
        print("Validated:", ", ".join(validate_archive(args.validate)))
    else:
        path = package(args.data_dir, args.output)
        print(f"Created {path.resolve()} ({path.stat().st_size:,} bytes)")
