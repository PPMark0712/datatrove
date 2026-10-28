"""Build FCD resources from local inputs, without downloading Google Ngram."""

import argparse
import json
import shutil
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

from get_basic_words import get_basic_words
from calc_dis_to_basic import get_basic_synsets, calc_dis_to_basic


def prepare(cefr_csv, word_freq, output_dir, nltk_data=None):
    if nltk_data:
        nltk.data.path.insert(0, str(nltk_data))
    # Check the local corpus before writing resources. Never download implicitly.
    wordnet_version = wn.get_version()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, filename in ((cefr_csv, "oxford_5000.csv"), (word_freq, "word_freq.txt")):
        source, dest = Path(source).resolve(), (output_dir / filename).resolve()
        if source != dest:
            shutil.copyfile(source, dest)

    words = get_basic_words(output_dir / "oxford_5000.csv")
    if not words:
        raise ValueError("No CEFR A1/A2 words found")
    missing = set(words)
    frequency_rows = 0
    with (output_dir / "word_freq.txt").open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) != 2 or not fields[1].isdigit() or int(fields[1]) <= 0:
                raise ValueError(f"Invalid word frequency at line {number}")
            missing.discard(fields[0])
            frequency_rows += 1
    if missing:
        raise ValueError(f"Basic words missing from frequency table: {sorted(missing)}")
    (output_dir / "basic_words.txt").write_text("".join(word + "\n" for word in words), encoding="utf-8", newline="\n")
    distances = calc_dis_to_basic(get_basic_synsets(words))
    (output_dir / "dis_to_basic.txt").write_text(
        "".join(f"{name} {distance}\n" for name, distance in distances.items()), encoding="utf-8", newline="\n")
    summary = {
        "cefr_levels": ["a1", "a2"], "nltk_version": nltk.__version__,
        "wordnet_version": wordnet_version, "basic_words": len(words),
        "frequency_rows": frequency_rows, "noun_synsets": len(distances),
        "unreachable_distance": 100,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cefr-csv", type=Path, required=True)
    parser.add_argument("--word-freq", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--nltk-data", type=Path)
    args = parser.parse_args()
    prepare(args.cefr_csv, args.word_freq, args.output_dir, args.nltk_data)
