import json
import math
import sys
from pathlib import Path

import nltk
import pytest

from datatrove.pipeline.fcd.fcd_calculator import FcdCalculator
from datatrove.pipeline.readers import JsonlReader


BUILD_DIR = Path(__file__).resolve().parents[2] / "src/datatrove/pipeline/fcd/build_dict"
LOCAL_NLTK = Path(__file__).resolve().parents[2] / "webui/.cache/nltk_data"
nltk.data.path.insert(0, str(LOCAL_NLTK))
sys.path.insert(0, str(BUILD_DIR))
from prepare_local import prepare


def require_wordnet():
    try:
        nltk.corpus.wordnet.get_version()
        nltk.corpus.stopwords.words("english")
    except LookupError:
        pytest.skip("Local NLTK WordNet and stopwords are required")


def test_local_dictionary_build_is_reproducible(tmp_path):
    require_wordnet()
    csv = tmp_path / "source.csv"
    csv.write_text("word,cefr\ncat,a1\ndog,a2\nCat,a1\nhigh-tech,a1\nadvanced,c1\n", encoding="utf-8")
    freq = tmp_path / "freq.txt"
    freq.write_text("cat\t100\ndog 200\n", encoding="utf-8")
    target = tmp_path / "data"
    first = prepare(csv, freq, target)
    assert (target / "basic_words.txt").read_bytes() == b"cat\ndog\n"
    assert first["basic_words"] == 2
    assert first["noun_synsets"] > 0
    assert first == prepare(target / "oxford_5000.csv", target / "word_freq.txt", target)
    assert not (target / "manifest.json").exists()


def test_missing_basic_word_frequency_is_rejected(tmp_path):
    require_wordnet()
    csv, freq = tmp_path / "source.csv", tmp_path / "freq.txt"
    csv.write_text("word,cefr\ncat,a1\ndog,a2\n", encoding="utf-8")
    freq.write_text("cat 100\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dog"):
        prepare(csv, freq, tmp_path / "data")


def test_fcd_dictionary_accepts_utf8_and_whitespace(tmp_path):
    require_wordnet()
    freq, basic, distance = (tmp_path / name for name in ("freq.txt", "basic.txt", "distance.txt"))
    freq.write_text("cat\t100\ndog 200\ncafé 50\n", encoding="utf-8")
    basic.write_text("cat\ndog\n", encoding="utf-8")
    distance.write_text("cat.n.01 0\ndog.n.01 2\nentity.n.01 100\n", encoding="utf-8")
    calculator = FcdCalculator(str(tmp_path / "out"), word_freq_path=str(freq),
                               basic_words_path=str(basic), dis_to_basic_path=str(distance))
    calculator.init_dict()
    assert math.isclose(calculator.word_log_freq["café"], math.log(50))
    assert math.isclose(calculator.log_freq_center, math.log(100))


def test_jsonl_reader_preserves_utf8_test_records(tmp_path):
    records = [{"id": "a", "text": "English café and 中文文本。"},
               {"id": "b", "text": "Second document."}]
    (tmp_path / "test.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    documents = list(JsonlReader(str(tmp_path))())
    assert [(doc.id, doc.text) for doc in documents] == [(r["id"], r["text"]) for r in records]
