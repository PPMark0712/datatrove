import os
import multiprocessing as mp
from collections import defaultdict
import gzip


def normalize_word(word):
    if "_" in word:
        word = word[:word.rfind("_")]
    word = word.lower()
    if word.isalpha():
        return word
    else:
        return None


def stat_file(fn):
    counter = defaultdict(int)
    with gzip.open(fn, "rt") as f:
        for line in f:
            line = line.split("\t")
            word = line[0]
            word = normalize_word(word)
            if word is None:
                continue
            for s in line[1:]:
                year, match_count, volume_count = s.split(",")
                counter[word] += int(match_count)
    return counter


if __name__ == "__main__":
    data_path = "google_books_ngram"
    output_file = "word_freq.txt"
    file_list = [os.path.join(data_path, fn) for fn in os.listdir(data_path)]
    
    result = defaultdict(int)
    with mp.Pool(24) as pool:
        for counter in pool.imap_unordered(stat_file, file_list):
            for word, cnt in counter.items():
                result[word] += cnt
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True) if v >= 30}
        
    with open(os.path.join(data_path, output_file), "w") as f:
        for k, v in result.items():
            f.write(f"{k}\t{v}\n")
