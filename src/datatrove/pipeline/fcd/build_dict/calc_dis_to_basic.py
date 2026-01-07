import csv
import os

from collections import deque
import nltk
from nltk.corpus import wordnet as wn


def get_basic_synsets(basic_words):
    basic_synsets = []
    for word in basic_words:
        synsets = wn.synsets(word, pos="n")
        if synsets:
            basic_synsets.extend(synsets)
    basic_synsets = list(set(basic_synsets))
    print(f"get {len(basic_synsets)} basic synsets")
    return basic_synsets


def calc_dis_to_basic(basic_synsets):
    print("begin bfs")
    q = deque()
    for synset in basic_synsets:
        q.append(synset)
    dis_to_basic = {synset.name(): 100 for synset in wn.all_synsets("n")}
    for synset in basic_synsets:
        dis_to_basic[synset.name()] = 0
    visited_nodes = set()
    while q:
        synset = q.popleft()
        x = synset.name()
        if x in visited_nodes:
            continue
        visited_nodes.add(x)

        for neighbor in synset.hypernyms() + synset.instance_hypernyms() + synset.hyponyms() + synset.instance_hyponyms():
            y = neighbor.name()
            dis_to_basic[y] = min(dis_to_basic[y], dis_to_basic[x] + 1)
            q.append(neighbor)
    dis_to_basic = {k: v for k, v in sorted(dis_to_basic.items(), key=lambda item: (item[1], item[0]))}
    return dis_to_basic


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_path, exist_ok=True)
    basic_words = []
    with open(os.path.join(output_path, "basic_words.txt"), "r") as f:
        for line in f:
            basic_words.append(line.strip())
    basic_synsets = get_basic_synsets(basic_words)
    dis_to_basic = calc_dis_to_basic(basic_synsets)
    with open(os.path.join(output_path, "dis_to_basic.txt"), "w") as f:
        for k, v in dis_to_basic.items():
            f.write(f"{k} {v}\n")
    print(f"result saved in {os.path.join(output_path, 'dis_to_basic.txt')}")
