from collections import deque

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.wsd import lesk

nltk.data.path.append("/data1/yyz/downloads/models/nltk_data")
stop_words = set(stopwords.words("english"))

def is_valid_word(word):
    return word.isalpha() and len(word) > 1 and word not in stop_words


def get_wordnet_pos(treebank_tag):
    """
    将Treebank词性标签转换为WordNet词性标签。
    Lesk算法可以使用词性信息来限制搜索范围，提高准确性。
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None


def get_subtree_max_depth(root_synset):
    """
    计算 WordNet 中以 root_synset 为根的子树最大深度。
    :param root_synset: 子树根节点（Synset 对象，例如 wn.synset('entity.n.01')）
    :return: 子树最大深度（整数）
    """
    max_depth = 0
    visited = set()
    queue = deque([(root_synset, 0)])
    while queue:
        current_synset, current_depth = queue.popleft()
        max_depth = max(max_depth, current_depth)
        if current_synset in visited:
            continue
        visited.add(current_synset)
        hyponyms = current_synset.hyponyms() + current_synset.instance_hyponyms()
        for hypo in hyponyms:
            queue.append((hypo, current_depth + 1))
    return max_depth


def get_normalized_depth(synset, subtree_max_depth_cache=None):
    """
    计算一个词义的归一化深度（相对深度）。
    :param synset: 要计算的 Synset 对象
    :param subtree_max_depth_cache: 预计算的子树最大深度缓存（字典，key 为根节点，value 为最大深度）
    :return: 归一化深度（float，范围 [0, 1]）
    """
    # 获取词义的绝对深度（最短路径）
    depth = synset.max_depth()
    if subtree_max_depth_cache and synset in subtree_max_depth_cache:
        subtree_depth = subtree_max_depth_cache[synset]
    else:
        subtree_depth = get_subtree_max_depth(synset)
        if subtree_max_depth_cache is not None:
            subtree_max_depth_cache[synset] = subtree_depth
    max_depth = subtree_depth + depth
    if max_depth == 0:  # 避免除零
        return 0.0
    return depth / max_depth


def disambiguate_sentence(sentence):
    """
    对句子中的每个词进行词义消歧，并返回词 -> Synset 的映射。
    """
    word_synsets = {}
    
    words = word_tokenize(sentence, language="english")
    tagged_words = pos_tag(words)
    
    for i, (word, tag) in enumerate(tagged_words):
        if word.isalpha() and len(word) > 1:
            pos = get_wordnet_pos(tag)
            best_synset = lesk(words, word, pos=pos)
            if best_synset is None:
                continue
            paths = best_synset.hypernyms()
            print(best_synset, paths)
            depth = max(0, len(paths) - 1)
            if best_synset:
                word_synsets[word] = best_synset
    return word_synsets


def show(word):
    print(word)
    synsets = wn.synsets(word)
    print(synsets)
    for synset in synsets:
        hypernyms = synset.hypernym_paths()[0]
        print(hypernyms, synset)


def calc_avg_hypernym_depth(text):
    sentences = sent_tokenize(text)
    cache = {}
    for sentence in sentences:
        words = word_tokenize(sentence)
        words_with_pos = pos_tag(words)
        for word, pos in words_with_pos:
            if not is_valid_word(word):
                print("invalid word: ", word)
                continue
            wn_pos = get_wordnet_pos(pos)
            synset = lesk(words, word, pos=wn_pos)
            if synset is not None:
                paths = synset.hypernym_paths()
                hypernyms = synset.hypernyms()
                # print(word, len(path) - 1)
                print("-" * 20)
                print(word, synset.min_depth(), synset.max_depth(), get_normalized_depth(synset, cache))
                for path in paths:
                    print(len(path), path)


if __name__ == "__main__":
    text = "Sign up for our latest news in your inbox.\nRegister to subscribe to newsletters\nDavid Murphy talks funding and click fraud with Trademob CEO, Ravi Kamran.\nYou can listen to the interview below, or download it as a podcast by clicking here.\nTechnical Project Manager / Scrum Master, London\nUK Sales Manager, London\nContent and Communications Manager , London\nBusiness Development Manager - Mobile Media, London\nSenior Account Manager - Mobile Media, London\nHead of Sales - Mobile Application Agency, London\nMobile Marketing Manager - Music, London\nHead of Sales UK - Mobile Advertising, London"
    calc_avg_hypernym_depth(text)
