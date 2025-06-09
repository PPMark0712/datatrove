import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from itertools import tee
from typing import Iterable

import regex

from datatrove.utils.typeshelper import Languages


PUNCTUATION = "!/—”:％１〈&(、━\\【#%「」，】；+^]~“《„';’{|∶´[=-`*．（–？！：$～«〉,><》)?）。…@_.\"}►»" + "".join(
    map(
        chr,
        (x for a, b in ((0, 9), (11, 13), (13, 32), (127, 160)) for x in range(a, b)),
    )
)
TERMINAL_PUNCTUATION = {
    "᪩",
    "？",
    "⁈",
    "𑩂",
    "．",
    "꩞",
    "𑅃",
    "﹗",
    "𑂾",
    "\u1b7d",
    "፧",
    "𑅂",
    "꡶",
    "꘎",
    "⁉",
    "࠾",
    "᪨",
    "𑊩",
    "𑱂",
    "᱿",
    "𖩮",
    "᥅",
    "\U00011f43",
    "\U00011f44",
    "﹒",
    "𑈹",
    "𑈸",
    "።",
    "܂",
    "؞",
    "꛳",
    "\U00010f88",
    "𑗍",
    "𐩖",
    "𑙂",
    "\u061d",
    "꩟",
    "᠉",
    "\u1b7e",
    "𑗗",
    "᰼",
    "𑻸",
    "؟",
    "𑪜",
    "꧉",
    "𑗉",
    "𐽙",
    "𖫵",
    "𖬷",
    "܀",
    "꓿",
    "᜵",
    "𑗏",
    "𑁇",
    "𑗓",
    "𑥄",
    "៖",
    "𑥆",
    "𑗑",
    "𑗒",
    "꯫",
    "۔",
    "𐩗",
    "\U00010f86",
    "꡷",
    "\u2e54",
    "｡",
    "៕",
    "߹",
    "⸮",
    ".",
    "𑇅",
    "࠹",
    "𛲟",
    "꫰",
    "꤯",
    "𐽗",
    "᭞",
    "𑜼",
    "፨",
    "𑃁",
    "꣏",
    "𑇟",
    "𖬸",
    "𑪛",
    "𑜾",
    "࠷",
    "𝪈",
    "?",
    "𑃀",
    "𑗃",
    "！",
    "։",
    "꣎",
    "॥",
    "𑗖",
    "᭛",
    "᠃",
    "!",
    "၊",
    "𖺘",
    "⁇",
    "𑗌",
    "𑑋",
    "𖭄",
    "᭟",
    "𑅁",
    "𑙁",
    "⸼",
    "꩝",
    "𑗋",
    "。",
    "꧈",
    "꫱",
    "𑜽",
    "𐽖",
    "𑂿",
    "᙮",
    "។",
    "꛷",
    "\U00010f89",
    "៚",
    "᥄",
    "𑗕",
    "𑗎",
    "᪪",
    "᭚",
    "࠽",
    "𑇞",
    "𑗊",
    "𐽘",
    "\u2e53",
    "𑗔",
    "𖩯",
    "𑇍",
    "𑻷",
    "𐽕",
    "𑩃",
    "।",
    "𑗂",
    "𑇆",
    "𑁈",
    "။",
    "᱾",
    "𑱁",
    "꘏",
    "܁",
    "᜶",
    "‼",
    "𑈻",
    "‽",
    "᪫",
    "﹖",
    "𑑌",
    "𑈼",
    "\U00010f87",
    "𑗐",
    "៙",
    "᰻",
}
# add other scripts
PUNCTUATION_SET = set(PUNCTUATION).union(TERMINAL_PUNCTUATION)
PUNCTUATION_TRANS = str.maketrans(PUNCTUATION, " " * len(PUNCTUATION))


@dataclass
class TextNormConfig:
    lowercase: bool = True
    norm_whitespace: bool = True
    remove_punctuation: bool = True
    norm_unicode_diacritics: bool = True
    norm_numbers: bool = True
    norm_weekdays: bool = False
    norm_monthnames: bool = False


DEF_TEXT_NORM_CONFIG = TextNormConfig()
# Match digits in any script, allowing for different decimal separators
# One or more digits in any script
# Common decimal separators (period, comma, Arabic decimal, etc)
# Optional decimal part with digits
# we need regex and not re for this one to match unicode
NUMBERS_PATTERN = regex.compile(
    r"\p{Nd}+([.,،٫⎖⎗⎘]{1}\p{Nd}+)?",
    regex.VERBOSE | regex.UNICODE,
)
WHITESPACE_PATTERN = re.compile(r"\s+")
# WARNING: english specific
WEEKDAYS_PATTERN = re.compile(r"monday|tuesday|wednesday|thursday|friday|saturday|sunday")
MONTHS_PATTERN = re.compile(r"january|february|march|april|may|june|july|august|september|october|november|december")


def simplify_text(text: str, config=DEF_TEXT_NORM_CONFIG) -> str:
    """Performs the following operations to increase recall when looking for matches between documents:
    - number normalization
    - weekday normalization
    - month normalization
    - lowercase text
    - replace all whitespace with a single " "
    - remove all punctuation
    - convert diacritics
    - unicode normalize

    Args:
        text

    Returns:
        modified text
    """
    # We should apply the transformation in such order so that, we do same transformations
    # incrementaly as we would do if we applied each from scratch.
    # Eg.
    # 1|2|3 -> 000
    # vs
    # 1|2|3 -> 0

    # lower case
    if config.lowercase:
        text = text.lower()
    if config.norm_numbers:
        text = NUMBERS_PATTERN.sub("0", text)
    if config.norm_weekdays:
        text = WEEKDAYS_PATTERN.sub("WEEKDAY", text)
    if config.norm_monthnames:
        text = MONTHS_PATTERN.sub("MONTH", text)

    # convert punctuation to spaces
    if config.remove_punctuation:
        text = text.translate(PUNCTUATION_TRANS)

    # remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    if config.norm_whitespace:
        text = WHITESPACE_PATTERN.sub(" ", text.strip())
    # diacritics/unicode normalization
    if config.norm_unicode_diacritics:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")

    return text.strip()


# from https://tedboy.github.io/nlps/_modules/nltk/util.html#ngrams
def ngrams(sequence: Iterable, n: int):
    iterables = tee(sequence, n)

    for i, sub_iterable in enumerate(iterables):  # For each window,
        for _ in range(i):  # iterate through every order of ngrams
            next(sub_iterable, None)  # generate the ngrams within the window.
    return zip(*iterables)  # Unpack and flattens the iterables.


SPLIT_TEXT_DOCUMENTS = "DOCUMENT"
SPLIT_TEXT_SENTENCES = "SENTENCE"
SPLIT_TEXT_PARAGRAPHS = "PARAGRAPH"
SPLIT_TEXT_WORDS = "WORDS"


@lru_cache(5)
def split_into_parts(text, mode="DOCUMENT", language=Languages.english):
    from datatrove.utils.word_tokenizers import load_word_tokenizer

    if mode == SPLIT_TEXT_DOCUMENTS:
        return [text]
    elif mode == SPLIT_TEXT_SENTENCES:
        tokenizer = load_word_tokenizer(language)
        spans = [b for _, b in tokenizer.span_tokenize(text)]
        return [text[a:b] for a, b in zip([0] + spans[:-1], spans[:-1] + [len(text)])]
    elif mode == SPLIT_TEXT_WORDS:
        tokenizer = load_word_tokenizer(language)
        return tokenizer.word_tokenize(text)
    elif mode == SPLIT_TEXT_PARAGRAPHS:
        # merge whitespace with prev line
        og_lines = text.splitlines()
        lines = []
        next_line = []
        for li, line in enumerate(og_lines):
            if line.strip() and next_line:
                lines.append("".join(next_line))
                next_line = []
            next_line.append(line)
            if li != len(og_lines) - 1:
                next_line.append("\n")
        if next_line:
            lines.append("".join(next_line))
        return lines
    else:
        raise ValueError(f"Unknown {mode=}")


def split_into_words(text, language=Languages.english):
    return split_into_parts(text, mode=SPLIT_TEXT_WORDS, language=language)


def split_into_sentences(text, language=Languages.english):
    return split_into_parts(text, mode=SPLIT_TEXT_SENTENCES, language=language)


def split_into_paragraphs(text, language=Languages.english):
    return split_into_parts(text, mode=SPLIT_TEXT_PARAGRAPHS, language=language)


def preprocess_for_repetition(text: str)->str:
    text =  text.replace('|', ' ')
    text =  text.replace('<br>', '\n')
    text = re.sub(r'\d+\.\s+', '', text)
    lines = text.splitlines()
    processed_lines = [line.lstrip("+-*• ") for line in lines]
    text = '\n'.join(processed_lines)
    text = re.sub(r' +', ' ', text)
    return text
    

def preprocess_for_alpha_word_ratio(text: str)->str:
    text =  text.replace('|', ' ')
    text =  text.replace('<br>', ' ')
    text = re.sub(r'\d+\.\s+', '', text)
    lines = text.splitlines()
    processed_lines = [line.lstrip("+-*• ") for line in lines]
    text = '\n'.join(processed_lines)
    text = re.sub(r'\d+[\.\、\)）]\s+', '', text)
    text = re.sub(r'[A-Ha-h][\.\、\)）]\s+', '', text)
    text = re.sub(r'[\(\[\{（［【]\s*\d+\s*[\)\]\}）］】]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text