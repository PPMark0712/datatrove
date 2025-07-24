import json
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.data.path.append("/data1/yyz/downloads/models/nltk_data")

stop_words = stopwords.words("english")
stop_words = set(stop_words)
def is_valid_word(word: str) -> bool:
    return len(word) > 1 and any(c.isalpha() for c in word) and word not in stop_words
fn = "/data1/yyz/downloads/datasets/lexical_concreteness/lexical_concreteness.json"
with open(fn, "r") as f:
    concreteness_score = json.load(f)

def conver_pos(pos_tag: str):
    c = pos_tag[0]
    d = {
        "N": "n",
        "V": "v",
        "J": "a",
        "R": "r",
    }
    return d.get(c, "n")

text = "Tommy Pi started DJing at small private parties at the age of 13. He was always into music, so it was no surprise when he went on and bought his first equipment, some Omnictronic Turntables and a small Gemini mixer. From that day on he always used any chance he got to play at parties, and it wasn't long before his day would come, for him play on a big stage in front of more than 50 people!"
text = "As design philosophers, we believe that the act of creation transcends mere functionality, endeavoring instead to illuminate the intricate relationship between form, purpose, and the human experience, ultimately revealing profound insights about our needs, aspirations, and the very nature of interaction."
# text = "To make Stir-fried Tomatoes and Scrambled Eggs, you first whisk the eggs and fry them until semi-cooked, then set them aside; next, stir-fry the tomato chunks until they release their juices, finally add the eggs back in, season, and stir-fry until everything is well combined."
# text = "A Fenwick Tree, also known as a Binary Indexed Tree (BIT), is a data structure that can efficiently update elements and calculate prefix sums in an array. It achieves this by representing an array as a tree-like structure where each node stores the sum of a specific range of elements. The key insight lies in how it uses the binary representation of indices to determine which nodes to update or query. This allows for both updates and prefix sum queries to be performed in O(logN) time, making it significantly faster than a simple array for these operations, which would take O(N) for updates (if maintaining prefix sums) or O(N) for prefix sums (if updating individual elements). It's particularly useful in competitive programming and scenarios requiring frequent range sum queries and single-point updates."
# text = "K-State put themselves in sole position of first place in the Big 12 with their 79-70 over Iowa State on Saturday, and K-State is now #10 in the AP poll heading into Monday night’s Sunflower Showdown. With losses last week at TCU and Oklahoma, KU drops nine spots to #14. Oklahoma State has won five straight including at Texas on Saturday, and is now #17, up five spots from last week’s poll.\nK-State and KU tip off on Monday night at 8:00pm at Allen Fieldhouse. Coverage on SportsRadio 1350 KMAN and 101.5 K-Rock begins at 7:00 from the K-State Sports Network with Wyatt Thompson and Stan Weber.\nStay connected to all things KSU on the go just text EMAW to 88474\nFor full video wrap-ups, including analysis, highlights, coaches & player interviews of K-State Football & Basketball check out PowerCatGameday.com"
words = word_tokenize(text)
words_with_pos_tag = pos_tag(words)
lemmatizer = WordNetLemmatizer()
total = 0
cnt = 0
for word, pos in words_with_pos_tag:
    if not is_valid_word(word):
        continue
    converted_pos = conver_pos(pos)
    org_word = lemmatizer.lemmatize(word, converted_pos)
    score = concreteness_score.get(org_word, None)
    if score:
        cnt += 1
        total += score
        if score < 3:
            print(word, score)
print(total / cnt)
    