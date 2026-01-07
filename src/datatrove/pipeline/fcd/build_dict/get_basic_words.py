import csv
import os
import requests


def download_github_file(raw_url, save_path):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    response = requests.get(raw_url, timeout=10)
    response.raise_for_status()

    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded Oxford with CEFR labels: {save_path}")


def get_basic_words(cefr_fn):
    basic_words = []
    basic_levels = ["a1", "a2"]
    with open(cefr_fn, 'r', encoding='utf-8') as f:
        dict_reader = csv.DictReader(f, delimiter=',')
        for row in dict_reader:
            if row["cefr"] in basic_levels and row["word"].isalpha():
                basic_words.append(row["word"].lower())
    basic_words = sorted(list(set(basic_words)))
    print(f"basic word count: {len(basic_words)}")
    return basic_words


if __name__ == "__main__":
    # source: https://github.com/winterdl/oxford-5000-vocabulary-audio-definition  file_path: data/oxford_5000.csv
    github_raw_url = "https://raw.githubusercontent.com/winterdl/oxford-5000-vocabulary-audio-definition/main/data/oxford_5000.csv"
    output_path = os.path.join(os.path.dirname(__file__), "data")
    output_file = os.path.join(output_path, "basic_words.txt")
    cefr_fn =  os.path.join(output_path, "oxford_5000.csv")
    download_github_file(github_raw_url, cefr_fn)

    basic_words = get_basic_words(cefr_fn)
    os.makedirs(output_path, exist_ok=True)
    with open(output_file, "w", encoding='utf-8') as f:
        f.writelines([word + "\n" for word in basic_words])
    print(f"basic words saved to {output_file}")
