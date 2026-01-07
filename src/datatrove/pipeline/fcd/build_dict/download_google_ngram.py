import os
import subprocess
import sys


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "data", "google_books_ngram")
    os.makedirs(output_path, exist_ok=True)
    # source: https://storage.googleapis.com/books/ngrams/books/datasetsv3.html
    # download URL format (file 00000 to 00023)
    url_format = "http://storage.googleapis.com/books/ngrams/books/20200217/eng/1-{i:05d}-of-00024.gz"

    for i in range(24):
        url = url_format.format(i=i)
        output_file = os.path.join(output_path, f"{i:02d}.gz")
        print(f"Downloading: {url}")
        print(f"Saving to: {output_file}")
        try:
            result = subprocess.run(["curl", "-f", url, "--output", output_file])
            if result.returncode == 0:
                print(f"File {i:02d} downloaded successfully")
            else:
                print(f"Failed to download file {i:02d}")
        except KeyboardInterrupt:
            print("\nDownload interrupted by user")
            sys.exit(1)
    print("\nAll files downloaded!")
