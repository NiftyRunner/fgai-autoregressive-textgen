import os
import re
import urllib.request
from collections import Counter
import numpy as np

DATA_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "sherlock_holmes.txt")

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(RAW_FILE):
        print(f"Downloading dataset from {DATA_URL}...")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            raw_text = response.read().decode("utf-8-sig", errors="replace")
        
        # Strip Gutenberg header and footer
        start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
        end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
        
        start_idx = raw_text.find(start_marker)
        if start_idx != -1:
            start_idx = raw_text.find("\n", start_idx) + 1
        else:
            start_idx = 0
            
        end_idx = raw_text.find(end_marker)
        if end_idx == -1:
            end_idx = len(raw_text)
            
        clean_text = raw_text[start_idx:end_idx].strip()
        with open(RAW_FILE, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"Saved cleaned text to {RAW_FILE} ({len(clean_text)} characters).")
    else:
        print(f"Dataset already exists at {RAW_FILE}.")

if __name__ == "__main__":
    download_data()
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print("Character count:", len(text))
    # Tokenize: words and punctuation . , ; : ! ?
    tokens = re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?|[.,;:!?]", text.lower())
    print("Token count:", len(tokens))
    vocab = Counter(tokens)
    print("Unique tokens:", len(vocab))
    print("Top 20 tokens:", vocab.most_common(20))
