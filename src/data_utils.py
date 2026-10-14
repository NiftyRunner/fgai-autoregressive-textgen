import os
import re
import urllib.request
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
DATA_DIR = "data"
RAW_FILE = os.path.join(DATA_DIR, "sherlock_holmes.txt")

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

def download_and_clean_data(data_dir=DATA_DIR, raw_file=RAW_FILE):
    """
    Downloads The Adventures of Sherlock Holmes (Project Gutenberg #1661)
    and removes the Project Gutenberg header and footer.
    """
    os.makedirs(data_dir, exist_ok=True)
    if not os.path.exists(raw_file):
        print(f"Downloading from {DATA_URL}...")
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            raw_text = response.read().decode("utf-8-sig", errors="replace")
        
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
        with open(raw_file, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"Saved clean text to {raw_file} ({len(clean_text)} characters).")
    else:
        with open(raw_file, "r", encoding="utf-8") as f:
            clean_text = f.read()
            
    return clean_text

def tokenize_text(text):
    """
    Lowercases text and extracts words and specific punctuation (. , ; : ! ?)
    as separate tokens. Contractions such as 'don't' or 'sherlock's' are preserved.
    """
    pattern = r"[a-z0-9]+(?:'[a-z0-9]+)?|[.,;:!?]"
    tokens = re.findall(pattern, text.lower())
    return tokens

def compute_corpus_statistics(raw_text, tokens):
    """
    Computes statistical characteristics of the dataset:
    characters, total tokens, unique words, sentence length stats, etc.
    """
    char_count = len(raw_text)
    token_count = len(tokens)
    token_counts = Counter(tokens)
    unique_tokens = len(token_counts)
    
    # Sentence statistics (splitting by sentence terminators . ! ?)
    sentences = re.split(r"[.!?]+", raw_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_lengths = [len(tokenize_text(s)) for s in sentences if len(tokenize_text(s)) > 0]
    
    stats = {
        "char_count": char_count,
        "token_count": token_count,
        "unique_tokens": unique_tokens,
        "sentence_count": len(sentence_lengths),
        "sentence_len_mean": float(np.mean(sentence_lengths)),
        "sentence_len_median": float(np.median(sentence_lengths)),
        "sentence_len_min": int(np.min(sentence_lengths)),
        "sentence_len_max": int(np.max(sentence_lengths)),
        "sentence_len_std": float(np.std(sentence_lengths)),
        "top_20_tokens": token_counts.most_common(20)
    }
    return stats

class Vocabulary:
    def __init__(self, max_vocab_size=8000):
        self.max_vocab_size = max_vocab_size
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        
        self.word2idx = {self.pad_token: 0, self.unk_token: 1}
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        self.token_counts = Counter()
        
    def build_vocab(self, tokens):
        self.token_counts = Counter(tokens)
        # Cap vocabulary at max_vocab_size including special tokens
        num_words_to_add = self.max_vocab_size - len(self.word2idx)
        most_common = self.token_counts.most_common(num_words_to_add)
        
        for word, _ in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
    def __len__(self):
        return len(self.word2idx)
    
    def encode(self, tokens):
        unk_idx = self.word2idx[self.unk_token]
        return [self.word2idx.get(t, unk_idx) for t in tokens]
    
    def decode(self, indices):
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]

class SlidingWindowDataset(Dataset):
    def __init__(self, token_indices, seq_len=25):
        self.seq_len = seq_len
        self.data = token_indices
        self.num_samples = max(0, len(self.data) - self.seq_len)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + self.seq_len], dtype=torch.long)
        return x, y

def prepare_dataloaders(raw_text, seq_len=25, max_vocab_size=8000, batch_size=128, fast_mode=False, seed=42):
    """
    Tokenizes, builds vocabulary, splits into 80% train, 10% val, 10% test,
    and returns DataLoaders with num_workers=0.
    """
    tokens = tokenize_text(raw_text)
    
    if fast_mode:
        print("FAST_MODE enabled: truncating tokens to 10,000 for smoke testing.")
        tokens = tokens[:10000]
        max_vocab_size = min(1000, max_vocab_size)
        
    vocab = Vocabulary(max_vocab_size=max_vocab_size)
    vocab.build_vocab(tokens)
    encoded = vocab.encode(tokens)
    
    # Sequential split on token sequence to avoid data leakage
    n_tokens = len(encoded)
    train_end = int(0.80 * n_tokens)
    val_end = int(0.90 * n_tokens)
    
    train_tokens = encoded[:train_end]
    val_tokens = encoded[train_end:val_end]
    test_tokens = encoded[val_end:]
    
    train_dataset = SlidingWindowDataset(train_tokens, seq_len=seq_len)
    val_dataset = SlidingWindowDataset(val_tokens, seq_len=seq_len)
    test_dataset = SlidingWindowDataset(test_tokens, seq_len=seq_len)
    
    # Windows DataLoader
    # Train is shuffled, val and test are sequential
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return {
        "tokens": tokens,
        "vocab": vocab,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset
    }
