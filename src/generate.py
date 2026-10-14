import math
import re
from collections import Counter
import torch
import torch.nn.functional as F
import pandas as pd

# 4 fixed seeds of ~10-25 words taken from or inspired by The Adventures of Sherlock Holmes
FIXED_SEEDS = [
    "to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name .",
    "it has long been an axiom of mine that the little things are infinitely the most important .",
    "i had called upon my friend sherlock holmes upon the second morning after christmas , with the intention of wishing him",
    "there is nothing more deceptive than an obvious fact , he remarked as he sat back in his chair ."
]

PUNCTUATION_TOKENS = {".", ",", ";", ":", "!", "?"}

def detokenize(tokens):
    """
    Converts token list back into readable English text with appropriate punctuation spacing.
    """
    if not tokens:
        return ""
    result = []
    for i, tok in enumerate(tokens):
        if tok in PUNCTUATION_TOKENS:
            if result:
                result[-1] = result[-1] + tok
            else:
                result.append(tok)
        else:
            result.append(tok)
    text = " ".join(result)
    
    # Capitalize first letter and after . ! ?
    sentences = re.split(r"([.!?]\s+)", text)
    capitalized = ""
    for part in sentences:
        if capitalized and capitalized[-1] in ".!? ":
            capitalized += part.capitalize()
        elif not capitalized:
            capitalized += part.capitalize()
        else:
            capitalized += part
    return capitalized

def generate_text(
    model,
    vocab,
    seed_text,
    num_words_to_generate=60,
    temperature=1.0,
    seq_len=25,
    device=None,
    seed=None
):
    """
    Generates text autoregressively from a seed string using temperature sampling.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    
    from src.data_utils import tokenize_text
    seed_tokens = tokenize_text(seed_text)
    encoded_tokens = vocab.encode(seed_tokens)
    
    generated_indices = []
    
    with torch.no_grad():
        for _ in range(num_words_to_generate):
            # Context window: keep the last seq_len tokens
            context = (encoded_tokens + generated_indices)[-seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            
            logits = model(x)  # (1, vocab_size)
            
            # Temperature scaling
            scaled_logits = logits / max(1e-5, temperature)
            probs = F.softmax(scaled_logits, dim=-1)
            
            # Multinomial sampling
            next_idx = torch.multinomial(probs, num_samples=1).item()
            generated_indices.append(next_idx)
            
    generated_tokens = vocab.decode(generated_indices)
    full_tokens = seed_tokens + generated_tokens
    
    return {
        "seed_text": seed_text,
        "seed_tokens": seed_tokens,
        "generated_tokens": generated_tokens,
        "full_tokens": full_tokens,
        "generated_text": detokenize(generated_tokens),
        "full_text": detokenize(full_tokens)
    }

def compute_text_quality_metrics(tokens):
    """
    Computes lexical diversity and repetition metrics on generated tokens:
    - distinct-1: unique unigrams / total unigrams
    - distinct-2: unique bigrams / total bigrams
    - repeated-trigram ratio: (total trigrams - unique trigrams) / total trigrams
    - unique-word ratio: unique words (excluding punctuation) / total words
    """
    if not tokens:
        return {
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repeated_trigram_ratio": 0.0,
            "unique_word_ratio": 0.0
        }
        
    n_tokens = len(tokens)
    unigrams = tokens
    unique_unigrams = len(set(unigrams))
    distinct_1 = unique_unigrams / max(1, n_tokens)
    
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens) - 1)]
    distinct_2 = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
    
    trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens) - 2)]
    if trigrams:
        repeated_trigram_ratio = (len(trigrams) - len(set(trigrams))) / len(trigrams)
    else:
        repeated_trigram_ratio = 0.0
        
    word_tokens = [t for t in tokens if t not in PUNCTUATION_TOKENS]
    if word_tokens:
        unique_word_ratio = len(set(word_tokens)) / len(word_tokens)
    else:
        unique_word_ratio = distinct_1
        
    return {
        "distinct_1": float(distinct_1),
        "distinct_2": float(distinct_2),
        "repeated_trigram_ratio": float(repeated_trigram_ratio),
        "unique_word_ratio": float(unique_word_ratio)
    }
