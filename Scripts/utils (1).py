"""
utils.py - Helper functions for Arabic AI-text detection project
Contains common utilities, configuration, and shared functions
"""

import os
import re
import subprocess
import sys
from typing import List, Set

# ============================================================================
# Environment Setup
# ============================================================================

def setup_environment(base_dir: str = "./arabic_ai_detection_f011_f034"):
    """
    Create project directory structure if it doesn't exist.
    
    Args:
        base_dir: Root directory for the project
        
    Returns:
        Dictionary with all directory paths
    """
    dirs = {
        "BASE_DIR": os.path.abspath(base_dir),
        "RAW_DIR": os.path.join(base_dir, "data", "raw"),
        "PROC_DIR": os.path.join(base_dir, "data", "processed"),
        "MODEL_DIR": os.path.join(base_dir, "models"),
        "FIG_DIR": os.path.join(base_dir, "reports", "figures"),
        "STREAM_IN": os.path.join(base_dir, "stream", "incoming"),
        "STREAM_OUT": os.path.join(base_dir, "stream", "predictions"),
        "CHK_DIR": os.path.join(base_dir, "stream", "checkpoint"),
    }
    
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    return dirs


def install_libraries():
    """Install required Python libraries if not already present."""
    required_pkgs = ["pyarabic", "nltk", "datasets", "pyarrow"]
    
    for pkg in required_pkgs:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", 
             "--break-system-packages", pkg],
            check=False
        )
    
    import nltk
    for resource in ["stopwords", "punkt", "punkt_tab"]:
        try:
            if resource == "stopwords":
                nltk.data.find(f"corpora/{resource}")
            else:
                nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def get_feature_names(clean_text_column: str = "clean_text_columns") -> tuple:
    """
    Return standard feature names used throughout the project.
    
    Args:
        clean_text_column: Name of the cleaned text column
        
    Returns:
        Tuple of (feature_name_f011, feature_name_f034)
    """
    feature_f011 = f'{clean_text_column}_f011_num_long_words_over_N'
    feature_f034 = f'{clean_text_column}_f034_total_lines'
    return feature_f011, feature_f034


# ============================================================================
# Text Processing Functions
# ============================================================================

def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text using pyarabic.araby.
    Strips diacritics, normalizes hamza/alef/yaa/taa.
    
    Args:
        text: Raw Arabic text
        
    Returns:
        Normalized text
    """
    if text is None:
        return None
    
    import pyarabic.araby as araby
    
    t = araby.strip_tashkeel(text)   # remove diacritics
    t = araby.strip_tatweel(t)       # remove kashida
    t = araby.normalize_hamza(t)     # ء/أ/إ/آ → ا
    t = araby.normalize_alef(t)      # alef variants
    t = araby.normalize_ligature(t)  # لا ligatures
    
    # Keep Arabic letters, ASCII letters/digits, basic punct
    t = re.sub(r"[^\u0600-\u06FF\sA-Za-z0-9\.\,\!\?\:\;\n]", " ", t)
    t = re.sub(r"[ \t]+", " ", t).strip()
    
    return t


def compute_long_word_ratio(words: List[str], min_length: int = 6) -> float:
    """
    Compute ratio of long words (length > min_length) to total words.
    Feature f011.
    
    Args:
        words: List of tokens
        min_length: Minimum length to consider a word "long"
        
    Returns:
        Float ratio of long words to total words
    """
    total_words = len(words)
    if total_words == 0:
        return 0.0
    
    long_words_count = sum(1 for w in words if len(w) > min_length)
    return float(long_words_count) / total_words


def count_total_lines(raw_text: str) -> int:
    """
    Count physical lines in the original text.
    Feature f034.
    
    Args:
        raw_text: Original text (un-normalized)
        
    Returns:
        Number of lines in the text
    """
    if not raw_text:
        return 0
    return raw_text.count("\n") + 1


def tokenize_arabic(text: str, stopwords: Set[str]) -> List[str]:
    """
    Tokenize Arabic text and filter stopwords and short tokens.
    
    Args:
        text: Normalized Arabic text
        stopwords: Set of Arabic stopwords
        
    Returns:
        List of filtered tokens
    """
    if not text:
        return []
    
    import pyarabic.araby as araby
    
    toks = araby.tokenize(text)
    return [w for w in toks if len(w) > 1 and w not in stopwords and not w.isspace()]


def generate_bigrams(tokens: List[str]) -> List[str]:
    """
    Generate bigrams from a list of tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        List of bigram strings formatted as "token1_token2"
    """
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


# ============================================================================
# Dataset Utilities
# ============================================================================

def load_and_melt_dataset(raw_data_path: str, ai_columns: List[str]) -> pd.DataFrame:
    """
    Load the dataset from CSV and melt it into long format.
    
    Args:
        raw_data_path: Path to the CSV file
        ai_columns: List of column names containing AI-generated abstracts
        
    Returns:
        DataFrame with columns: text, label, model, source
    """
    import pandas as pd
    
    df_pd = pd.read_csv(raw_data_path)
    rows = []
    
    for _, r in df_pd.iterrows():
        # Human-written abstracts (label 0)
        if isinstance(r.get("original_abstract"), str) and r["original_abstract"].strip():
            rows.append((r["original_abstract"], 0, "human", "huggingface_data"))
        
        # AI-generated abstracts (label 1)
        for col in ai_columns:
            v = r.get(col)
            if isinstance(v, str) and v.strip():
                rows.append((v, 1, col.replace("_generated_abstract", ""), "huggingface_data"))
    
    return pd.DataFrame(rows, columns=["text", "label", "model", "source"])