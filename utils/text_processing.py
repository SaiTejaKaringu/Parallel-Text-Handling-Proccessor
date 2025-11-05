"""
Text Processing Utilities
Cleaning, preprocessing, and analysis functions
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from collections import Counter
import multiprocessing as mp

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def clean_text(text):
    """
    Clean and normalize text data
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        text = ' '.join(filtered_words)
    except:
        pass
    
    return text

def batch_clean_texts(texts, show_progress=False):
    """
    Clean multiple texts with optional progress tracking
    
    Args:
        texts: List of text strings
        show_progress: Whether to show progress
        
    Returns:
        List of cleaned texts
    """
    import streamlit as st
    
    cleaned = []
    total = len(texts)
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i, text in enumerate(texts):
        cleaned.append(clean_text(text))
        
        if show_progress and i % 10 == 0:
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Cleaning text: {i+1}/{total}")
    
    if show_progress:
        progress_bar.progress(1.0)
        status_text.text(f"✅ Cleaned {total} texts")
    
    return cleaned

def count_words_chunk(texts):
    """Count words in a chunk of texts (for multiprocessing)"""
    word_counts = Counter()
    for text in texts:
        if text:
            word_counts.update(text.split())
    return word_counts

def get_top_words(texts, top_n=20, use_parallel=True):
    """
    Get most frequent words from texts
    
    Args:
        texts: List of text strings
        top_n: Number of top words to return
        use_parallel: Whether to use multiprocessing
        
    Returns:
        List of (word, count) tuples
    """
    if use_parallel and len(texts) > 100:
        # Use multiprocessing for large datasets
        n_processes = min(mp.cpu_count(), 4)
        chunk_size = max(1, len(texts) // n_processes)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        with mp.Pool(processes=n_processes) as pool:
            results = pool.map(count_words_chunk, chunks)
        
        total_counts = Counter()
        for count in results:
            total_counts.update(count)
    else:
        # Sequential processing
        total_counts = Counter()
        for text in texts:
            if text:
                total_counts.update(text.split())
    
    return total_counts.most_common(top_n)

def get_text_statistics(texts):
    """
    Calculate text statistics
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with statistics
    """
    word_counts = [len(str(text).split()) for text in texts]
    char_counts = [len(str(text)) for text in texts]
    
    return {
        'total_texts': len(texts),
        'avg_words': sum(word_counts) / len(word_counts) if word_counts else 0,
        'avg_chars': sum(char_counts) / len(char_counts) if char_counts else 0,
        'min_words': min(word_counts) if word_counts else 0,
        'max_words': max(word_counts) if word_counts else 0,
        'total_words': sum(word_counts),
        'total_chars': sum(char_counts)
    }