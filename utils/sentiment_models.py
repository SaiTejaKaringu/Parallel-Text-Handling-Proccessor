"""
Sentiment Analysis Models
Streamlit-safe implementations for TextBlob and RoBERTa
✅ Thread-safe (no pickling)
✅ Parallelizable
✅ Works with Streamlit reruns
"""

import pandas as pd
import streamlit as st
import time
from concurrent.futures import ThreadPoolExecutor
from textblob import TextBlob
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================
# TEXTBLOB IMPLEMENTATION
# ============================================================

def textblob_sentiment(text):
    """
    Analyze sentiment using TextBlob
    Returns (sentiment_label, polarity_score)
    """
    if pd.isna(text) or text == "":
        return 'neutral', 0.0

    try:
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity

        # More sensitive thresholds for cleaner data
        if polarity > 0.05:
            sentiment = 'positive'
        elif polarity < -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        return sentiment, polarity
    except Exception:
        return 'neutral', 0.0


def textblob_sequential(texts, show_progress=False):
    """Sequential TextBlob sentiment analysis"""
    results = []
    total = len(texts)

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    for i, text in enumerate(texts):
        sentiment, polarity = textblob_sentiment(text)
        results.append((sentiment, polarity))

        if show_progress and i % 10 == 0:
            progress_bar.progress((i + 1) / total)
            status_text.text(f"TextBlob Sequential: {i+1}/{total}")

    if show_progress:
        progress_bar.progress(1.0)
        status_text.text(f"✅ TextBlob Sequential: {total} analyzed")

    return results


def textblob_parallel(texts, max_workers=4, show_progress=False):
    """Parallel TextBlob sentiment analysis (thread-safe)"""
    results = []
    total = len(texts)

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(textblob_sentiment, texts)
        for i, result in enumerate(futures):
            results.append(result)
            if show_progress and i % 10 == 0:
                progress_bar.progress((i + 1) / total)
                status_text.text(f"TextBlob Parallel: {i+1}/{total}")

    if show_progress:
        progress_bar.progress(1.0)
        status_text.text(f"✅ TextBlob Parallel: {total} analyzed")

    return results


# ============================================================
# ROBERTA IMPLEMENTATION
# ============================================================

@st.cache_resource
def load_roberta_model():
    """Load and cache RoBERTa model"""
    try:
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        with st.spinner("🔄 Loading RoBERTa model (first time: ~1 min)..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()

        st.success("✅ RoBERTa model loaded successfully!")
        return tokenizer, model, device

    except Exception as e:
        st.error(f"❌ Error loading RoBERTa: {str(e)}")
        return None, None, None


def roberta_sentiment_single(text, tokenizer, model, device):
    """Analyze single text using RoBERTa model"""
    if pd.isna(text) or text == "":
        return 'neutral', 0.0

    try:
        inputs = tokenizer(str(text), return_tensors="pt", truncation=True,
                           max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            scores = torch.softmax(outputs.logits, dim=1)
            confidence, predicted = torch.max(scores, dim=1)

        labels = ['negative', 'neutral', 'positive']
        sentiment = labels[predicted.item()]
        conf = confidence.item()

        return sentiment, conf
    except Exception:
        return 'neutral', 0.0


def roberta_sequential(texts, tokenizer, model, device, show_progress=False):
    """Sequential RoBERTa sentiment analysis"""
    results = []
    total = len(texts)

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    for i, text in enumerate(texts):
        sentiment, conf = roberta_sentiment_single(text, tokenizer, model, device)
        results.append((sentiment, conf))

        if show_progress and i % 10 == 0:
            progress_bar.progress((i + 1) / total)
            status_text.text(f"RoBERTa Sequential: {i+1}/{total}")

    if show_progress:
        progress_bar.progress(1.0)
        status_text.text(f"✅ RoBERTa Sequential: {total} analyzed")

    return results


def roberta_parallel(texts, tokenizer, model, device, max_workers=4, show_progress=False):
    """Parallel RoBERTa sentiment analysis (Thread-safe)"""
    results = []
    total = len(texts)

    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(lambda t: roberta_sentiment_single(t, tokenizer, model, device), texts)
        for i, result in enumerate(futures):
            results.append(result)
            if show_progress and i % 10 == 0:
                progress_bar.progress((i + 1) / total)
                status_text.text(f"RoBERTa Parallel: {i+1}/{total}")

    if show_progress:
        progress_bar.progress(1.0)
        status_text.text(f"✅ RoBERTa Parallel: {total} analyzed")

    return results


# ============================================================
# UNIFIED RUN FUNCTION
# ============================================================

def run_complete_analysis(texts, show_progress=True):
    """
    Run sentiment analysis for both TextBlob and RoBERTa
    Returns dictionary with results + timing
    """
    results = {}

    # ---------- TextBlob Sequential ----------
    if show_progress:
        st.info("🧠 Running TextBlob Sequential...")
    start_time = time.time()
    tb_seq_results = textblob_sequential(texts, show_progress)
    results['textblob_sequential'] = {
        'results': tb_seq_results,
        'time': time.time() - start_time
    }

    # ---------- TextBlob Parallel ----------
    if show_progress:
        st.info("⚙️ Running TextBlob Parallel...")
    start_time = time.time()
    tb_par_results = textblob_parallel(texts, show_progress=True)
    results['textblob_parallel'] = {
        'results': tb_par_results,
        'time': time.time() - start_time
    }

    # ---------- Load RoBERTa ----------
    tokenizer, model, device = load_roberta_model()
    if tokenizer is None:
        return results  # Skip if model not available

    # ---------- RoBERTa Sequential ----------
    if show_progress:
        st.info("🤖 Running RoBERTa Sequential...")
    start_time = time.time()
    rob_seq_results = roberta_sequential(texts, tokenizer, model, device, show_progress=True)
    results['roberta_sequential'] = {
        'results': rob_seq_results,
        'time': time.time() - start_time
    }

    # ---------- RoBERTa Parallel ----------
    if show_progress:
        st.info("🚀 Running RoBERTa Parallel...")

    start_time = time.time()
    rob_par_results = roberta_parallel(
        texts, tokenizer, model, device, show_progress=True
    )
    results['roberta_parallel'] = {
        'results': rob_par_results,
        'time': time.time() - start_time
    }
