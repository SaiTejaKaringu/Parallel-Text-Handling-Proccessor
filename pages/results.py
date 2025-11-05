
"""
Results & Comparison Page
Visualize and compare model results
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import (
    plot_sentiment_distribution,
    plot_confusion_matrix,
    plot_model_comparison,
    create_metrics_cards,
    plot_polarity_distribution,
    plot_confidence_distribution
)
from sklearn.metrics import confusion_matrix

def show():
    """Display results and comparison page"""
    st.session_state.current_step = 5
    
    st.markdown("""
    <div class="main-header">
        <h1>📊 Results & Comparison</h1>
        <p>Detailed analysis results and model comparison</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if results exist
    if st.session_state.results_df is None:
        st.warning("⚠️ No analysis results found. Please run sentiment analysis first.")
        if st.button("🤖 Go to Sentiment Analysis"):
            st.session_state.current_step = 4
            st.rerun()
        return
    
    df = st.session_state.results_df
    timing = st.session_state.get('timing', {})
    
    # Check which models were run
    has_textblob = 'textblob_sentiment_par' in df.columns
    has_roberta = 'roberta_sentiment_par' in df.columns
    
    # Overview metrics
    st.markdown("### 📈 Overview")
    
    metrics = {
        "Total Texts": f"{len(df):,}",
        "Models Used": f"{int(has_textblob) + int(has_roberta)}",
    }
    
    if timing:
        total_time = sum(timing.values())
        metrics["Total Time"] = f"{total_time:.2f}s"
        metrics["Avg Time/Text"] = f"{(total_time/len(df)):.3f}s"
    
    st.markdown(create_metrics_cards(metrics), unsafe_allow_html=True)
    
    # Tabs for different views
    tabs = []
    if has_textblob:
        tabs.append("📚 TextBlob Results")
    if has_roberta:
        tabs.append("🧠 RoBERTa Results")
    if has_textblob and has_roberta:
        tabs.append("⚖️ Model Comparison")
    
    tab_objects = st.tabs(tabs)
    tab_idx = 0
    
    # TextBlob Results
    if has_textblob:
        with tab_objects[tab_idx]:
            show_textblob_results(df, timing)
        tab_idx += 1
    
    # RoBERTa Results
    if has_roberta:
        with tab_objects[tab_idx]:
            show_roberta_results(df, timing)
        tab_idx += 1
    
    # Model Comparison
    if has_textblob and has_roberta:
        with tab_objects[tab_idx]:
            show_model_comparison(df)
    
    # Next step
    st.markdown("---")
    if st.button("➡️ Email Results", type="primary", use_container_width=True):
        st.info("Navigate to **📧 Email Results** in the sidebar")

def show_textblob_results(df, timing):
    """Show TextBlob analysis results"""
    
    st.markdown("### 📚 TextBlob Analysis Results")
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sentiment Distribution (Sequential)")
        counts_seq = df['textblob_sentiment_seq'].value_counts()
        for sentiment, count in counts_seq.items():
            percentage = (count / len(df)) * 100
            st.metric(sentiment.capitalize(), f"{count}", f"{percentage:.1f}%")
    
    with col2:
        st.markdown("#### Sentiment Distribution (Parallel)")
        counts_par = df['textblob_sentiment_par'].value_counts()
        for sentiment, count in counts_par.items():
            percentage = (count / len(df)) * 100
            st.metric(sentiment.capitalize(), f"{count}", f"{percentage:.1f}%")
    
    # Visualizations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_sentiment_distribution(df, 'textblob_sentiment_seq', 
                                         'TextBlob Sequential Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_sentiment_distribution(df, 'textblob_sentiment_par', 
                                         'TextBlob Parallel Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Polarity distribution
    st.markdown("---")
    st.markdown("#### Polarity Score Distribution")
    
    fig = plot_polarity_distribution(df['textblob_polarity_par'].tolist())
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("#### Polarity Statistics")
    
    polarity_stats = df['textblob_polarity_par'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{polarity_stats['mean']:.3f}")
    with col2:
        st.metric("Std Dev", f"{polarity_stats['std']:.3f}")
    with col3:
        st.metric("Min", f"{polarity_stats['min']:.3f}")
    with col4:
        st.metric("Max", f"{polarity_stats['max']:.3f}")
    
    # Timing
    if timing:
        st.markdown("---")
        st.markdown("#### Processing Time")
        
        if 'TextBlob Sequential' in timing and 'TextBlob Parallel' in timing:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sequential", f"{timing['TextBlob Sequential']:.2f}s")
            with col2:
                st.metric("Parallel", f"{timing['TextBlob Parallel']:.2f}s")
            with col3:
                speedup = timing['TextBlob Sequential'] / timing['TextBlob Parallel']
                st.metric("Speedup", f"{speedup:.2f}x")

def show_roberta_results(df, timing):
    """Show RoBERTa analysis results"""
    
    st.markdown("### 🧠 RoBERTa Analysis Results")
    
    # Sentiment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Sentiment Distribution (Sequential)")
        counts_seq = df['roberta_sentiment_seq'].value_counts()
        for sentiment, count in counts_seq.items():
            percentage = (count / len(df)) * 100
            st.metric(sentiment.capitalize(), f"{count}", f"{percentage:.1f}%")
    
    with col2:
        st.markdown("#### Sentiment Distribution (Parallel)")
        counts_par = df['roberta_sentiment_par'].value_counts()
        for sentiment, count in counts_par.items():
            percentage = (count / len(df)) * 100
            st.metric(sentiment.capitalize(), f"{count}", f"{percentage:.1f}%")
    
    # Visualizations
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_sentiment_distribution(df, 'roberta_sentiment_seq', 
                                         'RoBERTa Sequential Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_sentiment_distribution(df, 'roberta_sentiment_par', 
                                         'RoBERTa Parallel Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Confidence distribution
    st.markdown("---")
    st.markdown("#### Confidence Score Distribution")
    
    fig = plot_confidence_distribution(df['roberta_confidence_par'].tolist())
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("---")
    st.markdown("#### Confidence Statistics")
    
    conf_stats = df['roberta_confidence_par'].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean", f"{conf_stats['mean']:.3f}")
    with col2:
        st.metric("Std Dev", f"{conf_stats['std']:.3f}")
    with col3:
        st.metric("Min", f"{conf_stats['min']:.3f}")
    with col4:
        st.metric("Max", f"{conf_stats['max']:.3f}")
    
    # Timing
    if timing:
        st.markdown("---")
        st.markdown("#### Processing Time")
        
        if 'RoBERTa Sequential' in timing and 'RoBERTa Parallel' in timing:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sequential", f"{timing['RoBERTa Sequential']:.2f}s")
            with col2:
                st.metric("Parallel", f"{timing['RoBERTa Parallel']:.2f}s")
            with col3:
                speedup = timing['RoBERTa Sequential'] / timing['RoBERTa Parallel']
                st.metric("Speedup", f"{speedup:.2f}x")

def show_model_comparison(df):
    """Show comparison between TextBlob and RoBERTa"""
    
    st.markdown("### ⚖️ Model Comparison")
    
    # Agreement metrics
    agreement = (df['textblob_sentiment_par'] == df['roberta_sentiment_par']).sum()
    agreement_rate = (agreement / len(df)) * 100
    
    st.markdown("#### Agreement Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
    with col2:
        st.metric("Agreements", f"{agreement:,}")
    with col3:
        st.metric("Disagreements", f"{len(df) - agreement:,}")
    
    # Sentiment distribution comparison
    st.markdown("---")
    st.markdown("#### Sentiment Distribution Comparison")
    
    textblob_counts = df['textblob_sentiment_par'].value_counts()
    roberta_counts = df['roberta_sentiment_par'].value_counts()
    
    # Ensure all labels are present
    labels = ['positive', 'neutral', 'negative']
    tb_dist = [textblob_counts.get(label, 0) for label in labels]
    rob_dist = [roberta_counts.get(label, 0) for label in labels]
    
    fig = plot_model_comparison(tb_dist, rob_dist)
    st.plotly_chart(fig, use_container_width=True)
    
    # Confusion matrix
    st.markdown("---")
    st.markdown("#### Confusion Matrix: TextBlob vs RoBERTa")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        fig = plot_confusion_matrix(
            df['textblob_sentiment_par'],
            df['roberta_sentiment_par'],
            labels,
            'Confusion Matrix (Counts)',
            normalize=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_confusion_matrix(
            df['textblob_sentiment_par'],
            df['roberta_sentiment_par'],
            labels,
            'Confusion Matrix (Normalized)',
            normalize=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Disagreement analysis
    st.markdown("---")
    st.markdown("#### Disagreement Examples")
    
    disagreements = df[df['textblob_sentiment_par'] != df['roberta_sentiment_par']]
    
    if len(disagreements) > 0:
        st.info(f"Found {len(disagreements)} disagreements. Showing first 10:")
        
        display_cols = [
            st.session_state.text_column,
            'textblob_sentiment_par',
            'roberta_sentiment_par'
        ]
        
        st.dataframe(disagreements[display_cols].head(10), use_container_width=True)
        
        # Download disagreements
        csv = disagreements.to_csv(index=False)
        st.download_button(
            label="📥 Download All Disagreements",
            data=csv,
            file_name="disagreements.csv",
            mime="text/csv"
        )
    else:
        st.success("✅ Perfect agreement! Both models produced identical results.")
    
    # Save comparison
    st.markdown("---")
    comparison_df = pd.DataFrame({
        'original_text': df[st.session_state.text_column],
        'textblob_sentiment': df['textblob_sentiment_par'],
        'textblob_polarity': df['textblob_polarity_par'],
        'roberta_sentiment': df['roberta_sentiment_par'],
        'roberta_confidence': df['roberta_confidence_par'],
        'agreement': df['textblob_sentiment_par'] == df['roberta_sentiment_par']
    })
    
    comparison_df.to_csv('comparison_textblob_vs_roberta.csv', index=False)
    
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Comparison Report",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv",
        use_container_width=True
    )