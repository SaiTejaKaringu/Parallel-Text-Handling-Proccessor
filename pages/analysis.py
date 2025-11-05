"""
Sentiment Analysis Page
Run TextBlob and RoBERTa analysis
"""

import streamlit as st
import pandas as pd
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.sentiment_models import (
    textblob_sequential, textblob_parallel,
    roberta_sequential, roberta_parallel,
    load_roberta_model
)
from utils.visualization import plot_sentiment_distribution, plot_time_comparison

def show():
    """Display sentiment analysis page"""
    st.session_state.current_step = 4
    
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Sentiment Analysis</h1>
        <p>Analyze sentiment using TextBlob and RoBERTa models</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if cleaned data exists
    if st.session_state.cleaned_df is None:
        st.warning("⚠️ No cleaned data found. Please clean your data first.")
        if st.button("🧹 Go to Data Cleaning"):
            st.session_state.current_step = 3
            st.rerun()
        return
    
    df = st.session_state.cleaned_df
    text_column = st.session_state.text_column
    
    # Display info
    st.markdown("""
    <div class="info-box">
        <strong>📊 Analysis Configuration:</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Texts to Analyze", f"{len(df):,}")
    with col2:
        st.metric("Source Column", text_column)
    
    st.markdown("---")
    
    # Model information
    st.markdown("### 🎯 Available Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📚 TextBlob</h3>
            <p><strong>Type:</strong> Lexicon-based</p>
            <p><strong>Output:</strong> Sentiment + Polarity Score</p>
            <p><strong>Speed:</strong> Very Fast ⚡</p>
            <p><strong>Processing Modes:</strong> Sequential & Parallel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 RoBERTa</h3>
            <p><strong>Type:</strong> Transformer (Deep Learning)</p>
            <p><strong>Output:</strong> Sentiment + Confidence Score</p>
            <p><strong>Speed:</strong> Moderate 🔄</p>
            <p><strong>Processing Modes:</strong> Sequential & Parallel</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Analysis options
    st.markdown("---")
    st.markdown("### ⚙️ Analysis Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        run_textblob = st.checkbox("Run TextBlob", value=True)
    with col2:
        run_roberta = st.checkbox("Run RoBERTa", value=True)
    with col3:
        save_results = st.checkbox("Save results to CSV", value=True)
    
    # Start analysis button
    st.markdown("---")
    
    if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
        if not run_textblob and not run_roberta:
            st.error("❌ Please select at least one model")
        else:
            run_analysis(df, text_column, run_textblob, run_roberta, save_results)

def run_analysis(df, text_column, run_textblob, run_roberta, save_results):
    """Execute sentiment analysis"""
    
    results = {}
    timing = {}
    
    # Get texts
    texts = df[text_column].tolist()
    
    st.markdown("---")
    st.markdown("### 🔄 Analysis in Progress")
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        # TextBlob Analysis
        if run_textblob:
            st.markdown("#### 📚 TextBlob Analysis")
            
            # Sequential
            st.info("🔄 Running TextBlob Sequential...")
            start_time = time.time()
            tb_seq_results = textblob_sequential(texts, show_progress=True)
            timing['TextBlob Sequential'] = time.time() - start_time
            
            df['textblob_sentiment_seq'] = [r[0] for r in tb_seq_results]
            df['textblob_polarity_seq'] = [r[1] for r in tb_seq_results]
            
            st.success(f"✅ TextBlob Sequential completed in {timing['TextBlob Sequential']:.2f}s")
            
            # Parallel
            st.info("🔄 Running TextBlob Parallel...")
            start_time = time.time()
            tb_par_results = textblob_parallel(texts)
            timing['TextBlob Parallel'] = time.time() - start_time
            
            df['textblob_sentiment_par'] = [r[0] for r in tb_par_results]
            df['textblob_polarity_par'] = [r[1] for r in tb_par_results]
            
            st.success(f"✅ TextBlob Parallel completed in {timing['TextBlob Parallel']:.2f}s")
            
            # Save TextBlob results
            if save_results:
                df.to_csv('textblob_results_sequential.csv', index=False)
                df.to_csv('textblob_results_parallel.csv', index=False)
            
            # Show distribution
            fig = plot_sentiment_distribution(df, 'textblob_sentiment_par', 
                                            'TextBlob Sentiment Distribution')
            st.plotly_chart(fig, use_container_width=True)
        
        # RoBERTa Analysis
        if run_roberta:
            st.markdown("---")
            st.markdown("#### 🧠 RoBERTa Analysis")

            # Load model
            tokenizer, model, device = load_roberta_model()

            if tokenizer is None or model is None:
                st.error("❌ Failed to load RoBERTa model. Skipping RoBERTa analysis.")
            else:
                # Sequential
                st.info("🔄 Running RoBERTa Sequential...")
                start_time = time.time()
                rob_seq_results = roberta_sequential(
                    texts, tokenizer, model, device, show_progress=True
                )
                timing['RoBERTa Sequential'] = time.time() - start_time

                df['roberta_sentiment_seq'] = [r[0] for r in rob_seq_results]
                df['roberta_confidence_seq'] = [r[1] for r in rob_seq_results]

                st.success(f"✅ RoBERTa Sequential completed in {timing['RoBERTa Sequential']:.2f}s")

                # ✅ FIXED: Parallel call now passes all required arguments
                st.info("🔄 Running RoBERTa Parallel...")
                start_time = time.time()
                rob_par_results = roberta_parallel(
                    texts, tokenizer, model, device, show_progress=True
                )
                timing['RoBERTa Parallel'] = time.time() - start_time

                df['roberta_sentiment_par'] = [r[0] for r in rob_par_results]
                df['roberta_confidence_par'] = [r[1] for r in rob_par_results]

                st.success(f"✅ RoBERTa Parallel completed in {timing['RoBERTa Parallel']:.2f}s")

                # Save RoBERTa results
                if save_results:
                    df.to_csv('roberta_results_sequential.csv', index=False)
                    df.to_csv('roberta_results_parallel.csv', index=False)

                # Show distribution
                fig = plot_sentiment_distribution(
                    df, 'roberta_sentiment_par', 'RoBERTa Sentiment Distribution'
                )
                st.plotly_chart(fig, use_container_width=True)

    
    # Store results
    st.session_state.results_df = df
    st.session_state.timing = timing
    
    # Show timing comparison
    st.markdown("---")
    st.markdown("### ⏱️ Processing Time Comparison")
    
    if timing:
        fig = plot_time_comparison(timing)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show speedup
        if run_textblob and 'TextBlob Sequential' in timing and 'TextBlob Parallel' in timing:
            speedup = timing['TextBlob Sequential'] / timing['TextBlob Parallel']
            st.info(f"📊 **TextBlob Speedup:** {speedup:.2f}x faster with parallel processing")
        
        if run_roberta and 'RoBERTa Sequential' in timing and 'RoBERTa Parallel' in timing:
            speedup = timing['RoBERTa Sequential'] / timing['RoBERTa Parallel']
            st.info(f"📊 **RoBERTa Speedup:** {speedup:.2f}x faster with parallel processing")
    
    # Results preview
    st.markdown("---")
    st.markdown("### 📊 Results Preview")
    
    display_cols = [text_column]
    if run_textblob:
        display_cols.extend(['textblob_sentiment_par', 'textblob_polarity_par'])
    if run_roberta:
        display_cols.extend(['roberta_sentiment_par', 'roberta_confidence_par'])
    
    st.dataframe(df[display_cols].head(10), use_container_width=True)
    
    # Download results
    if save_results:
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Results",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Next step
    st.markdown("---")
    st.success("✅ Analysis complete! View detailed results and comparisons.")
    st.balloons()
    
    if st.button("➡️ View Results & Comparison", type="primary", use_container_width=True):
        st.info("Navigate to **📊 Results & Comparison** in the sidebar")