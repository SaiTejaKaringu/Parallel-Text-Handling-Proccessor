"""
Data Cleaning Page
Text preprocessing and cleaning
"""

import streamlit as st
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_processing import batch_clean_texts, get_top_words, get_text_statistics
from utils.visualization import plot_top_words, create_metrics_cards

st.markdown("""
<style>
.metric-container { display: flex; gap: 1rem; flex-wrap: wrap; }
.metric-card { transition: all 0.2s; }
</style>
""", unsafe_allow_html=True)

def show():
    """Display data cleaning page"""
    st.session_state.current_step = 3
    
    st.markdown("""
    <div class="main-header">
        <h1>🧹 Data Cleaning & Preprocessing</h1>
        <p>Clean and prepare your text data for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if data exists
    if st.session_state.df is None:
        st.warning("⚠️ No data loaded. Please upload data first.")
        if st.button("📤 Go to Upload Page"):
            st.session_state.current_step = 2
            st.rerun()
        return
    
    df = st.session_state.df
    text_column = st.session_state.text_column
    num_rows = st.session_state.num_rows
    
    # Display current configuration
    st.markdown("""
    <div class="info-box">
        <strong>📊 Current Configuration:</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Selected Column", text_column)
    with col3:
        st.metric("Rows to Process", f"{num_rows:,}")
    
    st.markdown("---")
    
    # Cleaning options
    st.markdown("### ⚙️ Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Automatic cleaning includes:**
        - ✅ Remove URLs
        - ✅ Remove special characters
        - ✅ Remove emojis
        - ✅ Remove mentions and hashtags
        - ✅ Convert to lowercase
        - ✅ Remove stopwords
        - ✅ Remove extra whitespace
        """)
    
    with col2:
        show_preview = st.checkbox("Show before/after preview", value=True)
        use_parallel = st.checkbox("Use parallel processing (faster)", value=True)
    
    # Start cleaning button
    st.markdown("---")
    
    if st.button("🚀 Start Cleaning", type="primary", use_container_width=True):
        clean_data(df, text_column, num_rows, show_preview, use_parallel)

def clean_data(df, text_column, num_rows, show_preview, use_parallel):
    """Execute data cleaning"""
    
    # Get subset
    df_subset = df.head(num_rows).copy()
    
    # Show original statistics
    st.markdown("### 📊 Original Text Statistics")
    original_texts = df_subset[text_column].tolist()
    original_stats = get_text_statistics(original_texts)
    
    stats_html = create_metrics_cards({
        "Total Texts": f"{original_stats['total_texts']:,}",
        "Avg Words": f"{original_stats['avg_words']:.1f}",
        "Avg Characters": f"{original_stats['avg_chars']:.1f}",
        "Total Words": f"{original_stats['total_words']:,}"
    })
    st.markdown(stats_html, unsafe_allow_html=True) 
    
    # Preview before cleaning
    if show_preview:
        st.markdown("### 👁️ Before Cleaning (Sample)")
        st.dataframe(df_subset[[text_column]].head(5), use_container_width=True)
    
    # Clean texts
    st.markdown("---")
    st.markdown("### 🔄 Cleaning in Progress...")
    
    with st.spinner("Cleaning text data..."):
        cleaned_texts = batch_clean_texts(original_texts, show_progress=True)
    
    df_subset['cleaned_text'] = cleaned_texts
    
    # Store cleaned data
    st.session_state.cleaned_df = df_subset
    
    st.success("✅ Text cleaning completed!")
    
    # Show cleaned statistics
    st.markdown("---")
    st.markdown("### 📊 Cleaned Text Statistics")
    cleaned_stats = get_text_statistics(cleaned_texts)
    
    stats_html = create_metrics_cards({
        "Total Texts": f"{cleaned_stats['total_texts']:,}",
        "Avg Words": f"{cleaned_stats['avg_words']:.1f}",
        "Avg Characters": f"{cleaned_stats['avg_chars']:.1f}",
        "Total Words": f"{cleaned_stats['total_words']:,}"
    })
    st.markdown(stats_html, unsafe_allow_html=True)
    
    # Preview after cleaning    
    if show_preview:
        st.markdown("### 👁️ After Cleaning (Sample)")
        st.dataframe(df_subset[['cleaned_text']].head(5), use_container_width=True)
    
    # Comparison
    st.markdown("---")
    st.markdown("### 📈 Before vs After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Words Removed:**")
        words_removed = original_stats['total_words'] - cleaned_stats['total_words']
        st.metric("Total", f"{words_removed:,}", f"-{(words_removed/original_stats['total_words']*100):.1f}%")
    
    with col2:
        st.markdown("**Characters Removed:**")
        chars_removed = original_stats['total_chars'] - cleaned_stats['total_chars']
        st.metric("Total", f"{chars_removed:,}", f"-{(chars_removed/original_stats['total_chars']*100):.1f}%")
    
    # Top words visualization
    st.markdown("---")
    st.markdown("### 🔤 Top 20 Most Frequent Words")
    
    with st.spinner("Computing word frequencies..."):
        top_words = get_top_words(cleaned_texts, top_n=20, use_parallel=use_parallel)
    
    fig = plot_top_words(top_words)
    st.plotly_chart(fig, use_container_width=True)
    
    # Word frequency table
    with st.expander("📋 View Word Frequency Table"):
        words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
        st.dataframe(words_df, use_container_width=True)
    
    # Download cleaned data
    st.markdown("---")
    st.markdown("### 💾 Download Cleaned Data")
    
    csv = df_subset.to_csv(index=False)
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Next step
    st.markdown("---")
    st.success("✅ Data cleaning complete! Ready for sentiment analysis.")
    
    if st.button("➡️ Proceed to Sentiment Analysis", type="primary", use_container_width=True):
        st.balloons()
        st.session_state.current_step = 4
        st.success("✅ Redirecting to Sentiment Analysis...")
        st.switch_page("pages/analysis.py")

