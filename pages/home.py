"""
Home Page
Overview and introduction
"""

import streamlit as st

def show():
    """Display home page"""
    st.session_state.current_step = 1
    
    # Hero section
    st.markdown("""
    <div class="main-header">
        <h1>🎭 Welcome to Sentiment Analysis Suite</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            AI-Powered Text Analytics with TextBlob & RoBERTa
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🚀 What This App Does
        
        This powerful sentiment analysis tool allows you to:
        
        - 📤 **Upload CSV files** or **enter text manually**
        - 🧹 **Clean and preprocess** text automatically
        - 🤖 **Analyze sentiment** using two state-of-the-art models
        - ⚡ **Compare performance** (Sequential vs Parallel)
        - 📊 **Visualize results** with interactive charts
        - 📧 **Receive results via email** automatically
        
        ### 🎯 Key Features
        
        - **Dual Model Analysis**: TextBlob (rule-based) + RoBERTa (deep learning)
        - **Performance Optimization**: Sequential and parallel processing
        - **Comprehensive Reports**: Confusion matrices, accuracy metrics
        - **Automated Workflow**: Set it and forget it!
        - **Beautiful Visualizations**: Interactive Plotly charts
        """)
    
    with col2:
        st.markdown("""
        ### 📋 How to Use
        
        Follow these simple steps:
        
        1. **Upload Data** 📤
           - Upload a CSV file, or
           - Enter text manually
        
        2. **Clean Text** 🧹
           - Automatic text preprocessing
           - Remove noise, stopwords
        
        3. **Analyze** 🤖
           - Run TextBlob & RoBERTa
           - Sequential & Parallel modes
        
        4. **Review Results** 📊
           - View interactive charts
           - Compare model performance
        
        5. **Get Results** 📧
           - Download CSV files
           - Receive via email
        
        ### 💡 Quick Start
        
        Click **📤 Upload Data** in the sidebar to begin!
        """)
    
    # Feature cards
    st.markdown("---")
    st.markdown("### ✨ Model Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #667eea;">
            <h3>📚 TextBlob</h3>
            <p><strong>Type:</strong> Rule-based (Lexicon)</p>
            <p><strong>Speed:</strong> Very Fast ⚡</p>
            <p><strong>Accuracy:</strong> Good for general text</p>
            <p><strong>Best for:</strong> Quick analysis, high volume</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left-color: #764ba2;">
            <h3>🧠 RoBERTa</h3>
            <p><strong>Type:</strong> Deep Learning (Transformer)</p>
            <p><strong>Speed:</strong> Moderate 🔄</p>
            <p><strong>Accuracy:</strong> State-of-the-art</p>
            <p><strong>Best for:</strong> Nuanced sentiment, complex text</p>
        </div>
        """, unsafe_allow_html=True)
    
    # System info
    st.markdown("---")
    st.markdown("### ⚙️ System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Framework:** Streamlit")
    with col2:
        st.info("**ML Libraries:** Transformers, NLTK")
    with col3:
        st.info("**Database:** SQLite")
    
    # Call to action
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;">
        <h2>Ready to Get Started?</h2>
        <p style="font-size: 1.1rem;">Navigate to <strong>📤 Upload Data</strong> in the sidebar to begin your analysis!</p>
    </div>
    """, unsafe_allow_html=True)