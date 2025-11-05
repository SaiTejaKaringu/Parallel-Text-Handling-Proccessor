"""
Upload Page
CSV upload or manual text entry
"""

import streamlit as st
import pandas as pd

def show():
    """Display upload page"""
    st.session_state.current_step = 2
    
    st.markdown("""
    <div class="main-header">
        <h1>📤 Upload Your Data</h1>
        <p>Choose how you want to provide text for analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 Upload CSV File", use_container_width=True, type="primary"):
            st.session_state.input_mode = 'upload'
    
    with col2:
        if st.button("✍️ Enter Text Manually", use_container_width=True):
            st.session_state.input_mode = 'manual'
    
    st.markdown("---")
    
    # CSV Upload Mode
    if st.session_state.input_mode == 'upload':
        show_csv_upload()
    
    # Manual Entry Mode
    elif st.session_state.input_mode == 'manual':
        show_manual_entry()

def show_csv_upload():
    """Show CSV upload interface"""
    st.markdown("### 📁 Upload CSV File")
    
    st.markdown("""
    <div class="info-box">
        <strong>📋 CSV Requirements:</strong>
        <ul>
            <li>File must be in CSV format</li>
            <li>Must contain at least one text column</li>
            <li>UTF-8 encoding recommended</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing text data for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                st.error("❌ Could not decode file. Please check the file encoding.")
                return
            
            # Store in session state
            st.session_state.df = df
            
            st.success(f"✅ File loaded successfully! **{len(df):,}** rows found")
            
            # Dataset preview
            st.markdown("### 📊 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Dataset statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            with col4:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Column selection
            st.markdown("---")
            st.markdown("### ⚙️ Configure Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                text_column = st.selectbox(
                    "📝 Select text column to analyze:",
                    options=df.columns.tolist(),
                    help="Choose the column containing text data"
                )
                st.session_state.text_column = text_column
            
            with col2:
                max_rows = len(df)
                num_rows = st.slider(
                    "📊 Number of rows to process:",
                    min_value=10,
                    max_value=min(max_rows, 10000),
                    value=min(100, max_rows),
                    help="Select how many rows to analyze"
                )
                st.session_state.num_rows = num_rows
            
            # Preview selected data
            if text_column:
                st.markdown(f"### 👁️ Preview of '{text_column}' (First 5 rows)")
                preview_df = df[[text_column]].head(5)
                st.dataframe(preview_df, use_container_width=True)
            
            # Next button
            st.markdown("---")
            if st.button("➡️ Proceed to Data Cleaning", type="primary", use_container_width=True):
                if text_column:
                    st.success("✅ Configuration saved! Navigate to **🧹 Data Cleaning**")
                    st.balloons()
                else:
                    st.error("❌ Please select a text column")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Make sure your file is a valid CSV format")

def show_manual_entry():
    """Show manual text entry interface"""
    st.markdown("### ✍️ Enter Text Manually")
    
    st.markdown("""
    <div class="info-box">
        <strong>📝 Text Entry Options:</strong>
        <ul>
            <li>Enter single or multiple texts</li>
            <li>One text per line for multiple entries</li>
            <li>Supports up to 1000 texts</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Text entry modes
    entry_mode = st.radio(
        "Choose entry mode:",
        ["Single Text", "Multiple Texts (one per line)"],
        horizontal=True
    )
    
    if entry_mode == "Single Text":
        text_input = st.text_area(
            "Enter your text:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        if text_input:
            # Create dataframe
            df = pd.DataFrame({'text': [text_input]})
            st.session_state.df = df
            st.session_state.text_column = 'text'
            st.session_state.num_rows = 1
            
            st.success("✅ Text loaded successfully!")
            
            # Show preview
            st.markdown("### 📊 Preview")
            st.info(f"**Character count:** {len(text_input)} | **Word count:** {len(text_input.split())}")
    
    else:  # Multiple Texts
        text_input = st.text_area(
            "Enter multiple texts (one per line):",
            height=300,
            placeholder="Text 1\nText 2\nText 3\n..."
        )
        
        if text_input:
            # Split by lines and filter empty lines
            texts = [line.strip() for line in text_input.split('\n') if line.strip()]
            
            if len(texts) > 1000:
                st.warning("⚠️ Maximum 1000 texts allowed. Only the first 1000 will be processed.")
                texts = texts[:1000]
            
            # Create dataframe
            df = pd.DataFrame({'text': texts})
            st.session_state.df = df
            st.session_state.text_column = 'text'
            st.session_state.num_rows = len(texts)
            
            st.success(f"✅ {len(texts)} texts loaded successfully!")
            
            # Show preview
            st.markdown("### 📊 Preview (First 10)")
            st.dataframe(df.head(10), use_container_width=True)
    
    # Next button
    if st.session_state.df is not None:
        st.markdown("---")
        if st.button("➡️ Proceed to Data Cleaning", type="primary", use_container_width=True):
            st.success("✅ Text saved! Navigate to **🧹 Data Cleaning**")
            st.balloons()