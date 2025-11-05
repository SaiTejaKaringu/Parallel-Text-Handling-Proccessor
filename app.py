"""
Modern Sentiment Analysis Suite
Multi-page Streamlit Application
Author: Your Name
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import re
import time
import pickle
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Suite",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Step indicator */
    .step-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 8px;
    }
    
    .step {
        flex: 1;
        text-align: center;
        padding: 0.5rem;
        border-radius: 5px;
        transition: all 0.3s;
    }
    
    .step.active {
        background: #667eea;
        color: white;
        font-weight: bold;
    }
    
    .step.completed {
        background: #10b981;
        color: white;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Success message styling */
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        background-color: #f3f4f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_email' not in st.session_state:
        st.session_state.user_email = None
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'cleaned_df' not in st.session_state:
        st.session_state.cleaned_df = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    if 'input_mode' not in st.session_state:
        st.session_state.input_mode = 'upload'
    if 'text_column' not in st.session_state:
        st.session_state.text_column = None
    if 'num_rows' not in st.session_state:
        st.session_state.num_rows = 100

init_session_state()

# Database initialization
def init_db():
    """Initialize SQLite database"""
    conn = sqlite3.connect('sentiment_analysis.db')
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, email TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS raw_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, 
                  upload_time TEXT, row_count INTEGER, data BLOB)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS processed_data
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, 
                  process_time TEXT, method TEXT, data BLOB)''')
    
    conn.commit()
    conn.close()

init_db()

# Authentication functions
def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password, email):
    """Register new user"""
    try:
        conn = sqlite3.connect('sentiment_analysis.db')
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", 
                  (username, hash_password(password), email))
        conn.commit()
        conn.close()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_user(username, password):
    """Authenticate user"""
    try:
        conn = sqlite3.connect('sentiment_analysis.db')
        c = conn.cursor()
        c.execute("SELECT password, email FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()
        
        if result and result[0] == hash_password(password):
            return True, result[1]
        return False, None
    except Exception as e:
        return False, None

# Step indicator component
def show_step_indicator(current_step):
    """Display step progress indicator"""
    steps = [
        "🏠 Home",
        "📤 Upload",
        "🧹 Clean",
        "🤖 Analyze",
        "📊 Results",
        "📧 Email"
    ]
    
    html = '<div class="step-indicator">'
    for i, step in enumerate(steps, 1):
        if i < current_step:
            status = "completed"
        elif i == current_step:
            status = "active"
        else:
            status = ""
        html += f'<div class="step {status}">{step}</div>'
    html += '</div>'
    
    st.markdown(html, unsafe_allow_html=True)

# Login page
def show_login_page():
    """Display login and registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="main-header">
            <h1>🎭 Sentiment Analysis Suite</h1>
            <p>Advanced Text Analytics with AI-Powered Models</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔑 Login", "✨ Register"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            username = st.text_input("👤 Username", key="login_username")
            password = st.text_input("🔒 Password", type="password", key="login_password")
            
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Login", key="login_button", use_container_width=True):
                    if username and password:
                        success, email = login_user(username, password)
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.username = username
                            st.session_state.user_email = email
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid credentials")
                    else:
                        st.warning("⚠️ Please enter credentials")
        
        with tab2:
            st.markdown("### Create New Account")
            new_username = st.text_input("👤 Username", key="reg_username")
            new_email = st.text_input("📧 Email", key="reg_email")
            new_password = st.text_input("🔒 Password", type="password", key="reg_password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password", key="reg_confirm")
            
            if st.button("Register", key="register_button", use_container_width=True):
                if all([new_username, new_email, new_password, confirm_password]):
                    if new_password == confirm_password:
                        if '@' in new_email and '.' in new_email:
                            success, message = register_user(new_username, new_password, new_email)
                            if success:
                                st.success(f"✅ {message}")
                                st.info("Please login with your credentials")
                            else:
                                st.error(f"❌ {message}")
                        else:
                            st.error("❌ Invalid email address")
                    else:
                        st.error("❌ Passwords don't match")
                else:
                    st.warning("⚠️ Please fill all fields")

# Main app entry point
def main():
    """Main application entry point"""
    if not st.session_state.logged_in:
        show_login_page()
    else:
        # Import pages only after login
        from pages import home, upload, cleaning, analysis, results, email_results
        
        # Sidebar navigation
        st.sidebar.markdown(f"""
        <div class="metric-card">
            <h3>👤 User Profile</h3>
            <p><strong>{st.session_state.username}</strong></p>
            <p>📧 {st.session_state.user_email}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.sidebar.markdown("---")
        
        # Page navigation
        page = st.sidebar.radio(
            "Navigate",
            ["🏠 Home", "📤 Upload Data", "🧹 Data Cleaning", 
             "🤖 Sentiment Analysis", "📊 Results & Comparison", "📧 Email Results"],
            label_visibility="collapsed"
        )
        
        # Show step indicator
        show_step_indicator(st.session_state.current_step)
        
        # Route to pages
        if page == "🏠 Home":
            home.show()
        elif page == "📤 Upload Data":
            upload.show()
        elif page == "🧹 Data Cleaning":
            cleaning.show()
        elif page == "🤖 Sentiment Analysis":
            analysis.show()
        elif page == "📊 Results & Comparison":
            results.show()
        elif page == "📧 Email Results":
            email_results.show()
        
        # Logout button
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            

if __name__ == "__main__":
    main()

