"""
Email Results Page
Send analysis results via email
"""

import streamlit as st
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.email_service import (
    send_results_email,
    validate_email_config,
    get_email_setup_instructions
)

def show():
    """Display email results page"""
    st.session_state.current_step = 6
    
    st.markdown("""
    <div class="main-header">
        <h1>📧 Email Results</h1>
        <p>Send analysis results to your email</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if results exist
    if st.session_state.results_df is None:
        st.warning("⚠️ No analysis results found. Please complete the analysis first.")
        if st.button("🤖 Go to Sentiment Analysis"):
            st.session_state.current_step = 4
            st.rerun()
        return
    
    # Validate email configuration
    is_valid, message = validate_email_config()
    
    if not is_valid:
        st.markdown(get_email_setup_instructions(), unsafe_allow_html=True)
        st.markdown("---")
        st.info("After configuring email, restart the app for changes to take effect.")
        return
    
    # Show email configuration status
    st.success("✅ Email configuration is valid!")
    
    # Display user info
    st.markdown("""
    <div class="info-box">
        <strong>📬 Email will be sent to:</strong>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Recipient", st.session_state.user_email)
    with col2:
        st.metric("Username", st.session_state.username)
    
    # Files to be sent
    st.markdown("---")
    st.markdown("### 📎 Files to be Attached")
    
    files = []
    file_info = []
    
    # Check which files exist
    potential_files = [
        ('textblob_results_sequential.csv', 'TextBlob Sequential Results'),
        ('textblob_results_parallel.csv', 'TextBlob Parallel Results'),
        ('roberta_results_sequential.csv', 'RoBERTa Sequential Results'),
        ('roberta_results_parallel.csv', 'RoBERTa Parallel Results'),
        ('comparison_textblob_vs_roberta.csv', 'Model Comparison Report')
    ]
    
    for filename, description in potential_files:
        if os.path.exists(filename):
            files.append(filename)
            size = os.path.getsize(filename) / 1024  # KB
            file_info.append({
                'File': description,
                'Filename': filename,
                'Size': f"{size:.2f} KB"
            })
    
    if file_info:
        st.table(pd.DataFrame(file_info))
        st.info(f"📊 Total files to send: **{len(files)}**")
    else:
        st.error("❌ No result files found. Please run the analysis first.")
        return
    
    # Email preview
    st.markdown("---")
    st.markdown("### 📧 Email Preview")
    
    with st.expander("View email content"):
        st.markdown(f"""
        **Subject:** Sentiment Analysis Results
        
        **To:** {st.session_state.user_email}
        
        **Body:**
        ```
        Hello {st.session_state.username},

        Your automated sentiment analysis has been completed successfully! 🎉

        This email contains the following results:
        ✓ TextBlob Sequential Analysis
        ✓ TextBlob Parallel Analysis  
        ✓ RoBERTa Sequential Analysis
        ✓ RoBERTa Parallel Analysis
        ✓ Comparison Report (TextBlob vs RoBERTa)

        All results are attached as CSV files for your convenience.

        Best regards,
        Automated Sentiment Analysis System
        ```
        
        **Attachments:** {len(files)} files
        """)
    
    # Send email options
    st.markdown("---")
    st.markdown("### ⚙️ Email Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        send_to_custom = st.checkbox("Send to different email", value=False)
    
    with col2:
        include_summary = st.checkbox("Include analysis summary", value=True)
    
    # Custom email input
    recipient_email = st.session_state.user_email
    
    if send_to_custom:
        custom_email = st.text_input(
            "Enter email address:",
            placeholder="example@email.com"
        )
        if custom_email:
            recipient_email = custom_email
    
    # Send button
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("📧 Send Email Now", type="primary", use_container_width=True):
            send_email(recipient_email, files)

def send_email(recipient_email, files):
    """Send email with results"""
    
    # Validate email
    if not recipient_email or '@' not in recipient_email:
        st.error("❌ Please enter a valid email address")
        return
    
    # Show progress
    with st.spinner("📤 Sending email..."):
        success, message = send_results_email(
            recipient_email,
            files,
            st.session_state.username
        )
    
    if success:
        st.success(message)
        st.balloons()
        
        # Show success details
        st.markdown(f"""
        <div class="success-message">
            <h3>✅ Email Sent Successfully!</h3>
            <p>Your results have been sent to: <strong>{recipient_email}</strong></p>
            <p>Please check your inbox (and spam folder if needed).</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional actions
        st.markdown("---")
        st.markdown("### 🎉 What's Next?")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🏠 Return to Home", use_container_width=True):
                st.session_state.current_step = 1
                st.rerun()
        
        with col2:
            if st.button("📤 Upload New Data", use_container_width=True):
                # Clear session state
                st.session_state.df = None
                st.session_state.cleaned_df = None
                st.session_state.results_df = None
                st.session_state.current_step = 2
                st.rerun()
        
        with col3:
            if st.button("📊 View Results Again", use_container_width=True):
                st.session_state.current_step = 5
                st.rerun()
    
    else:
        st.error(message)
        
        # Troubleshooting tips
        with st.expander("❓ Troubleshooting"):
            st.markdown("""
            ### Common Issues:
            
            1. **Authentication Failed**
               - Verify your Gmail and App Password are correct
               - Make sure 2-Step Verification is enabled
               - Generate a new App Password if needed
            
            2. **SMTP Connection Error**
               - Check your internet connection
               - Verify Gmail SMTP is not blocked by firewall
               - Try again in a few minutes
            
            3. **Invalid Email**
               - Make sure the recipient email is valid
               - Check for typos in the email address
            
            4. **Files Not Found**
               - Ensure analysis has been completed
               - Check that CSV files were generated
               - Re-run the analysis if necessary
            
            ### Need Help?
            
            - Check email configuration in `utils/email_service.py`
            - Verify all CSV files exist in the project directory
            - Restart the application after configuration changes
            """)
        
        # Retry button
        st.markdown("---")
        if st.button("🔄 Retry Sending Email", use_container_width=True):
            st.rerun()
    
    # Download alternative
    st.markdown("---")
    st.markdown("### 💾 Alternative: Download Results")
    st.info("If email is not working, you can download the results manually:")
    
    col1, col2 = st.columns(2)
    
    for i, filepath in enumerate(files):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = f.read()
            
            col = col1 if i % 2 == 0 else col2
            col.download_button(
                label=f"📄 {os.path.basename(filepath)}",
                data=data,
                file_name=os.path.basename(filepath),
                mime='text/csv',
                use_container_width=True
            )