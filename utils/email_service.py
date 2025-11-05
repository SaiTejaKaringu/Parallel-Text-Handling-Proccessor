"""
Email Service Utility
Send analysis results via email
"""

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os
from datetime import datetime

# ⚠️ CONFIGURE YOUR EMAIL CREDENTIALS HERE
SENDER_EMAIL = "teja09076@gmail.com"  # Replace with your Gmail
SENDER_APP_PASSWORD = "dvrp ilaf ytvf lklk"  # Replace with Gmail App Password

def send_results_email(recipient_email, files, username):
    """
    Send email with CSV attachments
    
    Args:
        recipient_email: Recipient's email address
        files: List of file paths to attach
        username: Username for personalization
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        # Validate email configuration
        if SENDER_EMAIL == "your_email@gmail.com":
            return False, "⚠️ Please configure SENDER_EMAIL and SENDER_APP_PASSWORD in utils/email_service.py"
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = f"Sentiment Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        # Email body
        body = f"""
Hello {username},

Your automated sentiment analysis has been completed successfully! 🎉

This email contains the following results:
✓ TextBlob Sequential Analysis
✓ TextBlob Parallel Analysis  
✓ RoBERTa Sequential Analysis
✓ RoBERTa Parallel Analysis
✓ Comparison Report (TextBlob vs RoBERTa)

All results are attached as CSV files for your convenience.

Performance Summary:
- Total texts analyzed: {get_file_row_count(files[0]) if files else 'N/A'}
- Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Best regards,
Automated Sentiment Analysis System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach files
        attached_count = 0
        for filepath in files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        part.add_header(
                            'Content-Disposition',
                            f'attachment; filename={os.path.basename(filepath)}'
                        )
                        msg.attach(part)
                        attached_count += 1
                except Exception as e:
                    print(f"Could not attach {filepath}: {str(e)}")
        
        if attached_count == 0:
            return False, "❌ No files were attached to the email"
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True, f"✅ Email sent successfully with {attached_count} attachments!"
    
    except smtplib.SMTPAuthenticationError:
        return False, "❌ Authentication failed. Please check your email and app password."
    except smtplib.SMTPException as e:
        return False, f"❌ SMTP error: {str(e)}"
    except Exception as e:
        return False, f"❌ Error sending email: {str(e)}"

def get_file_row_count(filepath):
    """Get row count from CSV file"""
    try:
        with open(filepath, 'r') as f:
            return len(f.readlines()) - 1  # Exclude header
    except:
        return 'N/A'

def validate_email_config():
    """
    Check if email configuration is valid
    
    Returns:
        Tuple of (is_valid: bool, message: str)
    """
    if SENDER_EMAIL == "your_email@gmail.com" or SENDER_APP_PASSWORD == "your_app_password_here":
        return False, """
        ⚠️ **Email not configured**
        
        Please update the following in `utils/email_service.py`:
        ```python
        SENDER_EMAIL = "your_email@gmail.com"
        SENDER_APP_PASSWORD = "your_app_password_here"
        ```
        
        **How to get Gmail App Password:**
        1. Go to Google Account settings
        2. Navigate to Security
        3. Enable 2-Step Verification
        4. Generate an App Password for "Mail"
        5. Copy the 16-character password
        """
    
    return True, "✅ Email configuration is valid"

def get_email_setup_instructions():
    """Return HTML instructions for email setup"""
    return """
    <div class="info-box" style="background: #eff6ff; border-left: 4px solid #3b82f6; padding: 1.5rem; border-radius: 8px;">
        <h4>📧 Email Configuration Required</h4>
        <p>To enable automatic email delivery of results, please configure your Gmail credentials:</p>
        
        <ol>
            <li><strong>Open</strong> <code>utils/email_service.py</code></li>
            <li><strong>Update</strong> these lines:
                <pre style="background: #1e293b; color: #e2e8f0; padding: 1rem; border-radius: 5px; overflow-x: auto;">
SENDER_EMAIL = "your_email@gmail.com"
SENDER_APP_PASSWORD = "xxxx xxxx xxxx xxxx"</pre>
            </li>
            <li><strong>Get Gmail App Password:</strong>
                <ul>
                    <li>Go to <a href="https://myaccount.google.com/" target="_blank">Google Account</a> → Security</li>
                    <li>Enable <strong>2-Step Verification</strong></li>
                    <li>Search for "App Passwords"</li>
                    <li>Select "Mail" and "Other (Custom name)"</li>
                    <li>Copy the 16-character password</li>
                </ul>
            </li>
        </ol>
        
        <p><strong>Note:</strong> Never share or commit your App Password to version control!</p>
    </div>
    """