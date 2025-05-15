# Phishing Email Scanner

An intelligent email scanning system that automatically detects and segregates potential phishing emails using machine learning.

## Features

- Real-time email monitoring
- Machine learning-based phishing detection
- Automatic segregation of suspicious emails
- Support for Gmail accounts
- Configurable scanning intervals and detection thresholds

## Prerequisites

- Python 3.6 or higher
- Gmail account with IMAP enabled
- Google Account 2-Step Verification enabled
- Gmail App Password (required for authentication)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cyberai
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Enable 2-Step Verification in your Google Account:
   - Go to [Google Account Security Settings](https://myaccount.google.com/security)
   - Find "2-Step Verification" and enable it

2. Generate Gmail App Password:
   - Go to [Google Account Security Settings](https://myaccount.google.com/security)
   - Click on "App passwords" (under "Signing in to Google")
   - Select "Mail" as the app and your device type
   - Click "Generate"
   - Copy the 16-character password provided

3. Update `config.yaml`:
   ```yaml
   # Email Server Settings
   imap_server: imap.gmail.com
   email_address: your.email@gmail.com
   password: your-16-char-app-password  # Paste your App Password here

   # Folder Settings
   scan_folder: INBOX
   suspicious_folder: Suspicious

   # Model Settings
   threshold: 0.5  # Probability threshold for flagging emails as suspicious
   model_path: phishing_model.pkl
   vectorizer_path: vectorizer.pkl
   ```

   Important: Replace `your.email@gmail.com` with your Gmail address and `your-16-char-app-password` with the App Password generated in step 2.

## Usage

1. Start the email scanner:
   ```bash
   python3 email_scanner.py
   ```

2. The scanner will:
   - Monitor your inbox continuously
   - Analyze incoming emails for phishing attempts
   - Move suspicious emails to a "Suspicious" folder
   - Print status updates to the console

3. To stop the scanner, press `Ctrl+C`

## How It Works

1. **Email Monitoring**: The system connects to your Gmail account using IMAP and monitors incoming emails.

2. **Text Processing**: Each email's subject and content are extracted and preprocessed for analysis.

3. **ML Detection**: A trained machine learning model (XGBoost) analyzes the processed text to determine the probability of the email being a phishing attempt.

4. **Automatic Segregation**: Emails with a phishing probability above the configured threshold are automatically moved to the "Suspicious" folder.

## Security Notes

- The system uses App Passwords for authentication, which is more secure than storing your main Gmail password
- App Passwords can be revoked at any time from your Google Account settings
- The system runs locally on your machine and doesn't share your emails with external services
- All ML processing is done locally using pre-trained models

## Troubleshooting

1. **Authentication Failed**
   - Ensure you're using an App Password, not your regular Gmail password
   - Verify that 2-Step Verification is enabled
   - Check that the App Password was copied correctly (no spaces)

2. **Connection Issues**
   - Verify your internet connection
   - Ensure IMAP is enabled in your Gmail settings
   - Check if your Gmail account has any security blocks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


