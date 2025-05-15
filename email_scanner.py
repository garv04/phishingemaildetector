import imaplib
import email
from email.header import decode_header
import os
import pickle
from phishing_detector import predict_phishing_probability
from data_converter import preprocess_text
import time
import yaml
from pathlib import Path

class EmailScanner:
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Load the trained model components
        self.model = pickle.load(open(self.config['model_path'], 'rb'))
        self.vectorizer = pickle.load(open(self.config['vectorizer_path'], 'rb'))
        
        # IMAP connection settings
        self.imap_server = self.config['imap_server']
        self.email_address = self.config['email_address']
        self.password = self.config['password']
        self.scan_folder = self.config['scan_folder']
        self.suspicious_folder = self.config['suspicious_folder']
        self.threshold = self.config['threshold']
        
    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            self._create_default_config(config_path)
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_default_config(self, config_path):
        """Create default configuration file"""
        default_config = {
            'imap_server': 'imap.gmail.com',
            'email_address': 'your_email@gmail.com',
            'password': 'your_app_specific_password',
            'scan_folder': 'INBOX',
            'suspicious_folder': 'Suspicious',
            'threshold': 0.7,
            'model_path': 'phishing_model.pkl',
            'vectorizer_path': 'vectorizer.pkl',
            'feature_selector_path': 'feature_selector.pkl'
        }
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print(f"Created default configuration file at {config_path}")
        print("Please update it with your email credentials before running the scanner.")
        exit(1)

    def connect(self):
        """Establish IMAP connection"""
        try:
            self.mail = imaplib.IMAP4_SSL(self.imap_server)
            self.mail.login(self.email_address, self.password)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def ensure_suspicious_folder_exists(self):
        """Create Suspicious folder if it doesn't exist"""
        try:
            self.mail.create(self.suspicious_folder)
        except:
            pass  # Folder might already exist

    def get_email_content(self, email_message):
        """Extract email content from email message"""
        content = ""
        if email_message.is_multipart():
            for part in email_message.walk():
                if part.get_content_type() == "text/plain":
                    content += part.get_payload(decode=True).decode()
        else:
            content = email_message.get_payload(decode=True).decode()
        return content

    def process_emails(self):
        """Process emails in the scan folder"""
        try:
            # Select the folder to scan
            self.mail.select(self.scan_folder)
            
            # Search for all emails in the folder
            _, message_numbers = self.mail.search(None, 'ALL')
            
            for num in message_numbers[0].split():
                # Fetch email message
                _, msg_data = self.mail.fetch(num, '(RFC822)')
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)
                
                # Get subject
                subject = decode_header(email_message["Subject"])[0][0]
                if isinstance(subject, bytes):
                    subject = subject.decode()
                
                # Get content
                content = self.get_email_content(email_message)
                
                # Combine subject and content for analysis
                full_text = f"{subject}\n{content}"
                
                # Predict phishing probability
                prob = predict_phishing_probability(full_text, self.model, self.vectorizer)
                
                # If probability is above threshold, move to suspicious folder
                if prob > self.threshold:
                    self.mail.copy(num, self.suspicious_folder)
                    print(f"Moved suspicious email: {subject} (Probability: {prob:.2f})")
            
            return True
        except Exception as e:
            print(f"Error processing emails: {e}")
            return False

    def disconnect(self):
        """Close IMAP connection"""
        try:
            self.mail.close()
            self.mail.logout()
        except:
            pass

def save_model_files(model, vectorizer):
    """Save model and vectorizer to files"""
    pickle.dump(model, open('phishing_model.pkl', 'wb'))
    pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
    print("Model and vectorizer saved to files")

def main():
    # Create scanner instance
    scanner = EmailScanner()
    
    # Connect to email server
    if not scanner.connect():
        return
    
    # Ensure Suspicious folder exists
    scanner.ensure_suspicious_folder_exists()
    
    print("Starting email scanning...")
    while True:
        scanner.process_emails()
        print("Waiting for new emails...")
        time.sleep(300)  # Wait 5 minutes before next scan

if __name__ == "__main__":
    main() 