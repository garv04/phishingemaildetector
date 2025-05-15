import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text):
    """Basic text preprocessing."""
    try:
        # Handle NaN values
        if pd.isna(text):
            return ""
            
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep important punctuation
        text = ''.join(char for char in text if char.isalnum() or char.isspace() or char in '.,@$!')
        
        # Tokenize
        tokens = text.split()
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        
        # Lemmatize
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in text preprocessing: {e}")
        return ""

def convert_and_combine_datasets():
    # Load the first dataset
    print("Loading first dataset...")
    df1 = pd.read_csv('phishing_email.csv')
    
    # Load the second dataset
    print("Loading second dataset...")
    df2 = pd.read_csv('Phishing.csv')
    
    # Convert Email Type to binary labels (0 for Safe, 1 for Phishing)
    print("Converting labels...")
    df2['label'] = (df2['Email Type'] == 'Phishing Email').astype(int)
    
    # Rename Email Text column to match first dataset
    df2 = df2.rename(columns={'Email Text': 'text_combined'})
    
    # Keep only necessary columns
    df2 = df2[['text_combined', 'label']]
    
    # Combine datasets
    print("Combining datasets...")
    combined_df = pd.concat([df1, df2], ignore_index=True)
    
    # Remove rows with empty text after preprocessing
    print("Preprocessing text data...")
    combined_df['text_combined'] = combined_df['text_combined'].apply(preprocess_text)
    combined_df = combined_df[combined_df['text_combined'] != ""]
    
    # Save combined and preprocessed dataset
    print("Saving preprocessed dataset...")
    combined_df.to_csv('preprocessed_phishing_dataset.csv', index=False)
    
    print("\nDataset statistics:")
    print(f"Total samples: {len(combined_df)}")
    print(f"Phishing emails: {len(combined_df[combined_df['label'] == 1])}")
    print(f"Safe emails: {len(combined_df[combined_df['label'] == 0])}")

if __name__ == "__main__":
    convert_and_combine_datasets() 