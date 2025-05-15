import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
import warnings
import pickle
warnings.filterwarnings('ignore')

def train_phishing_detector(data_path):
    """Train a simplified phishing email detector model using preprocessed data."""
    try:
        # Load preprocessed dataset
        print("Loading preprocessed dataset...")
        df = pd.read_csv(data_path)
        
        # Ensure correct column names
        text_column = 'text_combined'
        if text_column not in df.columns or 'label' not in df.columns:
            raise ValueError(f"Required columns not found in the dataset")
        
        # Print dataset statistics
        print(f"\nDataset statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Phishing emails: {len(df[df['label'] == 1])}")
        print(f"Safe emails: {len(df[df['label'] == 0])}")
        
        # Create features
        print("\nExtracting features...")
        vectorizer = TfidfVectorizer(
            max_features=3000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2)
        )
        X = vectorizer.fit_transform(df[text_column])
        y = df['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        print("\nTraining model...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        
        print("\nModel Performance:")
        print(f"ROC AUC Score: {auc_score:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, vectorizer
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return None, None

def predict_phishing_probability(email_text, model, vectorizer):
    """Predict the probability of an email being a phishing attempt."""
    try:
        # Note: For real-time predictions, we still need to preprocess new emails
        from data_converter import preprocess_text
        email = preprocess_text(email_text)
        email_vec = vectorizer.transform([email])
        probability = model.predict_proba(email_vec)[0][1]
        return probability
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return -1

if __name__ == "__main__":
    # Train the model using preprocessed dataset
    print("Starting model training...")
    model, vectorizer = train_phishing_detector('preprocessed_phishing_dataset.csv')
    
    if model and vectorizer:
        print("\nModel training completed successfully!")
        
        # Save model components
        print("\nSaving model components...")
        pickle.dump(model, open('phishing_model.pkl', 'wb'))
        pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
        print("Model components saved successfully!")
        
        # Example predictions
        test_emails = [
            "I am your dad, send money",
            "Dear valued customer, your account has been suspended. Click here to verify.",
            "Meeting scheduled for tomorrow at 10 AM in the conference room.",
            "Congratulations! You've won $1,000,000. Send your bank details to claim."
        ]
        
        print("\nTesting example emails:")
        for email in test_emails:
            prob = predict_phishing_probability(email, model, vectorizer)
            if prob != -1:
                print(f"\nEmail: {email}")
                print(f"Phishing probability: {prob:.4f}")
                print(f"Classification: {'Likely Phishing' if prob > 0.5 else 'Likely Legitimate'}") 