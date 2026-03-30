"""Sentiment analysis model with sklearn backend."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

class SentimentModel:
    """Sentiment analysis model with sklearn backend."""
    
    def __init__(self, model_path=None):
        """Initialize the sentiment model."""
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, texts, labels):
        """Train the model on text data."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_trained = True
    
    def predict(self, text):
        """Predict sentiment of a single text."""
        if not self.is_trained:
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "message": "Model not trained - using fallback"
            }
        
        X = self.vectorizer.transform([text])
        proba = self.classifier.predict_proba(X)[0]
        pred = self.classifier.predict(X)[0]
        
        sentiment = "positive" if pred == 1 else "negative"
        confidence = max(proba)
        
        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "input_length": len(text)
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts."""
        return [self.predict(text) for text in texts]
    
    def save(self, path):
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'classifier': self.classifier
        }, path)
    
    def load(self, path):
        """Load model from disk."""
        data = joblib.load(path)
        self.vectorizer = data['vectorizer']
        self.classifier = data['classifier']
        self.is_trained = True