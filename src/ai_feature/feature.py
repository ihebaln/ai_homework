"""Main AI feature interface."""

import os
import sys
from .sentiment_model import SentimentModel

class AIFeature:
    """Main AI feature class for sentiment analysis."""
    
    def __init__(self):
        """Initialize the AI feature."""
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        model_path = "models/sentiment_model.joblib"
        if os.path.exists(model_path):
            self.model = SentimentModel(model_path)
            print("✅ Model loaded successfully")
        else:
            print("⚠️  No model found. Run training first: python pipelines/train_sentiment.py")
            self.model = SentimentModel()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of input text."""
        if not text or not isinstance(text, str):
            return {
                "error": "Invalid input",
                "message": "Text must be a non-empty string"
            }
        
        result = self.model.predict(text)
        
        return {
            "text": text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "status": "success"
        }
    
    def batch_analyze(self, texts):
        """Analyze sentiment for multiple texts."""
        return [self.analyze_sentiment(text) for text in texts]