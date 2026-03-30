"""Model interface for AI feature."""

from .feature import AIFeature

# Global feature instance
_feature = None

def get_feature():
    """Get or create the AI feature instance."""
    global _feature
    if _feature is None:
        _feature = AIFeature()
    return _feature

def predict(text):
    """Predict sentiment for input text."""
    feature = get_feature()
    return feature.analyze_sentiment(text)