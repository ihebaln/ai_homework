"""Simple training script for sentiment analysis model."""

import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import our model
from src.ai_feature.sentiment_model import SentimentModel

def create_sample_data():
    """Create sample sentiment data."""
    texts = [
        "I love this product! It's amazing!",
        "This is terrible, worst purchase ever.",
        "Not bad, works as expected.",
        "Absolutely fantastic! Would recommend.",
        "Waste of money, very disappointed.",
        "Good quality, satisfied with purchase.",
        "Horrible experience, never again.",
        "Pretty good, does what it says.",
        "Disappointing, expected more.",
        "Excellent service and fast delivery!"
    ] * 100  # Multiply to create more samples
    
    labels = [
        1, 0, 1, 1, 0, 1, 0, 1, 0, 1
    ] * 100
    
    return texts, labels

def main():
    """Train sentiment model."""
    print("=" * 50)
    print("Training Sentiment Analysis Model")
    print("=" * 50)
    
    # Create data
    print("\n📊 Creating training data...")
    texts, labels = create_sample_data()
    print(f"   Total samples: {len(texts)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n🤖 Training model...")
    model = SentimentModel()
    model.train(X_train, y_train)
    
    # Evaluate
    print("\n📈 Evaluating model...")
    predictions = model.predict_batch(X_test)
    y_pred = [1 if p['sentiment'] == 'positive' else 0 for p in predictions]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Print results
    print("\n" + "=" * 50)
    print("✅ Training Results:")
    print("=" * 50)
    print(f"   Accuracy:  {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print("=" * 50)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/sentiment_model.joblib")
    print("\n💾 Model saved to: models/sentiment_model.joblib")
    
    # Test the model with examples
    print("\n🧪 Testing the model with examples:")
    test_examples = [
        "I absolutely love this!",
        "This is terrible",
        "It's okay, nothing special",
        "Best thing ever!",
        "Worst purchase of my life"
    ]
    
    for example in test_examples:
        result = model.predict(example)
        print(f"   '{example}' -> {result['sentiment']} (confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()