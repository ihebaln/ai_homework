"""Training script that loads data from CSV file."""

import sys
import os
import pandas as pd

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.ai_feature.sentiment_model import SentimentModel

def load_data_from_csv(csv_file):
    """Load training data from CSV file.
    
    CSV should have columns: 'review' and 'sentiment'
    Sentiment values should be 'positive' or 'negative'
    """
    print(f"📂 Loading data from {csv_file}...")
    
    # Load CSV
    data = pd.read_csv(csv_file)
    
    # Get texts and labels
    texts = data['review'].tolist()
    raw_labels = data['sentiment'].tolist()
    
    # Convert labels to numbers (positive=1, negative=0)
    labels = [1 if label.lower().strip() == 'positive' else 0 for label in raw_labels]
    
    print(f"   Loaded {len(texts)} samples")
    print(f"   Positive samples: {sum(labels)}")
    print(f"   Negative samples: {len(labels) - sum(labels)}")
    
    return texts, labels

def main():
    """Train sentiment model from CSV data."""
    print("=" * 50)
    print("Training Sentiment Analysis Model from CSV")
    print("=" * 50)
    
    # Load data from CSV
    texts, labels = load_data_from_csv('customer_reviews.csv')
    
    # Split data into train and test sets
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Train model
    print("\n🤖 Training model...")
    model = SentimentModel()
    model.train(X_train, y_train)
    
    # Evaluate on test set
    print("\n📈 Evaluating model...")
    predictions = model.predict_batch(X_test)
    y_pred = [1 if p['sentiment'] == 'positive' else 0 for p in predictions]
    
    # Calculate metrics
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
    
    # Test with some examples
    print("\n🧪 Testing with examples:")
    test_examples = [
        "This product exceeded my expectations!",
        "Very disappointed with the quality",
        "It's okay, nothing special"
    ]
    
    for example in test_examples:
        result = model.predict(example)
        print(f"   '{example}' -> {result['sentiment']} (confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()