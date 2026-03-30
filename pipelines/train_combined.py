"""Training script combining synthetic and real data."""

import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from src.ai_feature.sentiment_model import SentimentModel

def create_synthetic_data(n_samples=500):
    """Create synthetic sentiment data."""
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
    ]
    
    labels = [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    
    # Repeat to get desired number of samples
    multiplier = n_samples // len(texts)
    texts = texts * multiplier
    labels = labels * multiplier
    
    return texts[:n_samples], labels[:n_samples]

def load_real_data(csv_file):
    """Load real data from CSV."""
    data = pd.read_csv(csv_file)
    texts = data['review'].tolist()
    raw_labels = data['sentiment'].tolist()
    labels = [1 if label.lower().strip() == 'positive' else 0 for label in raw_labels]
    return texts, labels

def main():
    """Train model with combined data."""
    print("=" * 60)
    print("Training Sentiment Model with Combined Data")
    print("=" * 60)
    
    # Load synthetic data (500 samples for good training)
    print("\n📊 Creating synthetic data...")
    synth_texts, synth_labels = create_synthetic_data(500)
    print(f"   Synthetic samples: {len(synth_texts)}")
    print(f"   Positive: {sum(synth_labels)}, Negative: {len(synth_labels) - sum(synth_labels)}")
    
    # Load real data
    print("\n📂 Loading real data from CSV...")
    real_texts, real_labels = load_real_data('customer_reviews.csv')
    print(f"   Real samples: {len(real_texts)}")
    print(f"   Positive: {sum(real_labels)}, Negative: {len(real_labels) - sum(real_labels)}")
    
    # Combine data
    print("\n🔗 Combining datasets...")
    all_texts = synth_texts + real_texts
    all_labels = synth_labels + real_labels
    print(f"   Total samples: {len(all_texts)}")
    
    # Split data (80% train, 20% test)
    print("\n📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        all_texts, all_labels, test_size=0.2, random_state=42
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
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # Print results
    print("\n" + "=" * 60)
    print("✅ Training Results:")
    print("=" * 60)
    print(f"   Accuracy:  {accuracy:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall:    {recall:.3f}")
    print(f"   F1 Score:  {f1:.3f}")
    print("=" * 60)
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/sentiment_model.joblib")
    print("\n💾 Model saved to: models/sentiment_model.joblib")
    
    # Test with real-world examples
    print("\n🧪 Testing with real-world examples:")
    test_examples = [
        "This product is absolutely fantastic!",
        "Worst experience ever, never buying again",
        "It's okay, does the job",
        "Amazing quality and fast shipping!",
        "Complete waste of money",
        "The customer service was very helpful"
    ]
    
    for example in test_examples:
        result = model.predict(example)
        emoji = "😊" if result['sentiment'] == 'positive' else "😞"
        print(f"   {emoji} '{example}'")
        print(f"      → {result['sentiment']} (confidence: {result['confidence']:.3f})")

if __name__ == "__main__":
    main()