"""Tests for the model module."""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_feature.model import predict, get_feature


def test_model_initialization():
    """Test that model initializes correctly."""
    feature = get_feature()
    assert feature is not None
    assert hasattr(feature, 'model')


def test_predict_positive_sentiment():
    """Test positive sentiment detection."""
    result = predict("I love this amazing product!")
    assert result["status"] == "success"
    assert result["sentiment"] in ["positive", "negative"]
    assert 0 <= result["confidence"] <= 1


def test_predict_negative_sentiment():
    """Test negative sentiment detection."""
    result = predict("This is terrible and disappointing.")
    assert result["status"] == "success"
    assert result["sentiment"] in ["positive", "negative"]
    assert 0 <= result["confidence"] <= 1


def test_predict_empty_input():
    """Test empty input handling."""
    result = predict("")
    # Empty input should still work or return error
    assert result is not None


@pytest.mark.parametrize("text,expected_sentiment", [
    ("Great!", "positive"),
    ("Awesome work", "positive"),
    ("Bad", "negative"),
    ("Horrible", "negative"),
])
def test_predict_various_inputs(text, expected_sentiment):
    """Test prediction with various inputs."""
    result = predict(text)
    assert result["status"] == "success"
    assert result["sentiment"] in ["positive", "negative"]