"""Simple test that doesn't depend on model."""

def test_simple():
    """Simple test that always passes."""
    assert True

def test_imports():
    """Test that we can import the module."""
    try:
        from src.ai_feature import __version__
        assert __version__ is not None
    except ImportError as e:
        assert False, f"Import failed: {e}"