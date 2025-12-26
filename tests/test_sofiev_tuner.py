import pytest
from sofiev_model.sofiev_tuner import SofievTuner

def test_sofiev_tuner_initialization():
    """
    Tests that the SofievTuner class can be initialized.
    """
    try:
        SofievTuner()
    except Exception as e:
        pytest.fail(f"SofievTuner initialization failed: {e}")
