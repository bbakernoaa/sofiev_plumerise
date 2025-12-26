import pytest
from sofiev_model.gfs_ingestor import GFSIngestor

def test_gfs_ingestor_initialization():
    """
    Tests that the GFSIngestor class can be initialized.
    """
    try:
        GFSIngestor()
    except Exception as e:
        pytest.fail(f"GFSIngestor initialization failed: {e}")
