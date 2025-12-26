import pytest
from sofiev_model.satellite_ingestor import SatelliteIngestor
from sofiev_model.gfs_ingestor import GFSIngestor

def test_satellite_ingestor_initialization():
    """
    Tests that the SatelliteIngestor class can be initialized.
    """
    try:
        SatelliteIngestor()
    except Exception as e:
        pytest.fail(f"SatelliteIngestor initialization failed: {e}")

def test_get_collocated_dataset():
    """
    Tests that the get_collocated_dataset method returns a DataFrame.
    """
    ingestor = SatelliteIngestor()
    gfs_ingestor = GFSIngestor()
    df = ingestor.get_collocated_dataset(gfs_ingestor)
    assert not df.empty
