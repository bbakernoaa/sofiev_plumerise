import pytest
import os
from datetime import datetime
import pandas as pd
from sofiev_model.satellite_ingestor import (
    TempoIngestor,
    TropomiIngestor,
    OmpsIngestor,
    load_and_collocate_data,
)
from sofiev_model.gfs_ingestor import GFSIngestor
from requests.exceptions import ConnectionError

# Define a small AOI and time range for testing
TEST_AOI = [-120, 35, -118, 37]  # Small box over California
TEST_START_DATE = datetime(2024, 5, 1)
TEST_END_DATE = datetime(2024, 5, 1, 1, 0)


@pytest.fixture(scope="module")
def gfs_ingestor():
    return GFSIngestor()

# Basic initialization tests
def test_tempo_ingestor_initialization():
    """Tests that the TempoIngestor class can be initialized."""
    try:
        TempoIngestor()
    except Exception as e:
        pytest.fail(f"TempoIngestor initialization failed: {e}")

def test_tropomi_ingestor_initialization():
    """Tests that the TropomiIngestor class can be initialized."""
    try:
        TropomiIngestor()
    except Exception as e:
        pytest.fail(f"TropomiIngestor initialization failed: {e}")

def test_omps_ingestor_initialization():
    """Tests that the OmpsIngestor class can be initialized."""
    try:
        OmpsIngestor()
    except Exception as e:
        pytest.fail(f"OmpsIngestor initialization failed: {e}")


@pytest.mark.skipif(not os.getenv("EARTHDATA_USERNAME"), reason="EARTHDATA_USERNAME not set")
def test_tempo_ingestor_get_data():
    """Tests the TempoIngestor get_data method."""
    ingestor = TempoIngestor()
    try:
        df = ingestor.get_data(TEST_START_DATE, TEST_END_DATE, TEST_AOI)
        assert isinstance(df, pd.DataFrame)
        if df.empty:
            pytest.skip("No TEMPO data found for the given parameters, skipping validation.")
        assert 'no2' in df.columns
    except ConnectionError:
        pytest.skip("Skipping test due to network connection error.")


def test_tropomi_ingestor_get_data():
    """Tests the TropomiIngestor get_data method."""
    ingestor = TropomiIngestor()
    try:
        df = ingestor.get_data(TEST_START_DATE, TEST_END_DATE, TEST_AOI)
        assert isinstance(df, pd.DataFrame)
        if df.empty:
            pytest.skip("No TROPOMI data found for the given parameters, skipping validation.")
        assert 'so2' in df.columns
    except ConnectionError:
        pytest.skip("Skipping test due to network connection error.")

@pytest.mark.skipif(not os.getenv("EARTHDATA_USERNAME"), reason="EARTHDATA_USERNAME not set")
def test_omps_ingestor_get_data():
    """Tests the OmpsIngestor get_data method."""
    ingestor = OmpsIngestor()
    try:
        df = ingestor.get_data(TEST_START_DATE, TEST_END_DATE, TEST_AOI)
        assert isinstance(df, pd.DataFrame)
        if df.empty:
            pytest.skip("No OMPS data found for the given parameters, skipping validation.")
        assert 'so2_omps' in df.columns
    except ConnectionError:
        pytest.skip("Skipping test due to network connection error.")

# Test for the collocation function
@pytest.mark.skipif(not os.getenv("EARTHDATA_USERNAME"), reason="EARTHDATA_USERNAME not set")
def test_load_and_collocate_data(gfs_ingestor):
    """
    Tests that the load_and_collocate_data function returns a DataFrame
    with collocated GFS data.
    """
    ingestors = [TempoIngestor(), TropomiIngestor(), OmpsIngestor()]
    try:
        df = load_and_collocate_data(ingestors, gfs_ingestor, TEST_START_DATE, TEST_END_DATE, TEST_AOI)
        if df.empty:
            pytest.skip("No satellite data found for collocation, skipping test.")
        assert 'h_abl' in df.columns
        assert 'n_ft' in df.columns
        assert 'wind_speed' in df.columns
    except ConnectionError:
        pytest.skip("Skipping test due to network connection error.")
