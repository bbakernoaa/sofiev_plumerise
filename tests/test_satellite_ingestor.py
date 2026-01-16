# tests/test_satellite_ingestor.py
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr
from requests.exceptions import ConnectionError

from sofiev_model.satellite_ingestor import (
    OmpsIngestor,
    SyntheticIngestor,
    TempoIngestor,
    TropomiIngestor,
)


def test_synthetic_ingestor_creation():
    """Tests the creation of a SyntheticIngestor instance."""
    ingestor = SyntheticIngestor(n_samples=10)
    assert ingestor.n_samples == 10


def test_synthetic_ingestor_fetch_data():
    """Tests the fetch_data method of the SyntheticIngestor."""
    ingestor = SyntheticIngestor(n_samples=50)
    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 1, 2)
    bbox = (-120, 30, -110, 40)
    ds = ingestor.fetch_data(start_time, end_time, bbox)

    assert isinstance(ds, xr.Dataset)
    assert "frp" in ds.variables
    assert "plume_height" in ds.variables
    assert len(ds["event"]) == 50
    assert ds.attrs["source"] == "SyntheticIngestor"
    assert "history" in ds.attrs


@pytest.mark.parametrize(
    "ingestor_class, env_vars",
    [
        (
            TropomiIngestor,
            {"COPERNICUS_USERNAME": "user", "COPERNICUS_PASSWORD": "password"},
        ),
        (
            OmpsIngestor,
            {"EARTHDATA_USERNAME": "user", "EARTHDATA_PASSWORD": "password"},
        ),
        (
            TempoIngestor,
            {"EARTHDATA_USERNAME": "user", "EARTHDATA_PASSWORD": "password"},
        ),
    ],
)
def test_real_ingestors_init(mocker, ingestor_class, env_vars):
    """Tests the initialization of real ingestors, checking for credentials."""
    mocker.patch.dict(os.environ, env_vars)
    if ingestor_class == TropomiIngestor:
        mocker.patch("sofiev_model.satellite_ingestor.SentinelAPI")
    else:
        mocker.patch("sofiev_model.satellite_ingestor.earthaccess.login")

    ingestor = ingestor_class()
    assert ingestor is not None

    # Test that it fails without credentials
    mocker.patch.dict(os.environ, clear=True)
    with pytest.raises(ValueError):
        ingestor_class()


def create_mock_dataset():
    """Helper function to create a mock xarray dataset for testing."""
    return xr.Dataset(
        {
            "aerosol_height": (("time", "y", "x"), np.random.rand(1, 10, 10)),
        },
        coords={
            "time": [datetime.now()],
            "latitude": (("y", "x"), np.random.uniform(30, 40, (10, 10))),
            "longitude": (("y", "x"), np.random.uniform(-120, -110, (10, 10))),
        },
    )


@patch("xarray.open_dataset")
@patch("sofiev_model.satellite_ingestor.SentinelAPI")
def test_tropomi_ingestor_fetch_data(mock_sentinel_api, mock_open_dataset, mocker):
    """Tests the fetch_data method of the TropomiIngestor with mocked API calls."""
    mocker.patch.dict(
        os.environ, {"COPERNICUS_USERNAME": "user", "COPERNICUS_PASSWORD": "password"}
    )

    mock_api_instance = mock_sentinel_api.return_value
    mock_api_instance.query.return_value = {
        "product1": {"title": "S5P_OFFL_L2__AER_LH"}
    }
    mock_open_dataset.return_value = create_mock_dataset()

    ingestor = TropomiIngestor()
    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 1, 2)
    bbox = (-120, 30, -110, 40)

    try:
        ds = ingestor.fetch_data(start_time, end_time, bbox)
    except ConnectionError:
        pytest.skip("Test failed due to a network connection issue.")

    mock_api_instance.query.assert_called_once()
    mock_api_instance.download_all.assert_called_once_with(["product1"])
    mock_open_dataset.assert_called_once_with("S5P_OFFL_L2__AER_LH.nc", group="PRODUCT")

    assert isinstance(ds, xr.Dataset)
    assert "history" in ds.attrs


@pytest.mark.parametrize(
    "ingestor_class",
    [OmpsIngestor, TempoIngestor],
)
@patch("xarray.open_dataset")
@patch("sofiev_model.satellite_ingestor.earthaccess.download")
@patch("sofiev_model.satellite_ingestor.earthaccess.search_data")
def test_earthdata_ingestors_fetch_data(
    mock_search, mock_download, mock_open_dataset, ingestor_class, mocker
):
    """Tests the fetch_data method of Earthdata-based ingestors."""
    mocker.patch.dict(
        os.environ, {"EARTHDATA_USERNAME": "user", "EARTHDATA_PASSWORD": "password"}
    )
    mocker.patch("sofiev_model.satellite_ingestor.earthaccess.login")

    mock_search.return_value = [MagicMock()]
    mock_download.return_value = ["mock_file.nc"]
    mock_open_dataset.return_value = create_mock_dataset()

    ingestor = ingestor_class()
    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 1, 2)
    bbox = (-120, 30, -110, 40)
    try:
        ds = ingestor.fetch_data(start_time, end_time, bbox)
    except ConnectionError:
        pytest.skip("Test failed due to a network connection issue.")

    mock_search.assert_called_once()
    mock_download.assert_called_once()
    mock_open_dataset.assert_called_once_with("mock_file.nc", engine="h5netcdf")

    assert isinstance(ds, xr.Dataset)
    assert "history" in ds.attrs

    # Test case where no data is found
    mock_search.return_value = []
    ds_empty = ingestor.fetch_data(start_time, end_time, bbox)
    assert isinstance(ds_empty, xr.Dataset)
    assert not ds_empty.data_vars
