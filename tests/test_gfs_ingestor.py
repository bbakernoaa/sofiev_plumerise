from __future__ import annotations

import unittest.mock
from datetime import datetime

import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from sofiev_model.gfs_ingestor import GFSIngestor


@pytest.fixture
def mock_gfs_datasets():
    """Pytest fixture to create mock GFS xarray.Dataset objects."""
    # Common coordinates
    lat_coords = np.arange(50, 40, -1.0)  # Descending latitude
    lon_coords = np.arange(240, 250, 1.0)
    time_coord = datetime(2023, 1, 1, 6, 0, 0)

    # --- Mock Surface Dataset (for PBL Height) ---
    ds_surf = xr.Dataset(
        {
            "hpbl": (
                ("latitude", "longitude"),
                np.ones((len(lat_coords), len(lon_coords))) * 1200.0,
            ),
        },
        coords={
            "latitude": lat_coords,
            "longitude": lon_coords,
            "time": time_coord,
        },
    )

    # --- Mock Isobaric Dataset (for wind, temp) ---
    pressure_levels = [850, 700]
    ds_iso = xr.Dataset(
        {
            "u": (
                ("isobaricInhPa", "latitude", "longitude"),
                np.full((2, len(lat_coords), len(lon_coords)), 10.0),  # U-wind
            ),
            "v": (
                ("isobaricInhPa", "latitude", "longitude"),
                np.full((2, len(lat_coords), len(lon_coords)), -5.0),  # V-wind
            ),
            "t": (
                ("isobaricInhPa", "latitude", "longitude"),
                np.array(
                    [
                        np.full(
                            (len(lat_coords), len(lon_coords)), 283.0
                        ),  # Temp at 850mb
                        np.full(
                            (len(lat_coords), len(lon_coords)), 273.0
                        ),  # Temp at 700mb
                    ]
                ),
            ),
        },
        coords={
            "isobaricInhPa": pressure_levels,
            "latitude": lat_coords,
            "longitude": lon_coords,
            "time": time_coord,
        },
    )
    return ds_surf, ds_iso


def test_get_analysis_grid(monkeypatch, mock_gfs_datasets):
    """
    Unit test for GFSIngestor.get_analysis_grid.

    Tests that the function correctly processes mocked GFS data into a
    final dataset with derived variables. It patches s3fs and xr.open_dataset
    to avoid actual network calls.
    """
    # 1. Setup Mocks
    mock_s3fs = unittest.mock.MagicMock()
    monkeypatch.setattr("s3fs.S3FileSystem", mock_s3fs)
    monkeypatch.setattr("s3fs.S3Map", unittest.mock.MagicMock())

    ds_surf, ds_iso = mock_gfs_datasets

    # Mock xr.open_dataset to return different datasets based on filter keys
    def mock_open_dataset(*args, **kwargs):
        filter_keys = kwargs.get("backend_kwargs", {}).get("filter_by_keys", {})
        if filter_keys.get("typeOfLevel") == "surface":
            return ds_surf
        elif filter_keys.get("typeOfLevel") == "isobaricInhPa":
            return ds_iso
        raise ValueError("Unexpected call to xr.open_dataset with mock")

    monkeypatch.setattr("xarray.open_dataset", mock_open_dataset)

    # 2. Call the Method
    ingestor = GFSIngestor()
    target_time = datetime(2023, 1, 1, 7, 0, 0)  # Will round to 6Z cycle
    lat_range = (40.0, 50.0)
    lon_range = (-120.0, -110.0)  # Converts to 240-250
    result_ds = ingestor.get_analysis_grid(target_time, lat_range, lon_range)

    # 3. Create Expected Dataset for Comparison
    # Expected wind speed: sqrt(10^2 + (-5)^2) = sqrt(125) = 11.1803
    expected_wind = np.full_like(ds_surf.hpbl.values, 11.1803, dtype=np.float32)

    # Expected Brunt-Vaisala (N_ft)
    t850, t700 = 283.0, 273.0
    theta850 = t850 * (1000 / 850) ** 0.286
    theta700 = t700 * (1000 / 700) ** 0.286
    theta_avg = (theta850 + theta700) / 2.0
    d_theta = theta700 - theta850
    g, dz_approx = 9.81, 1500.0
    expected_n_ft_val = np.sqrt((g / theta_avg) * (d_theta / dz_approx))
    expected_n_ft = np.full_like(
        ds_surf.hpbl.values, expected_n_ft_val, dtype=np.float32
    )

    expected_ds = xr.Dataset(
        {
            "pbl_height": (("latitude", "longitude"), ds_surf.hpbl.values),
            "wind_speed_850mb": (("latitude", "longitude"), expected_wind),
            "n_ft": (("latitude", "longitude"), expected_n_ft),
        },
        coords={
            "latitude": ds_surf.latitude.values,
            "longitude": ds_surf.longitude.values,
            "time": ds_surf.time.values,
            "isobaricInhPa": 850,
        },
    ).drop_vars("isobaricInhPa")

    # 4. Assertions
    assert isinstance(result_ds, xr.Dataset)
    assert "history" in result_ds.attrs
    # Drop non-essential coords from result for comparison
    result_ds_simplified = result_ds.drop_vars("isobaricInhPa", errors="ignore")
    # Use xarray's testing utility for float comparisons
    assert_allclose(
        result_ds_simplified.astype(np.float32),
        expected_ds.astype(np.float32),
        rtol=1e-4,
    )
