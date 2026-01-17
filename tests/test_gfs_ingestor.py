import pytest
from unittest.mock import patch
from datetime import datetime
import xarray as xr
import numpy as np
import dask.array as da
from sofiev_model.gfs_ingestor import GFSIngestor


@pytest.fixture
def mock_gfs_data():
    """Creates mock GFS xarray.Dataset objects for surface and isobaric levels."""
    # Common coordinates
    lat_coords = np.arange(50, 40, -1.0)
    lon_coords = np.arange(240, 250, 1.0)

    # --- Mock Surface Dataset (for PBL Height) ---
    pbl_height_data = da.from_array(
        np.full((len(lat_coords), len(lon_coords)), 1500.0), chunks=(10, 10)
    )
    ds_surf = xr.Dataset(
        {"hpbl": (("latitude", "longitude"), pbl_height_data)},
        coords={"latitude": lat_coords, "longitude": lon_coords},
    )

    # --- Mock Isobaric Dataset (for Wind & Temperature) ---
    isobaric_levels = [850, 700]
    u_data = da.from_array(
        np.full((len(isobaric_levels), len(lat_coords), len(lon_coords)), 10.0),
        chunks=(2, 10, 10),
    )
    v_data = da.from_array(
        np.full((len(isobaric_levels), len(lat_coords), len(lon_coords)), -5.0),
        chunks=(2, 10, 10),
    )
    t_data = da.from_array(
        np.array(
            [
                np.full((len(lat_coords), len(lon_coords)), 285.0),  # Temp at 850mb
                np.full((len(lat_coords), len(lon_coords)), 275.0),  # Temp at 700mb
            ]
        ),
        chunks=(2, 10, 10),
    )

    ds_iso = xr.Dataset(
        {
            "u": (("isobaricInhPa", "latitude", "longitude"), u_data),
            "v": (("isobaricInhPa", "latitude", "longitude"), v_data),
            "t": (("isobaricInhPa", "latitude", "longitude"), t_data),
        },
        coords={
            "isobaricInhPa": isobaric_levels,
            "latitude": lat_coords,
            "longitude": lon_coords,
        },
    )

    return ds_surf, ds_iso


@patch("s3fs.S3FileSystem")
@patch("s3fs.S3Map")
def test_get_analysis_grid_lazy_and_correct(MockS3Map, MockS3FileSystem, mock_gfs_data):
    """
    Tests that get_analysis_grid returns a lazy dask-backed dataset and
    that the final computed values are correct.
    """
    # Arrange
    MockS3FileSystem.return_value
    MockS3Map.return_value

    ds_surf_mock, ds_iso_mock = mock_gfs_data

    # Mock xr.open_dataset to return our mock datasets in order
    with patch(
        "xarray.open_dataset", side_effect=[ds_surf_mock, ds_iso_mock]
    ) as mock_open_dataset:
        ingestor = GFSIngestor()

        # Act
        target_time = datetime(2023, 10, 27, 14, 0)
        lat_range = (40.0, 50.0)
        lon_range = (-120.0, -110.0)  # Converted to 240, 250

        result_ds = ingestor.get_analysis_grid(target_time, lat_range, lon_range)

        # --- Assertions ---

        # 1. Assert Laziness: The data should be a Dask array
        assert isinstance(result_ds["pbl_height"].data, da.Array)
        assert isinstance(result_ds["wind_speed_850mb"].data, da.Array)
        assert isinstance(result_ds["n_ft"].data, da.Array)

        # 2. Assert history attribute is present
        assert "history" in result_ds.attrs
        assert "Processed GFS analysis data" in result_ds.attrs["history"]

        # 3. Assert correct structure
        expected_vars = {"pbl_height", "wind_speed_850mb", "n_ft"}
        assert set(result_ds.variables) == expected_vars.union(result_ds.coords)

        # 4. Trigger computation and assert correctness of values
        computed_ds = result_ds.compute()

        # PBL Height
        expected_pbl_height = 1500.0
        assert np.allclose(computed_ds["pbl_height"].values, expected_pbl_height)

        # Wind Speed at 850mb
        expected_wind_speed = np.sqrt(10.0**2 + (-5.0) ** 2)  # sqrt(100 + 25)
        assert np.allclose(computed_ds["wind_speed_850mb"].values, expected_wind_speed)

        # Brunt-Vaisala Frequency (N_ft)
        # Manually recalculate based on mock data
        g = 9.81
        dz_approx = 1500.0
        t850 = 285.0
        t700 = 275.0
        theta850 = t850 * (1000 / 850) ** 0.286
        theta700 = t700 * (1000 / 700) ** 0.286
        theta_avg = (theta850 + theta700) / 2.0
        d_theta = theta700 - theta850
        expected_n_ft = np.sqrt((g / theta_avg) * (d_theta / dz_approx))

        assert np.allclose(computed_ds["n_ft"].values, expected_n_ft)

        # 5. Assert S3 path was correctly formed
        assert mock_open_dataset.call_count == 2
        call_args = MockS3Map.call_args
        assert (
            call_args[1]["root"]
            == "noaa-gfs-bdp-pds/gfs.20231027/12/atmos/gfs.t12z.pgrb2.0p25.f000"
        )
