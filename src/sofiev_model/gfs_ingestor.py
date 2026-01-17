from __future__ import annotations
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Tuple

import numpy as np
import s3fs
import xarray as xr


class GFSIngestor:
    """
    Connects to the NOAA GFS bucket on AWS to retrieve analysis data (f000)
    for specific locations and times.
    """

    def __init__(self):
        """
        Initializes the GFSIngestor by connecting to NOAA's S3 bucket.
        """
        # Connect to NOAA's Public S3 Bucket (Anonymous access)
        self.fs = s3fs.S3FileSystem(anon=True)
        self.bucket = "noaa-gfs-bdp-pds"
        print("Connected to AWS S3 (noaa-gfs-bdp-pds).")

    def get_analysis_grid(
        self,
        target_time: datetime,
        lat_range: Tuple[float, float],
        lon_range: Tuple[float, float],
    ) -> xr.Dataset:
        """
        Fetches and computes PBL Height, Wind, and Stability for a grid.

        This method is vectorized and uses dask for lazy-loading to handle
        large geographical domains efficiently.

        Parameters
        ----------
        target_time : datetime
            The time of the desired GFS analysis.
        lat_range : tuple[float, float]
            The min and max latitude (e.g., (30.0, 50.0)).
        lon_range : tuple[float, float]
            The min and max longitude (e.g., (-120.0, -80.0)).

        Returns
        -------
        xr.Dataset
            A dataset containing the following variables over the specified grid:
            - `pbl_height`: Planetary boundary layer height (m)
            - `wind_speed_850mb`: Wind speed at 850mb (m/s)
            - `n_ft`: Brunt-Vaisala frequency in the free troposphere (s^-1)
            Returns an empty dataset if the fetch fails.
        """
        # 1. Round time to nearest 6H GFS cycle (00, 06, 12, 18)
        hour = target_time.hour
        cycle_hour = (hour // 6) * 6
        cycle_dt = target_time.replace(
            hour=cycle_hour, minute=0, second=0, microsecond=0
        )
        date_str = cycle_dt.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"

        # Path: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.f000
        s3_path = f"{self.bucket}/gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f000"

        # Adjust longitude (GFS uses 0-360)
        lon_min, lon_max = lon_range
        lon_gfs_min = lon_min if lon_min >= 0 else lon_min + 360
        lon_gfs_max = lon_max if lon_max >= 0 else lon_max + 360

        try:
            print(f"   Fetching GFS from S3: {s3_path}...")
            mapper = s3fs.S3Map(root=s3_path, s3=self.fs, check=False)

            # Define latitude and longitude slices
            # GFS latitude is descending, so slice max->min
            lat_slice = slice(lat_range[1], lat_range[0])
            lon_slice = slice(lon_gfs_min, lon_gfs_max)

            # --- A. Surface Data (PBL Height) ---
            ds_surf = xr.open_dataset(
                mapper,
                engine="cfgrib",
                chunks={"latitude": "auto", "longitude": "auto"},
                backend_kwargs={
                    "filter_by_keys": {"stepType": "instant", "typeOfLevel": "surface"}
                },
            )
            h_abl_grid = ds_surf["hpbl"].sel(latitude=lat_slice, longitude=lon_slice)
            h_abl_grid.name = "pbl_height"

            # --- B. Isobaric Data (Wind & Stability) ---
            ds_iso = xr.open_dataset(
                mapper,
                engine="cfgrib",
                chunks={"latitude": "auto", "longitude": "auto"},
                backend_kwargs={
                    "filter_by_keys": {
                        "stepType": "instant",
                        "typeOfLevel": "isobaricInhPa",
                    }
                },
            )
            iso_grid = ds_iso.sel(latitude=lat_slice, longitude=lon_slice)

            # --- C. Vectorized Computations ---
            # Wind speed at 850mb
            u850 = iso_grid["u"].sel(isobaricInhPa=850)
            v850 = iso_grid["v"].sel(isobaricInhPa=850)
            wind_speed = xr.apply_ufunc(
                np.sqrt,
                u850**2 + v850**2,
                dask="parallelized",
                output_dtypes=[u850.dtype],
            )
            wind_speed.name = "wind_speed_850mb"

            # Brunt-Vaisala Frequency (N) between 850mb and 700mb
            t850 = iso_grid["t"].sel(isobaricInhPa=850)
            t700 = iso_grid["t"].sel(isobaricInhPa=700)
            theta850 = t850 * (1000 / 850) ** 0.286
            theta700 = t700 * (1000 / 700) ** 0.286
            theta_avg = (theta850 + theta700) / 2.0
            d_theta = theta700 - theta850

            g = 9.81
            dz_approx = 1500.0  # Approx meters between 850mb and 700mb
            # Use np.sqrt which works on xarray DataArrays
            n_ft = np.sqrt((g / theta_avg) * (d_theta / dz_approx)).where(
                d_theta > 0, 0.001
            )
            n_ft.name = "n_ft"

            # --- D. Assemble Final Dataset ---
            result_ds = xr.merge([h_abl_grid, wind_speed, n_ft], compat="override")
            timestamp = datetime.now(timezone.utc).isoformat()
            result_ds.attrs["history"] = (
                f"[{timestamp}] Processed GFS analysis data. Calculated wind speed and Brunt-Vaisala frequency."
            )

            return result_ds

        except Exception as e:
            print(f"   [Error] GFS Fetch Failed for {target_time}: {e}")
            return xr.Dataset()

    def get_analysis_point(
        self, target_time: datetime, lat: float, lon: float
    ) -> tuple[float, float, float]:
        """
        .. deprecated:: 1.1
            Use :func:`get_analysis_grid` instead for vectorized performance.

        Fetches PBL Height (Surface) and Stability/Wind (Isobaric)
        from GFS Analysis for a single point.
        """
        warnings.warn(
            "`get_analysis_point` is deprecated and will be removed in a future version. "
            "Use `get_analysis_grid` for better performance.",
            DeprecationWarning,
            stacklevel=2,
        )
        grid_ds = self.get_analysis_grid(
            target_time, lat_range=(lat, lat), lon_range=(lon, lon)
        )
        if not grid_ds:
            return np.nan, np.nan, np.nan

        return (
            grid_ds["pbl_height"].values.item(),
            grid_ds["wind_speed_850mb"].values.item(),
            grid_ds["n_ft"].values.item(),
        )
