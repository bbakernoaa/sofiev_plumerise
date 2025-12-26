from __future__ import annotations
import numpy as np
import xarray as xr
import s3fs
from datetime import datetime

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

    def get_analysis_point(self, target_time: datetime, lat: float, lon: float) -> tuple[float, float, float]:
        """
        Fetches PBL Height (Surface) and Stability/Wind (Isobaric)
        from GFS Analysis for a single point.

        Parameters
        ----------
        target_time : datetime
            The time of the desired GFS analysis.
        lat : float
            The latitude of the desired point.
        lon : float
            The longitude of the desired point.

        Returns
        -------
        tuple[float, float, float]
            A tuple containing the planetary boundary layer height (m),
            wind speed (m/s), and Brunt-Vaisala frequency (s^-1).
            Returns (np.nan, np.nan, np.nan) if the fetch fails.
        """
        # 1. Round time to nearest 6H GFS cycle (00, 06, 12, 18)
        # GFS directories are organized by cycle start time
        hour = target_time.hour
        cycle_hour = (hour // 6) * 6
        cycle_dt = target_time.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)

        date_str = cycle_dt.strftime("%Y%m%d")
        cycle_str = f"{cycle_hour:02d}"

        # Path: gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.f000
        s3_path = f"{self.bucket}/gfs.{date_str}/{cycle_str}/atmos/gfs.t{cycle_str}z.pgrb2.0p25.f000"

        # Adjust longitude (GFS uses 0-360)
        lon_gfs = lon if lon >= 0 else lon + 360

        try:
            # We use a context manager to ensure files close properly
            # NOTE: We use 'cfgrib' engine. Ensure ecCodes is installed on your system.
            print(f"   Fetching GFS from S3: {s3_path}...")

            # Create a mapper to stream only required bytes
            mapper = s3fs.S3Map(root=s3_path, s3=self.fs, check=False)

            # --- A. Surface Data (PBL Height) ---
            # Filter for surface level variables to speed up read
            with xr.open_dataset(mapper, engine='cfgrib',
                                     backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'surface'}}) as ds_surf:
                # GFS Variable 'hpbl' = Planetary Boundary Layer Height
                h_abl = ds_surf['hpbl'].sel(latitude=lat, longitude=lon_gfs, method='nearest').values.item()

            # --- B. Isobaric Data (Wind & Stability in Free Troposphere) ---
            # We need 850mb and 700mb for stability calc
            with xr.open_dataset(mapper, engine='cfgrib',
                                    backend_kwargs={'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa'}}) as ds_iso:
                # Extract Winds at 850mb (Transport Level)
                u850 = ds_iso['u'].sel(isobaricInhPa=850, latitude=lat, longitude=lon_gfs, method='nearest').values.item()
                v850 = ds_iso['v'].sel(isobaricInhPa=850, latitude=lat, longitude=lon_gfs, method='nearest').values.item()
                wind_speed = np.sqrt(u850**2 + v850**2)

                # Extract Temps for Brunt-Vaisala (N)
                t850 = ds_iso['t'].sel(isobaricInhPa=850, latitude=lat, longitude=lon_gfs, method='nearest').values.item()
                t700 = ds_iso['t'].sel(isobaricInhPa=700, latitude=lat, longitude=lon_gfs, method='nearest').values.item()

            # Calculate Potential Temperature (Theta)
            theta850 = t850 * (1000/850)**0.286
            theta700 = t700 * (1000/700)**0.286

            # Calc N (Stability)
            g = 9.81
            dz_approx = 1500.0 # Approx meters between 850mb and 700mb
            theta_avg = (theta850 + theta700) / 2.0
            d_theta = theta700 - theta850

            if d_theta > 0:
                n_ft = np.sqrt((g / theta_avg) * (d_theta / dz_approx))
            else:
                n_ft = 0.001 # Unstable/Neutral

            return h_abl, wind_speed, n_ft

        except Exception as e:
            print(f"   [Error] GFS Fetch Failed for {target_time}: {e}")
            # Return NaNs so we can drop them later
            return np.nan, np.nan, np.nan
