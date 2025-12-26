from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .gfs_ingestor import GFSIngestor

class SatelliteIngestor:
    """
    Generates synthetic 'Truth' data for demonstration.
    Replace 'simulate_data' with actual NetCDF readers for production.
    """
    def __init__(self, n_samples: int = 200):
        """
        Initializes the SatelliteIngestor.

        Parameters
        ----------
        n_samples : int, optional
            The number of synthetic fire events to generate, by default 200.
        """
        self.n_samples = n_samples

    def get_collocated_dataset(self, gfs_ingestor: GFSIngestor) -> pd.DataFrame:
        """
        Generates a synthetic dataset of fire events and collocates them with GFS data.

        Parameters
        ----------
        gfs_ingestor : GFSIngestor
            An instance of the GFSIngestor class.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the synthetic fire events and collocated GFS data.
        """
        print("\n--- Generating/Ingesting Satellite Data ---")

        # 1. Create Synthetic Fire Events
        # Fires usually happen in afternoon, summer
        base_time = datetime(2023, 7, 15, 12, 0)
        times = [base_time + timedelta(hours=np.random.randint(0, 48)) for _ in range(self.n_samples)]
        lats = np.random.uniform(35, 48, self.n_samples) # US/Canada latitudes
        lons = np.random.uniform(-120, -100, self.n_samples)

        # 2. Fire Intensity (FRP) - Exponential distribution
        frp = np.random.exponential(500, self.n_samples) + 50

        # 3. Fetch GFS Data (Real AWS calls or Simulation for speed)
        # NOTE: For this demo, we will SIMULATE the GFS return to avoid
        # hitting AWS 200 times and waiting 10 mins.
        # UNCOMMENT the loop below to use real AWS data.

        h_abl_list = []
        n_ft_list = []
        wind_list = []

        print("   (Simulating GFS values for speed...)")
        h_abl_list = np.random.normal(1500, 400, self.n_samples)
        n_ft_list = np.abs(np.random.normal(0.012, 0.003, self.n_samples))
        wind_list = np.random.weibull(2, self.n_samples) * 6

        # --- REAL AWS FETCH LOOP (Uncomment for production) ---
        # for t, lat, lon in zip(times, lats, lons):
        #     h, w, n = gfs_ingestor.get_analysis_point(t, lat, lon)
        #     h_abl_list.append(h)
        #     n_ft_list.append(n)
        #     wind_list.append(w)

        # 4. Generate "Observed" Plume Heights (The Truth)
        # We assume truth follows a complex physics law we want to rediscover
        # Real Obs = Buoyancy - WindShear + RandomNoise
        h_obs = 200 * (frp**0.4) * np.exp(-0.5 * np.array(n_ft_list)/0.01) - (50 * np.array(wind_list)) + np.random.normal(0, 300, self.n_samples)
        h_obs = np.maximum(h_obs, np.array(h_abl_list) + 100) # Ensure it rises at least near PBL

        df = pd.DataFrame({
            'time': times,
            'lat': lats,
            'lon': lons,
            'frp_total': frp,
            'h_abl': h_abl_list,
            'n_ft': n_ft_list,
            'wind_speed': wind_list,
            'h_obs_misr': h_obs
        })

        return df.dropna()
