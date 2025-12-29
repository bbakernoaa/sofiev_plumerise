import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from scipy.ndimage import gaussian_filter

# Import your rigorous vectorized routines
from .cffwi import FWI_Engine_Vectorized


class UFSCATChemFireGenerator:
    """A class to generate fire emissions using a trained XGBoost model and FWI data."""

    def __init__(
        self, model_path: str, climo_path: str, target_res: float = 0.04
    ) -> None:
        """
        Initializes the UFS/CATChem Fire Generator.

        This setup involves loading a pre-trained XGBoost model and a climatology
        dataset, which are used to predict fire emissions based on meteorological
        and fuel conditions.

        Parameters
        ----------
        model_path : str
            Path to the trained XGBoost model file (.json or .model).
        climo_path : str
            Path to the aggregated GBBEPx climatology NetCDF file. This dataset
            is expected to be chunked for lazy loading with Dask.
        target_res : float, optional
            The target resolution in degrees for the output grid, by default 0.04.
        """
        self.res = target_res
        self.target_lats = np.arange(-90 + self.res / 2, 90, self.res)
        self.target_lons = np.arange(-180 + self.res / 2, 180, self.res)

        # Initialize FWI Engine
        self.fwi_engine = FWI_Engine_Vectorized()

        # Load the XGBoost model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model not found at {model_path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

        # Load the GBBEPx Climatology lazily using Dask
        self.climo = xr.open_dataset(climo_path, chunks="auto")

    def calculate_vpd(
        self, t2m: np.ndarray, rh2m: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Vapor Pressure Deficit (VPD) in hPa.

        VPD is a critical variable for fire behavior, indicating the dryness
        of the air. This implementation uses the Tetens equation.

        Parameters
        ----------
        t2m : np.ndarray
            2-meter temperature in Kelvin.
        rh2m : np.ndarray
            2-meter relative humidity in percent.

        Returns
        -------
        np.ndarray
            The calculated Vapor Pressure Deficit in hPa.
        """
        # Tetens equation for saturated vapor pressure
        t_c = t2m - 273.15
        es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        return es * (1.0 - rh2m / 100.0)

    def get_fire_memory(
        self, history_ds: xr.Dataset, current_time: pd.Timestamp
    ) -> np.ndarray:
        """
        Calculates 6-month cumulative FRP to handle biomass depletion.

        This "fire memory" term prevents the model from generating perpetual
        fires in the same location by accounting for the fuel consumed in
        recent months.

        Parameters
        ----------
        history_ds : xr.Dataset
            An xarray Dataset containing a time-series of previously generated
            Fire Radiative Power (FRP) emissions.
        current_time : pd.Timestamp
            The current timestamp for the model run.

        Returns
        -------
        np.ndarray
            A 2D array of the cumulative FRP over the last six months.
        """
        six_months_ago = pd.to_datetime(current_time) - pd.DateOffset(months=6)
        # Select and sum emissions history
        memory = (
            history_ds["FRP"]
            .sel(time=slice(six_months_ago, current_time))
            .sum(dim="time")
        )
        return memory.values

    def run_step(
        self,
        ufs_met: xr.Dataset,
        prev_states: Dict[str, np.ndarray],
        memory_grid: np.ndarray,
        igbp_map: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Executes a single daily timestep for the global 4km grid.

        This is the core method that integrates meteorological data, FWI moisture
        codes, and static predictors to generate a daily fire emission map.

        Parameters
        ----------
        ufs_met : xr.Dataset
            Dataset with the current day's UFS meteorology, including t2m, rh2m,
            u10, v10, and precip.
        prev_states : Dict[str, np.ndarray]
            Dictionary containing 2D arrays of the previous day's FWI moisture
            codes: 'ffmc', 'dmc', and 'dc'.
        memory_grid : np.ndarray
            2D array of the 6-month cumulative FRP (fire memory).
        igbp_map : np.ndarray
            2D array of IGBP land cover classes.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, np.ndarray]]
            - A 2D numpy array of the final scaled emissions for CATChem.
            - A dictionary containing the updated FWI moisture codes for the next day.
        """
        # 1. Parse Time
        current_dt = pd.to_datetime(ufs_met.time.values[0])
        month = current_dt.month

        # 2. Update FWI Moisture Codes via ccfwi.py
        wind_speed = np.sqrt(
            ufs_met["u10"].values ** 2 + ufs_met["v10"].values ** 2
        )
        new_ffmc = self.fwi_engine.calculate_ffmc(
            ufs_met["t2m"].values,
            ufs_met["rh2m"].values,
            wind_speed,
            ufs_met["precip"].values,
            prev_states["ffmc"],
        )
        new_dmc = self.fwi_engine.calculate_dmc(
            ufs_met["t2m"].values,
            ufs_met["rh2m"].values,
            ufs_met["precip"].values,
            prev_states["dmc"],
            month,
        )
        new_dc = self.fwi_engine.calculate_dc(
            ufs_met["t2m"].values,
            ufs_met["precip"].values,
            prev_states["dc"],
            month,
        )

        # 3. Calculate Behavioral Indices
        bui = self.fwi_engine.calculate_bui(new_dmc, new_dc)

        # 4. Supplemental Predictors
        vpd = self.calculate_vpd(ufs_met["t2m"].values, ufs_met["rh2m"].values)

        # 5. ML Scaling
        # Feature vector alignment: [DC, BUI, Wind, VPD, Memory, IGBP]
        X = np.stack(
            [
                new_dc.ravel(),
                bui.ravel(),
                wind_speed.ravel(),
                vpd.ravel(),
                memory_grid.ravel(),
                igbp_map.ravel(),
            ],
            axis=1,
        )

        # Predict scaling factor
        raw_scale = self.model.predict(X).reshape(new_dc.shape)

        # 6. Post-processing: Gaussian smoothing and clipping
        smooth_scale = gaussian_filter(raw_scale, sigma=1.0)
        smooth_scale = np.clip(smooth_scale, 0.01, 20.0)

        # 7. Apply to Base Climatology
        base_emissions = self.climo["emissions"].sel(month=month).values
        final_emissions = base_emissions * smooth_scale
        final_emissions_ds = xr.Dataset(
            {"emissions": (("lat", "lon"), final_emissions)},
            coords={
                "lat": self.target_lats,
                "lon": self.target_lons,
                "time": current_dt,
            },
        )
        final_emissions_ds.attrs[
            "history"
        ] = f"[{pd.Timestamp.now()}] XGBoost scaling applied."

        new_states = {"ffmc": new_ffmc, "dmc": new_dmc, "dc": new_dc}

        return final_emissions_ds["emissions"].values, new_states

    def save_state(
        self, states: Dict[str, np.ndarray], filename: str
    ) -> None:
        """
        Saves FWI moisture codes to a NetCDF file for restart capability.

        Parameters
        ----------
        states : Dict[str, np.ndarray]
            A dictionary containing the FWI moisture code arrays ('ffmc', 'dmc', 'dc').
        filename : str
            The path to the output NetCDF file.
        """
        ds = xr.Dataset(
            {
                "ffmc": (["lat", "lon"], states["ffmc"]),
                "dmc": (["lat", "lon"], states["dmc"]),
                "dc": (["lat", "lon"], states["dc"]),
            },
            coords={"lat": self.target_lats, "lon": self.target_lons},
        )
        ds.to_netcdf(filename)
        ds.attrs[
            "history"
        ] = f"[{pd.Timestamp.now()}] FWI state saved."

    def load_state(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Loads FWI moisture codes from a previous day's output file.

        Parameters
        ----------
        filename : str
            The path to the NetCDF file containing the FWI state.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary with the loaded FWI moisture codes.
        """
        ds = xr.open_dataset(filename)
        return {
            "ffmc": ds["ffmc"].values,
            "dmc": ds["dmc"].values,
            "dc": ds["dc"].values,
        }

# --- END OF FILE ---
