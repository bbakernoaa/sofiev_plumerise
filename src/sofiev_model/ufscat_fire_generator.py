import os
import numpy as np
import xarray as xr
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter

# Import your rigorous vectorized routines
from .cffwi import FWI_Engine_Vectorized

class FireEmissionGenerator:
    def __init__(self, model_path: str, climo_path: str, target_res: float = 0.04):
        """UFS/CATChem Fire Generator for RISE.

        This class orchestrates the calculation of biomass burning emissions by
        integrating meteorological data, a fuel climatology, and a trained
        machine learning model to produce daily scaling factors.

        Parameters
        ----------
        model_path : str
            Path to the trained XGBoost .json or .model file.
        climo_path : str
            Path to the aggregated GBBEPx climatology NetCDF file. This file
            is expected to contain a monthly emission climatology.
        target_res : float, optional
            Target resolution in degrees for the output grid, by default 0.04,
            approximating a 4km grid spacing.

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

        # Load the GBBEPx Climatology lazily with Dask
        self.climo = xr.open_dataset(climo_path, chunks={'lat': 512, 'lon': 512})

    @staticmethod
    def calculate_vpd(t2m: xr.DataArray, rh2m: xr.DataArray) -> xr.DataArray:
        """Calculate Vapor Pressure Deficit (VPD) in a Dask-aware manner.

        This method uses the Tetens equation to estimate saturated vapor
        pressure from temperature.

        Parameters
        ----------
        t2m : xr.DataArray
            2-meter temperature in Kelvin.
        rh2m : xr.DataArray
            2-meter relative humidity in percent.

        Returns
        -------
        xr.DataArray
            Vapor Pressure Deficit in hPa.

        Examples
        --------
        >>> t2m = xr.DataArray([293.15, 303.15], dims=['x'])
        >>> rh2m = xr.DataArray([50, 60], dims=['x'])
        >>> vpd = FireEmissionGenerator.calculate_vpd(t2m, rh2m)
        """
        # Tetens equation for saturated vapor pressure
        t_c = t2m - 273.15
        es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
        return es * (1.0 - rh2m / 100.0)

    def get_fire_memory(self, history_ds: xr.Dataset, current_time: pd.Timestamp) -> xr.DataArray:
        """Calculate 6-month cumulative FRP for biomass depletion.

        Parameters
        ----------
        history_ds : xr.Dataset
            Time-series of previously generated FRP emissions. Must have a
            'time' coordinate.
        current_time : pd.Timestamp
            The current timestamp for the model run.

        Returns
        -------
        xr.DataArray
            A 2D DataArray of cumulative FRP over the last 6 months.

        Examples
        --------
        >>> time_range = pd.date_range('2022-01-01', '2023-01-01', freq='M')
        >>> history_ds = xr.Dataset(
        ...     {'FRP': (('time', 'lat', 'lon'), np.ones((len(time_range), 10, 10)))},
        ...     coords={'time': time_range, 'lat': np.arange(10), 'lon': np.arange(10)}
        ... )
        >>> current_time = pd.Timestamp('2023-01-01')
        >>> mem = FireEmissionGenerator.get_fire_memory(None, history_ds, current_time)
        >>> mem.shape
        (10, 10)
        """
        six_months_ago = current_time - pd.DateOffset(months=6)
        # Select and sum emissions history
        memory = history_ds['FRP'].sel(time=slice(six_months_ago, current_time)).sum(dim='time')
        return memory

    def run_step(self, ufs_met: xr.Dataset, prev_states: xr.Dataset,
                 memory_grid: xr.DataArray, igbp_map: xr.DataArray) -> tuple[xr.DataArray, xr.Dataset]:
        """Execute a single daily timestep for the global 4km grid.

        This is the core operational method. It takes the latest meteorological
        data and the previous day's FWI state, computes the next state, and
        predicts an emissions scaling factor using the XGBoost model.

        Parameters
        ----------
        ufs_met : xr.Dataset
            Current UFS meteorology. Must contain `t2m`, `rh2m`, `u10`, `v10`,
            and `precip` as DataArrays.
        prev_states : xr.Dataset
            Dataset of 2D arrays for `ffmc`, `dmc`, and `dc` from the
            previous day.
        memory_grid : xr.DataArray
            2D array of cumulative FRP (6-month lag).
        igbp_map : xr.DataArray
            2D array of IGBP land cover classes.

        Returns
        -------
        tuple[xr.DataArray, xr.Dataset]
            - The final scaled emissions as a 2D DataArray, preserving
              coordinates and including a history attribute.
            - An updated xr.Dataset containing the new FWI moisture codes
              (`ffmc`, `dmc`, `dc`).

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> import pandas as pd
        >>> from unittest.mock import MagicMock
        >>> # Create dummy model and climo files
        >>> dummy_model = '{"objective":"reg:squarederror"}'
        >>> with open('dummy_model.json', 'w') as f:
        ...     f.write(dummy_model)
        >>> dummy_climo = xr.Dataset({
        ...     'emissions': (('month', 'lat', 'lon'), np.ones((12, 10, 10)))
        ... }, coords={'month': range(1, 13), 'lat': np.arange(10), 'lon': np.arange(10)})
        >>> dummy_climo.to_netcdf('dummy_climo.nc')
        >>> # Setup FireEmissionGenerator
        >>> generator = FireEmissionGenerator(
        ...     model_path='dummy_model.json', climo_path='dummy_climo.nc'
        ... )
        >>> generator.model.predict = MagicMock(return_value=np.ones(100))
        >>> # Create sample inputs
        >>> lats, lons = np.arange(10), np.arange(10)
        >>> time = pd.to_datetime(['2023-07-01'])
        >>> coords = {'lat': lats, 'lon': lons, 'time': time}
        >>> ufs_met = xr.Dataset({
        ...     't2m': (('time', 'lat', 'lon'), np.full((1, 10, 10), 295.0)),
        ...     'rh2m': (('time', 'lat', 'lon'), np.full((1, 10, 10), 60.0)),
        ...     'u10': (('time', 'lat', 'lon'), np.full((1, 10, 10), 5.0)),
        ...     'v10': (('time', 'lat', 'lon'), np.full((1, 10, 10), 5.0)),
        ...     'precip': (('time', 'lat', 'lon'), np.zeros((1, 10, 10))),
        ... }, coords=coords).squeeze()
        >>> prev_states = xr.Dataset({
        ...     'ffmc': (('lat', 'lon'), np.full((10, 10), 85.0)),
        ...     'dmc': (('lat', 'lon'), np.full((10, 10), 15.0)),
        ...     'dc': (('lat', 'lon'), np.full((10, 10), 200.0)),
        ... }, coords={'lat': lats, 'lon': lons})
        >>> memory_grid = xr.DataArray(np.ones((10, 10)), coords=[lats, lons], dims=['lat', 'lon'])
        >>> igbp_map = xr.DataArray(np.ones((10, 10)), coords=[lats, lons], dims=['lat', 'lon'])
        >>> # Run the step
        >>> emissions, new_states = generator.run_step(ufs_met, prev_states, memory_grid, igbp_map)
        >>> print(emissions.shape)
        (10, 10)
        >>> print(new_states['dc'].shape)
        (10, 10)
        >>> 'history' in emissions.attrs
        True
        """
        # 1. Parse Time
        current_dt = pd.to_datetime(ufs_met.time.item())
        month = current_dt.month

        # 2. Update FWI Moisture Codes (Dask-aware)
        wind_speed = np.sqrt(ufs_met['u10']**2 + ufs_met['v10']**2)

        # Ensure FWI outputs are explicitly cast back to DataArrays to preserve metadata
        _coords = prev_states.coords
        _dims = prev_states['ffmc'].dims  # All states share the same dims

        _new_ffmc_data = self.fwi_engine.calculate_ffmc(
            ufs_met['t2m'], ufs_met['rh2m'], wind_speed, ufs_met['precip'], prev_states['ffmc']
        )
        new_ffmc = xr.DataArray(_new_ffmc_data, coords=_coords, dims=_dims)

        _new_dmc_data = self.fwi_engine.calculate_dmc(
            ufs_met['t2m'], ufs_met['rh2m'], ufs_met['precip'], prev_states['dmc'], month
        )
        new_dmc = xr.DataArray(_new_dmc_data, coords=_coords, dims=_dims)

        _new_dc_data = self.fwi_engine.calculate_dc(
            ufs_met['t2m'], ufs_met['precip'], prev_states['dc'], month
        )
        new_dc = xr.DataArray(_new_dc_data, coords=_coords, dims=_dims)


        # 3. Calculate Behavioral Indices
        _bui_data = self.fwi_engine.calculate_bui(new_dmc, new_dc)
        bui = xr.DataArray(_bui_data, coords=_coords, dims=_dims)

        # 4. Supplemental Predictors
        vpd = self.calculate_vpd(ufs_met['t2m'], ufs_met['rh2m'])

        # 5. ML Scaling (Dask-aware Feature Assembly)
        # Create a Dataset of predictors to ensure alignment
        predictors = xr.Dataset({
            'dc': new_dc,
            'bui': bui,
            'wind': wind_speed,
            'vpd': vpd,
            'memory': memory_grid,
            'igbp': igbp_map
        })

        # Stack into a DataArray for ML input - this remains a lazy Dask operation
        feature_stack = predictors.to_array(dim='variable')
        # Rechunk to ensure the 'variable' dimension is a single block for the ML model.
        feature_stack = feature_stack.chunk({'variable': -1})

        def predict_point(feature_vector):
            """Predicts the scale factor for a single spatial point."""
            # self.model.predict expects a 2D array of shape (n_samples, n_features)
            prediction = self.model.predict(feature_vector.reshape(1, -1))
            return prediction[0]

        # Use apply_ufunc with vectorize=True to apply the point-wise function
        # across the chunked spatial dimensions ('lat', 'lon').
        raw_scale = xr.apply_ufunc(
            predict_point,
            feature_stack,
            input_core_dims=[['variable']],  # The function operates on the 'variable' dim
            output_core_dims=[[]],           # It returns a scalar
            exclude_dims=set(('variable',)), # The 'variable' dim is consumed
            dask="parallelized",
            output_dtypes=[feature_stack.dtype],
            vectorize=True  # Automatically broadcast the function over non-core dims
        )

        # 6. Post-processing (Dask-aware)
        def smooth_and_clip(array, sigma):
            smoothed = gaussian_filter(array, sigma=sigma)
            return np.clip(smoothed, 0.01, 20.0)

        # map_overlap handles the stencil operation for gaussian_filter across chunks
        smooth_scale_data = raw_scale.data.map_overlap(
            smooth_and_clip,
            depth=2,  # Depth must be >= sigma
            boundary='reflect',
            sigma=1.0
        )
        smooth_scale_da = xr.DataArray(smooth_scale_data, coords=raw_scale.coords, dims=raw_scale.dims)

        # 7. Apply to Base Climatology
        base_emissions = self.climo['emissions'].sel(month=month)
        final_emissions = base_emissions * smooth_scale_da

        # 8. Format Output
        history_log = (
            f"{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}: "
            f"Fire emissions generated with UFSCATChemFireGenerator."
        )
        final_emissions.attrs['history'] = history_log
        final_emissions.name = 'emissions'

        new_states_ds = xr.Dataset({
            'ffmc': new_ffmc,
            'dmc': new_dmc,
            'dc': new_dc
        }, coords=ufs_met.coords)

        return final_emissions, new_states_ds

    def save_state(self, states: xr.Dataset, filename: str) -> None:
        """Saves FWI moisture codes to NetCDF for restart capability."""
        states.to_netcdf(filename)

    def load_state(self, filename: str) -> xr.Dataset:
        """Loads FWI moisture codes from a previous day's output."""
        return xr.open_dataset(filename)

# --- END OF FILE ---
