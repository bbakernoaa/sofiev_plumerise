import os
import numpy as np
import xarray as xr
import pandas as pd
import xgboost as xgb
from scipy.ndimage import gaussian_filter
from typing import Tuple

# Import your rigorous vectorized routines
from .cffwi import FWI_Engine_Vectorized

class FireEmissionGenerator:
    """Orchestrates the calculation of biomass burning emissions.

    This class integrates meteorological data, a fuel climatology, and a
    trained machine learning model to produce daily scaling factors for
    biomass burning emissions. It is designed to be used in a daily
    time-stepping loop, where each step produces a new emission field.

    Attributes
    ----------
    res : float
        The target resolution in degrees for the output grid.
    target_lats : np.ndarray
        The target latitudes for the output grid.
    target_lons : np.ndarray
        The target longitudes for the output grid.
    fwi_engine : FWI_Engine_Vectorized
        The vectorized FWI calculation engine.
    model : xgb.XGBRegressor
        The trained XGBoost model for emission scaling.
    climo : xr.Dataset
        The lazy-loaded GBBEPx climatology dataset.
    """
    def __init__(self, model_path: str, climo_path: str, target_res: float = 0.04):
        """Initializes the FireEmissionGenerator.

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

    @staticmethod
    def get_fire_memory(
        history_ds: xr.Dataset, current_time: pd.Timestamp
    ) -> xr.DataArray:
        """Calculates the 6-month cumulative FRP for biomass depletion.

        This method integrates the fire radiative power (FRP) over the last 6
        months to create a 'memory' field, which represents a proxy for fuel
        depletion.

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
        >>> mem = FireEmissionGenerator.get_fire_memory(history_ds, current_time)
        >>> mem.shape
        (10, 10)
        """
        six_months_ago = current_time - pd.DateOffset(months=6)
        # Select and sum emissions history. The slice end is made exclusive by
        # subtracting a nanosecond, ensuring we only get the last 6 full months.
        end_period = current_time - pd.Timedelta(nanoseconds=1)
        memory = history_ds['FRP'].sel(time=slice(six_months_ago, end_period)).sum(dim='time')
        return memory

    def run_step(
        self,
        ufs_met: xr.Dataset,
        prev_states: xr.Dataset,
        memory_grid: xr.DataArray,
        igbp_map: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.Dataset]:
        """Executes a single daily timestep for the global 4km grid.

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
        Tuple[xr.DataArray, xr.Dataset]
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

        Examples
        --------
        **Real-Data Workflow**

        This example demonstrates a complete workflow for generating an
        emission field, starting from creating realistic sample data files.

        First, create the necessary input NetCDF files. This includes a
        dummy XGBoost model, GBBEPx climatology, IGBP land cover map,
        meteorological data, and initial FWI moisture code states.

        .. code-block:: python

            import xarray as xr
            import numpy as np
            import pandas as pd
            import xgboost as xgb
            import os

            # Define grid dimensions
            lats = np.arange(40, 40.2, 0.04)
            lons = np.arange(-105.2, -105, 0.04)
            time = pd.to_datetime(['2023-07-01T12:00:00'])
            coords_2d = {'lat': lats, 'lon': lons}
            coords_3d = {'time': time, 'lat': lats, 'lon': lons}

            # 1. Dummy XGBoost Model
            model = xgb.XGBRegressor(objective='reg:squarederror')
            # A dummy fit is required before saving
            dummy_X = np.random.rand(1, 6)
            dummy_y = np.random.rand(1)
            model.fit(dummy_X, dummy_y)
            model_path = 'dummy_model.json'
            model.save_model(model_path)

            # 2. Dummy GBBEPx Climatology
            climo_path = 'dummy_climo.nc'
            climo_ds = xr.Dataset(
                {'emissions': (('month', 'lat', 'lon'), np.ones((12, len(lats), len(lons))))},
                coords={'month': range(1, 13), 'lat': lats, 'lon': lons}
            )
            climo_ds.to_netcdf(climo_path)

            # 3. Dummy IGBP Map
            igbp_path = 'dummy_igbp.nc'
            igbp_ds = xr.Dataset(
                {'band_1': (('lat', 'lon'), np.full((len(lats), len(lons)), 4))},
                coords=coords_2d
            )
            igbp_ds.to_netcdf(igbp_path)

            # 4. Dummy UFS Meteorology
            met_path = 'dummy_met.nc'
            met_ds = xr.Dataset({
                't2m': (('time', 'lat', 'lon'), np.full((1, len(lats), len(lons)), 298.0)),
                'rh2m': (('time', 'lat', 'lon'), np.full((1, len(lats), len(lons)), 45.0)),
                'u10': (('time', 'lat', 'lon'), np.full((1, len(lats), len(lons)), 3.0)),
                'v10': (('time', 'lat', 'lon'), np.full((1, len(lats), len(lons)), 3.0)),
                'precip': (('time', 'lat', 'lon'), np.zeros((1, len(lats), len(lons)))),
            }, coords=coords_3d)
            met_ds.to_netcdf(met_path)

            # 5. Dummy Previous FWI States
            states_path = 'dummy_prev_states.nc'
            states_ds = xr.Dataset({
                'ffmc': (('lat', 'lon'), np.full((len(lats), len(lons)), 88.0)),
                'dmc': (('lat', 'lon'), np.full((len(lats), len(lons)), 90.0)),
                'dc': (('lat', 'lon'), np.full((len(lats), len(lons)), 350.0)),
            }, coords=coords_2d)
            states_ds.to_netcdf(states_path)

            # 6. Dummy Fire Memory Grid
            memory_path = 'dummy_memory.nc'
            memory_ds = xr.Dataset(
                {'FRP': (('lat', 'lon'), np.ones((len(lats), len(lons))))},
                coords=coords_2d
            )
            memory_ds.to_netcdf(memory_path)

        Now, execute the model for a single time step.

        .. code-block:: python

            # Initialize the generator
            generator = FireEmissionGenerator(model_path=model_path, climo_path=climo_path)

            # Load the prepared data
            ufs_met = xr.open_dataset(met_path).squeeze()
            prev_states = xr.open_dataset(states_path)
            memory_grid = xr.open_dataset(memory_path)['FRP']
            igbp_map = xr.open_dataset(igbp_path)['band_1']

            # Run the core method
            final_emissions, new_states = generator.run_step(
                ufs_met=ufs_met,
                prev_states=prev_states,
                memory_grid=memory_grid,
                igbp_map=igbp_map
            )

            # Save the new FWI states for the next time step
            output_states_path = 'new_fwi_states.nc'
            generator.save_state(new_states, output_states_path)

            print(f"Final emissions grid shape: {final_emissions.shape}")
            print(f"New FWI states saved to: {output_states_path}")

        Finally, visualize the output emissions using ``hvplot``.

        .. code-block:: python

            import hvplot.xarray  # noqa
            import cartopy.crs as ccrs

            # Generate an interactive plot
            plot = final_emissions.hvplot.quadmesh(
                'lon', 'lat',
                geo=True,
                cmap='inferno',
                clim=(0, final_emissions.quantile(0.99)),
                tiles='OSM',
                frame_width=600,
                frame_height=400,
                title='Predicted Biomass Burning Emissions',
                rasterize=True  # Essential for large grids
            )
            # To view the plot in a notebook, simply display the 'plot' object
            # To save it, you might need a backend like hvplot.save(plot, 'emissions_map.html')

        This example provides a complete, runnable template for using the
        ``FireEmissionGenerator`` with real-world, gridded data.
        """
        # 1. Parse Time
        current_dt = pd.to_datetime(ufs_met.time.item())
        month = current_dt.month

        # 2. Update FWI Moisture Codes (Dask-aware)
        wind_speed = np.sqrt(ufs_met['u10']**2 + ufs_met['v10']**2)

        new_ffmc = self.fwi_engine.calculate_ffmc(
            ufs_met['t2m'], ufs_met['rh2m'], wind_speed, ufs_met['precip'], prev_states['ffmc']
        )
        new_dmc = self.fwi_engine.calculate_dmc(
            ufs_met['t2m'], ufs_met['rh2m'], ufs_met['precip'], prev_states['dmc'], month
        )
        new_dc = self.fwi_engine.calculate_dc(
            ufs_met['t2m'], ufs_met['precip'], prev_states['dc'], month
        )

        # 3. Calculate Behavioral Indices
        bui = self.fwi_engine.calculate_bui(new_dmc, new_dc)

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
        # Drop the scalar 'month' coordinate inherited from the climatology
        # to ensure the output grid is cleanly defined by lat/lon only.
        if 'month' in final_emissions.coords:
            final_emissions = final_emissions.drop_vars('month')

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
        """Saves FWI moisture codes to NetCDF for restart capability.

        This method serializes an xarray.Dataset containing the moisture
        codes ('ffmc', 'dmc', 'dc') to a NetCDF file. This allows the state
        to be saved at the end of a model run and loaded at the beginning
        of the next, enabling continuous simulations.

        Parameters
        ----------
        states : xr.Dataset
            The FWI moisture codes to save. Must contain 'ffmc', 'dmc', and
            'dc' as DataArrays with 'lat' and 'lon' coordinates.
        filename : str
            The path to the output NetCDF file. The directory will be
            created if it does not exist.

        Examples
        --------
        >>> states_ds = xr.Dataset({
        ...     'ffmc': (('lat', 'lon'), np.full((10, 10), 85.0)),
        ...     'dmc': (('lat', 'lon'), np.full((10, 10), 15.0)),
        ...     'dc': (('lat', 'lon'), np.full((10, 10), 200.0)),
        ... }, coords={'lat': np.arange(10), 'lon': np.arange(10)})
        >>> generator.save_state(states_ds, 'fwi_states.nc')
        >>> os.path.exists('fwi_states.nc')
        True
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        states.to_netcdf(filename)

    def load_state(self, filename: str) -> xr.Dataset:
        """Loads FWI moisture codes from a previous day's output.

        This method deserializes a NetCDF file into an xarray.Dataset,
        providing the initial moisture code states needed to start a model
        run.

        Parameters
        ----------
        filename : str
            The path to the input NetCDF file.

        Returns
        -------
        xr.Dataset
            The loaded FWI moisture codes, containing 'ffmc', 'dmc', and 'dc'
            DataArrays.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        Examples
        --------
        >>> # Assuming 'fwi_states.nc' was created by save_state
        >>> loaded_states = generator.load_state('fwi_states.nc')
        >>> 'dc' in loaded_states
        True
        >>> os.remove('fwi_states.nc') # Clean up the dummy file
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"State file not found at {filename}")
        return xr.open_dataset(filename)

# --- END OF FILE ---
