# satellite_ingestor.py
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Tuple

import earthaccess
import numpy as np
import xarray as xr
from sentinelsat import SentinelAPI


class SatelliteIngestor(ABC):
    """
    Abstract base class for ingesting satellite data.

    This class defines the interface for satellite data ingestors. Concrete
    subclasses must implement the `fetch_data` method to provide a standardized
    way to access satellite observations of fire events.
    """

    @abstractmethod
    def fetch_data(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Fetches satellite data for a given time range and bounding box.

        Parameters
        ----------
        start_time : datetime
            The start of the time range to fetch.
        end_time : datetime
            The end of the time range to fetch.
        bbox : Tuple[float, float, float, float]
            The bounding box in the format (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the satellite data. It is expected to have
            'lat', 'lon', and 'time' coordinates, and variables such as 'frp'
            (fire radiative power) and 'plume_height'.
        """
        pass


class SyntheticIngestor(SatelliteIngestor):
    """
    Generates synthetic 'Truth' data for demonstration purposes.

    This class simulates fire event data, providing a consistent source of
    plume observations for testing and development without requiring access to
    real satellite data.
    """

    def __init__(self, n_samples: int = 200):
        """
        Initializes the SyntheticIngestor.

        Parameters
        ----------
        n_samples : int, optional
            The number of synthetic fire events to generate, by default 200.
        """
        self.n_samples = n_samples

    def fetch_data(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Generates a synthetic dataset of fire events.

        The generated data simulates observations within the specified time
        range and bounding box.

        Parameters
        ----------
        start_time : datetime
            The start of the time range for data generation.
        end_time : datetime
            The end of the time range for data generation.
        bbox : Tuple[float, float, float, float]
            The bounding box (min_lon, min_lat, max_lon, max_lat) for the data.

        Returns
        -------
        xr.Dataset
            A Dataset containing the synthetic fire events.
        """
        print("\n--- Generating Synthetic Satellite Data ---")
        min_lon, min_lat, max_lon, max_lat = bbox
        time_delta = end_time - start_time
        total_seconds = time_delta.total_seconds()

        # 1. Create Synthetic Fire Events
        times = [
            start_time + timedelta(seconds=np.random.uniform(0, total_seconds))
            for _ in range(self.n_samples)
        ]
        lats = np.random.uniform(min_lat, max_lat, self.n_samples)
        lons = np.random.uniform(min_lon, max_lon, self.n_samples)
        frp = np.random.exponential(500, self.n_samples) + 50
        plume_height = 200 * (frp**0.4) + np.random.normal(0, 300, self.n_samples)
        plume_height = np.maximum(plume_height, 100)  # Ensure positive height

        ds = xr.Dataset(
            {
                "frp": (("event",), frp),
                "plume_height": (("event",), plume_height),
            },
            coords={
                "time": (("event",), times),
                "lat": (("event",), lats),
                "lon": (("event",), lons),
            },
        )

        # 2. Add Provenance
        history = (
            f"{datetime.now().isoformat()}: Generated synthetic fire events. "
            f"Time range: {start_time.isoformat()} to {end_time.isoformat()}. "
            f"Bounding box: {bbox}."
        )
        ds.attrs["history"] = history
        ds.attrs["source"] = "SyntheticIngestor"

        print(f"   Generated {self.n_samples} synthetic fire events.")
        return ds


class TropomiIngestor(SatelliteIngestor):
    """
    Ingests TROPOMI data from the Copernicus Open Access Hub.
    """

    def __init__(self):
        """
        Initializes the TropomiIngestor.

        Raises
        ------
        ValueError
            If the COPERNICUS_USERNAME or COPERNICUS_PASSWORD environment
            variables are not set.
        """
        user = os.getenv("COPERNICUS_USERNAME")
        password = os.getenv("COPERNICUS_PASSWORD")
        if not user or not password:
            raise ValueError(
                "COPERNICUS_USERNAME and COPERNICUS_PASSWORD environment variables must be set."
            )
        self.api = SentinelAPI(user, password, "https://apihub.copernicus.eu/apihub")

    def fetch_data(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Fetches TROPOMI data for a given time range and bounding box.

        Parameters
        ----------
        start_time : datetime
            The start of the time range to fetch.
        end_time : datetime
            The end of the time range to fetch.
        bbox : Tuple[float, float, float, float]
            The bounding box in the format (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the TROPOMI data.
        """
        min_lon, min_lat, max_lon, max_lat = bbox
        footprint = (
            f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
            f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
        )

        products = self.api.query(
            footprint,
            date=(start_time, end_time),
            platformname="Sentinel-5 Precursor",
            producttype="L2__AER_LH",
        )

        if not products:
            return xr.Dataset()

        product_ids = list(products.keys())
        self.api.download_all(product_ids)

        datasets = [
            xr.open_dataset(f"{products[pid]['title']}.nc", group="PRODUCT")
            for pid in product_ids
        ]
        combined_ds = xr.concat(datasets, dim="time")

        history = (
            f"{datetime.now().isoformat()}: Fetched TROPOMI data. "
            f"Time range: {start_time.isoformat()} to {end_time.isoformat()}. "
            f"Bounding box: {bbox}."
        )
        combined_ds.attrs["history"] = (
            history + "\n" + combined_ds.attrs.get("history", "")
        )
        return combined_ds


class OmpsIngestor(SatelliteIngestor):
    """
    Ingests OMPS data from NASA Earthdata.
    """

    def __init__(self):
        """
        Initializes the OmpsIngestor.

        Raises
        ------
        ValueError
            If the EARTHDATA_USERNAME or EARTHDATA_PASSWORD environment
            variables are not set.
        """
        if not (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")):
            raise ValueError(
                "EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables must be set."
            )
        self.auth = earthaccess.login()

    def fetch_data(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Fetches OMPS data for a given time range and bounding box.

        Parameters
        ----------
        start_time : datetime
            The start of the time range to fetch.
        end_time : datetime
            The end of the time range to fetch.
        bbox : Tuple[float, float, float, float]
            The bounding box in the format (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the OMPS data.
        """
        results = earthaccess.search_data(
            short_name="OMPS_NPP_L2_AER_DAILY",
            bounding_box=bbox,
            temporal=(start_time.isoformat(), end_time.isoformat()),
        )
        if not results:
            return xr.Dataset()

        file_paths = earthaccess.download(results, local_path="data")
        datasets = [xr.open_dataset(path, engine="h5netcdf") for path in file_paths]
        combined_ds = xr.concat(datasets, dim="time")

        history = (
            f"{datetime.now().isoformat()}: Fetched OMPS data. "
            f"Time range: {start_time.isoformat()} to {end_time.isoformat()}. "
            f"Bounding box: {bbox}."
        )
        combined_ds.attrs["history"] = (
            history + "\n" + combined_ds.attrs.get("history", "")
        )
        return combined_ds


class TempoIngestor(SatelliteIngestor):
    """
    Ingests TEMPO data from NASA Earthdata.
    """

    def __init__(self):
        """
        Initializes the TempoIngestor.

        Raises
        ------
        ValueError
            If the EARTHDATA_USERNAME or EARTHDATA_PASSWORD environment
            variables are not set.
        """
        if not (os.getenv("EARTHDATA_USERNAME") and os.getenv("EARTHDATA_PASSWORD")):
            raise ValueError(
                "EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables must be set."
            )
        self.auth = earthaccess.login()

    def fetch_data(
        self,
        start_time: datetime,
        end_time: datetime,
        bbox: Tuple[float, float, float, float],
    ) -> xr.Dataset:
        """
        Fetches TEMPO data for a given time range and bounding box.

        Parameters
        ----------
        start_time : datetime
            The start of the time range to fetch.
        end_time : datetime
            The end of the time range to fetch.
        bbox : Tuple[float, float, float, float]
            The bounding box in the format (min_lon, min_lat, max_lon, max_lat).

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the TEMPO data.
        """
        results = earthaccess.search_data(
            short_name="TEMPO_L2_AERDI_V1",
            bounding_box=bbox,
            temporal=(start_time.isoformat(), end_time.isoformat()),
        )
        if not results:
            return xr.Dataset()

        file_paths = earthaccess.download(results, local_path="data")
        datasets = [xr.open_dataset(path, engine="h5netcdf") for path in file_paths]
        combined_ds = xr.concat(datasets, dim="time")

        history = (
            f"{datetime.now().isoformat()}: Fetched TEMPO data. "
            f"Time range: {start_time.isoformat()} to {end_time.isoformat()}. "
            f"Bounding box: {bbox}."
        )
        combined_ds.attrs["history"] = (
            history + "\n" + combined_ds.attrs.get("history", "")
        )
        return combined_ds
