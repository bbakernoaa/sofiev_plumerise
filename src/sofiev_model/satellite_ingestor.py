from __future__ import annotations
from abc import ABC, abstractmethod
import os
import pandas as pd
from datetime import datetime
from typing import List
import earthaccess
import xarray as xr
from sentinelsat import SentinelAPI
from .gfs_ingestor import GFSIngestor

class SatelliteIngestor(ABC):
    """
    Abstract base class for satellite data ingestors.
    """
    @abstractmethod
    def get_data(self, start_date: datetime, end_date: datetime, aoi: List[float]) -> pd.DataFrame:
        """
        Fetches satellite data for a given time range and area of interest.

        Parameters
        ----------
        start_date : datetime
            The start date of the time range.
        end_date : datetime
            The end date of the time range.
        aoi : List[float]
            The area of interest, defined as [lon_min, lat_min, lon_max, lat_max].

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the satellite data.
        """
        pass

class TropomiIngestor(SatelliteIngestor):
    """
    Ingests TROPOMI data.
    """
    def get_data(self, start_date: datetime, end_date: datetime, aoi: List[float]) -> pd.DataFrame:
        print("Fetching TROPOMI data...")
        api = SentinelAPI('s5pguest', 's5pguest', 'https://s5phub.copernicus.eu/dhus')

        # Format AOI for the query
        footprint = f'POLYGON(({aoi[0]} {aoi[1]}, {aoi[2]} {aoi[1]}, {aoi[2]} {aoi[3]}, {aoi[0]} {aoi[3]}, {aoi[0]} {aoi[1]}))'

        products = api.query(
            area=footprint,
            date=(start_date, end_date),
            producttype='L2__SO2___',
        )

        if not products:
            print("No TROPOMI products found.")
            return pd.DataFrame()

        # Download the first product
        product_id = list(products.keys())[0]
        product_info = api.get_product_odata(product_id)

        # Check if file already exists
        filepath = f"./{product_info['title']}"
        if not os.path.exists(filepath):
            print(f"Downloading {product_info['title']}...")
            api.download(product_id)
        else:
            print(f"Using existing file: {product_info['title']}")

        # Process the NetCDF file
        with xr.open_dataset(filepath, group='PRODUCT') as ds:
            # Extract data
            lat = ds['latitude'].values.flatten()
            lon = ds['longitude'].values.flatten()
            so2 = ds['sulfurdioxide_total_vertical_column'].values.flatten()

            # The time variable is a single value for the granule
            time_val = pd.to_datetime(ds['time_utc'].values[0])
            times = [time_val] * len(lat)

            df = pd.DataFrame({
                'time': times,
                'lat': lat,
                'lon': lon,
                'so2': so2,
            })

        return df

class TempoIngestor(SatelliteIngestor):
    """
    Ingests TEMPO data.
    """
    def get_data(self, start_date: datetime, end_date: datetime, aoi: List[float]) -> pd.DataFrame:
        print("Fetching TEMPO data...")
        earthaccess.login(strategy="environment")

        results = earthaccess.search_data(
            short_name='TEMPO_NRT_L2_NO2_V01',
            bounding_box=(aoi[0], aoi[1], aoi[2], aoi[3]),
            temporal=(start_date.isoformat(), end_date.isoformat()),
            count=1
        )

        if not results:
            print("No TEMPO products found.")
            return pd.DataFrame()

        filepaths = earthaccess.download(results, local_path=".")
        filepath = filepaths[0]

        with xr.open_dataset(filepath, group='product_data') as ds:
            lat = ds['latitude'].values.flatten()
            lon = ds['longitude'].values.flatten()
            no2 = ds['nitrogendioxide_tropospheric_vertical_column'].values.flatten()

            # The time variable is a single value for the granule
            time_val = pd.to_datetime(ds['time'].values)
            times = [time_val] * len(lat)

            df = pd.DataFrame({
                'time': times,
                'lat': lat,
                'lon': lon,
                'no2': no2,
            })

        return df

class OmpsIngestor(SatelliteIngestor):
    """
    Ingests OMPS data.
    """
    def get_data(self, start_date: datetime, end_date: datetime, aoi: List[float]) -> pd.DataFrame:
        print("Fetching OMPS data...")
        earthaccess.login(strategy="environment")

        results = earthaccess.search_data(
            short_name='OMPS_NPP_NMSO2_L2_NRT_V2',
            bounding_box=(aoi[0], aoi[1], aoi[2], aoi[3]),
            temporal=(start_date.isoformat(), end_date.isoformat()),
            count=1
        )

        if not results:
            print("No OMPS products found.")
            return pd.DataFrame()

        filepaths = earthaccess.download(results, local_path=".")
        filepath = filepaths[0]

        with xr.open_dataset(filepath, group='Data Fields') as ds:
            lat = ds['Latitude'].values.flatten()
            lon = ds['Longitude'].values.flatten()
            so2 = ds['ColumnAmountSO2_PBL'].values.flatten()

            # The time variable is a single value for the granule
            time_val = pd.to_datetime(ds['Time'].values)
            times = [time_val] * len(lat)

            df = pd.DataFrame({
                'time': times,
                'lat': lat,
                'lon': lon,
                'so2_omps': so2,
            })

        return df

def load_and_collocate_data(ingestors: List[SatelliteIngestor], gfs_ingestor: GFSIngestor,
                              start_date: datetime, end_date: datetime, aoi: List[float]) -> pd.DataFrame:
    """
    Loads data from multiple satellite ingestors and collocates it with GFS data.

    Parameters
    ----------
    ingestors : List[SatelliteIngestor]
        A list of satellite ingestor instances.
    gfs_ingestor : GFSIngestor
        An instance of the GFSIngestor class.
    start_date : datetime
        The start date of the time range.
    end_date : datetime
        The end date of the time range.
    aoi : List[float]
        The area of interest.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the combined and collocated data.
    """
    all_data = []
    for ingestor in ingestors:
        all_data.append(ingestor.get_data(start_date, end_date, aoi))

    df = pd.concat(all_data, ignore_index=True)

    if df.empty:
        return df

    # Collocation with GFS data (placeholder for now)
    h_abl_list = []
    n_ft_list = []
    wind_list = []

    for index, row in df.iterrows():
        h, w, n = gfs_ingestor.get_analysis_point(row['time'], row['lat'], row['lon'])
        h_abl_list.append(h)
        n_ft_list.append(n)
        wind_list.append(w)

    df['h_abl'] = h_abl_list
    df['n_ft'] = n_ft_list
    df['wind_speed'] = wind_list

    return df.dropna()
