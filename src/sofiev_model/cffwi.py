import xarray as xr


class FWI_Engine_Vectorized:
    """
    A vectorized engine for calculating the Canadian Fire Weather Index (FWI) system.

    This class contains static methods that are designed to work with Dask-aware
    xarray.DataArray objects, allowing for lazy, parallelized computation of
    the FWI components. All calculations are based on the original technical
    report by Van Wagner & Pickett (1985).
    """

    @staticmethod
    def calculate_ffmc(
        temp: xr.DataArray,
        rh: xr.DataArray,
        wind: xr.DataArray,
        precip: xr.DataArray,
        ffmc_prev: xr.DataArray,
    ) -> xr.DataArray:
        """
        Calculate the Fine Fuel Moisture Code (FFMC).

        FFMC is a numeric rating of the moisture content of litter and other
        cured fine fuels. This component is intended to represent the moisture
        content of an approximate half-inch layer of fuel on the forest floor.
        It is sensitive to daily changes in weather.

        Parameters
        ----------
        temp : xr.DataArray
            2m Temperature in Celsius.
        rh : xr.DataArray
            2m Relative Humidity in percent.
        wind : xr.DataArray
            10m Wind speed in km/h.
        precip : xr.DataArray
            24-hour accumulated precipitation in mm.
        ffmc_prev : xr.DataArray
            The previous day's FFMC.

        Returns
        -------
        xr.DataArray
            The calculated Fine Fuel Moisture Code for the current day.
        """
        # Original FFMC calculation from the Fortran code.
        # This is a direct, vectorized implementation.
        mo = (147.2 * (101.0 - ffmc_prev)) / (59.5 + ffmc_prev)
        rf = precip - 0.5

        # Adjust moisture content for rain
        # Condition for mo > 150.0
        mo_gt_150 = (
            mo
            + 42.5 * rf * xr.ufuncs.exp(-100.0 / (251.0 - mo)) * (1.0 - xr.ufuncs.exp(-6.93 / rf))
            + 0.0015 * (mo - 150.0) ** 2 * xr.ufuncs.sqrt(rf)
        )
        # Condition for mo <= 150.0
        mo_le_150 = mo + 42.5 * rf * xr.ufuncs.exp(-100.0 / (251.0 - mo)) * (
            1.0 - xr.ufuncs.exp(-6.93 / rf)
        )

        # Apply conditions using xr.where
        mr_intermediate = xr.where(mo <= 150.0, mo_le_150, mo_gt_150)
        mr = xr.where(precip > 0.5, mr_intermediate, mo)
        mr = xr.where(mr > 250.0, 250.0, mr)  # Cap moisture at 250

        # Equilibrium Moisture Content (EMC) for drying and wetting
        ed = (
            0.942 * (rh**0.679)
            + 11.0 * xr.ufuncs.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - xr.ufuncs.exp(-0.115 * rh))
        )
        ew = (
            0.618 * (rh**0.753)
            + 10.0 * xr.ufuncs.exp((rh - 100.0) / 10.0)
            + 0.18 * (21.1 - temp) * (1.0 - xr.ufuncs.exp(-0.115 * rh))
        )

        # Calculate final moisture content (m) based on drying or wetting
        k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0) ** 1.7) + 0.0694 * xr.ufuncs.sqrt(wind) * (
            1.0 - ((100.0 - rh) / 100.0) ** 8
        )
        kw = 0.307 * (1.0 - ((100.0 - rh) / 100.0)**1.7) + 0.0512 * xr.ufuncs.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0)**8)

        m_drying = ed + (mr - ed) * 10 ** (-k1 * 0.581 * xr.ufuncs.exp(0.0365 * temp))
        m_wetting = ew - (ew - mr) * 10 ** (-kw * 0.581 * xr.ufuncs.exp(0.0365 * temp))

        m = xr.where(mr > ed, m_drying, xr.where(mr < ew, m_wetting, mr))

        # Convert final moisture content to FFMC scale
        ffmc = (59.5 * (250.0 - m)) / (147.2 + m)
        return ffmc.where(ffmc > 0, 0)  # Ensure FFMC is not negative

    @staticmethod
    def calculate_dmc(
        temp: xr.DataArray,
        rh: xr.DataArray,
        precip: xr.DataArray,
        dmc_prev: xr.DataArray,
        month: int,
    ) -> xr.DataArray:
        """
        Calculate the Duff Moisture Code (DMC).

        DMC is a numeric rating of the average moisture content of loosely
        compacted organic layers of moderate depth. This code is calculated
        for mid-afternoon fire weather observations.

        Parameters
        ----------
        temp : xr.DataArray
            2m Temperature in Celsius.
        rh : xr.DataArray
            2m Relative Humidity in percent.
        precip : xr.DataArray
            24-hour accumulated precipitation in mm.
        dmc_prev : xr.DataArray
            The previous day's DMC.
        month : int
            The calendar month (1-12).

        Returns
        -------
        xr.DataArray
            The calculated Duff Moisture Code for the current day.
        """
        # Day length adjustment based on latitude, simplified to a monthly constant
        ell_f = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        L = ell_f[month - 1]

        t = xr.where(temp < -1.1, -1.1, temp) # Temperature floor
        
        # Effective rainfall
        re = xr.where(precip > 1.5, 0.92 * precip - 1.27, precip)

        # Moisture content before rain
        mo = 20.0 + xr.ufuncs.exp(5.6348 - dmc_prev / 43.43)

        # Duff moisture content after rain
        b_le_33 = 100.0 / (0.5 + 0.3 * dmc_prev)
        b_le_65 = 14.0 - 1.3 * xr.ufuncs.log(dmc_prev)
        b_gt_65 = 6.2 * xr.ufuncs.log(dmc_prev) - 17.2
        
        b = xr.where(dmc_prev <= 33.0, b_le_33, xr.where(dmc_prev <= 65.0, b_le_65, b_gt_65))
        mr = mo + 1000.0 * re / (48.77 + b * re)
        pr = 43.43 * (5.6348 - xr.ufuncs.log(mr - 20.0))
        pr = pr.where(pr < 0, 0) # ensure pr is not negative
        
        dmc = xr.where(precip > 1.5, pr, dmc_prev)

        # Drying potential
        k = 1.894 * (t + 1.1) * (100.0 - rh) * L * 1e-4
        return dmc + k

    @staticmethod
    def calculate_dc(
        temp: xr.DataArray, precip: xr.DataArray, dc_prev: xr.DataArray, month: int
    ) -> xr.DataArray:
        """
        Calculate the Drought Code (DC).

        DC is a numeric rating of the average moisture content of deep, compact
        organic layers. This code is a useful indicator of seasonal drought
        effects on forest fuels.

        Parameters
        ----------
        temp : xr.DataArray
            2m Temperature in Celsius.
        precip : xr.DataArray
            24-hour accumulated precipitation in mm.
        dc_prev : xr.DataArray
            The previous day's DC.
        month : int
            The calendar month (1-12).

        Returns
        -------
        xr.DataArray
            The calculated Drought Code for the current day.
        """
        # Day length adjustment
        lfv = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        L = lfv[month - 1]

        t = xr.where(temp < -2.8, -2.8, temp) # Temperature floor

        # Effective rainfall
        pe = xr.where(precip > 2.8, 0.83 * precip - 1.27, precip)

        # Moisture equivalent before rain
        qo = 800.0 * xr.ufuncs.exp(-dc_prev / 400.0)
        qr = qo + 3.937 * pe
        
        # Drought Code after rain
        dr = 400.0 * xr.ufuncs.log(800.0 / qr)
        dr = dr.where(dr < 0, 0) # ensure dr is not negative
        
        dc = xr.where(precip > 2.8, dr, dc_prev)
        
        # Potential evapotranspiration (drying)
        v = (0.36 * (t + 2.8) + L) * 0.5
        return dc + v.where(v > 0, 0)

    @staticmethod
    def calculate_isi(ffmc: xr.DataArray, wind: xr.DataArray) -> xr.DataArray:
        """
        Calculate the Initial Spread Index (ISI).

        ISI is a numeric rating of the expected rate of fire spread. It combines
        the effects of wind and FFMC on fire spread.

        Parameters
        ----------
        ffmc : xr.DataArray
            The Fine Fuel Moisture Code.
        wind : xr.DataArray
            10m Wind speed in km/h.

        Returns
        -------
        xr.DataArray
            The calculated Initial Spread Index.
        """
        mo = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        fm = 91.9 * xr.ufuncs.exp(-0.1386 * mo) * (1.0 + (mo**5.31) / 4.93e7)
        fw = xr.ufuncs.exp(0.05039 * wind)
        return 0.208 * fw * fm

    @staticmethod
    def calculate_bui(dmc: xr.DataArray, dc: xr.DataArray) -> xr.DataArray:
        """
        Calculate the Buildup Index (BUI).

        BUI is a numeric rating of the total amount of fuel available for
        combustion. It combines the DMC and the DC.

        Parameters
        ----------
        dmc : xr.DataArray
            The Duff Moisture Code.
        dc : xr.DataArray
            The Drought Code.

        Returns
        -------
        xr.DataArray
            The calculated Buildup Index.
        """
        bui_le = (0.8 * dmc * dc) / (dmc + 0.4 * dc)
        bui_gt = dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (
            0.92 + (0.0114 * dmc) ** 1.7
        )
        bui = xr.where(dmc <= 0.4 * dc, bui_le, bui_gt)
        return xr.where(bui < 0, 0, bui) # ensure bui is not negative

    @staticmethod
    def calculate_fwi(isi: xr.DataArray, bui: xr.DataArray) -> xr.DataArray:
        """
        Calculate the Fire Weather Index (FWI).

        FWI is a numeric rating of fire intensity. It is suitable as a general
        index of fire danger throughout the forested areas of Canada.

        Parameters
        ----------
        isi : xr.DataArray
            The Initial Spread Index.
        bui : xr.DataArray
            The Buildup Index.

        Returns
        -------
        xr.DataArray
            The final Fire Weather Index.
        """
        fd_le = 0.626 * (bui**0.809) + 2.0
        fd_gt = 1000.0 / (25.0 + 108.64 * xr.ufuncs.exp(-0.023 * bui))
        fd = xr.where(bui <= 80.0, fd_le, fd_gt)

        b = 0.1 * isi * fd

        fwi_gt = xr.ufuncs.exp(2.72 * (0.434 * xr.ufuncs.log(b)) ** 0.647)
        fwi = xr.where(b > 1.0, fwi_gt, b)
        return fwi
