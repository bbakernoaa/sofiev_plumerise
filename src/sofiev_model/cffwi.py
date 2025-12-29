import numpy as np

class FWI_Engine_Vectorized:
    @staticmethod
    def calculate_ffmc(temp, rh, wind, precip, ffmc_prev):
        """Fine Fuel Moisture Code (FFMC) - Responds to hourly/daily weather"""
        # Rain effect
        rf = precip - 0.5
        mo = (147.2 * (101.0 - ffmc_prev)) / (59.5 + ffmc_prev)
        
        # Adjust moisture for rain
        mr = np.where(precip > 0.5,
                      np.where(mo <= 150.0, 
                               mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)),
                               mo + 42.5 * rf * np.exp(-100.0 / (251.0 - mo)) * (1.0 - np.exp(-6.93 / rf)) + 0.0015 * (mo - 150.0)**2 * np.sqrt(rf)),
                      mo)
        mr = np.minimum(mr, 250.0)
        
        # Equilibrium Moisture Content (EMC)
        ed = 0.942 * (rh**0.679) + 11.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        ew = 0.618 * (rh**0.753) + 10.0 * np.exp((rh - 100.0) / 10.0) + 0.18 * (21.1 - temp) * (1.0 - np.exp(-0.115 * rh))
        
        # Drying/Wetting logic
        k1 = 0.424 * (1.0 - ((100.0 - rh) / 100.0)**1.7) + 0.0694 * np.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0)**8)
        kw = 0.307 * (1.0 - ((100.0 - rh) / 100.0)**1.7) + 0.0512 * np.sqrt(wind) * (1.0 - ((100.0 - rh) / 100.0)**8)
        
        m = np.where(mr > ed, 
                     ed + (mr - ed) * 10**(-k1 * 0.581 * np.exp(0.0365 * temp)), # Drying
                     np.where(mr < ew, 
                              ew - (ew - mr) * 10**(-kw * 0.581 * np.exp(0.0365 * temp)), # Wetting
                              mr))
        
        return (59.5 * (147.2 - m)) / (147.2 + m)

    @staticmethod
    def calculate_dmc(temp, rh, precip, dmc_prev, month):
        """Duff Moisture Code (DMC) - Medium-term moisture (weeks)"""
        # Latitudinal adjustment for day length
        ell_f = [6.5, 7.5, 9.0, 12.8, 13.9, 13.9, 12.4, 10.9, 9.4, 8.0, 7.0, 6.0]
        L = ell_f[month-1]
        
        temp_c = np.maximum(temp, -1.1)
        # Rain effect
        ra = np.where(precip > 1.5, 0.92 * precip - 1.27, precip)
        mo = 20.0 + np.exp(5.6348 - dmc_prev / 43.43)
        
        # Rain adjustment to DMC
        b = np.where(dmc_prev <= 33.0, 
                     100.0 / (0.5 + 0.3 * dmc_prev),
                     np.where(dmc_prev <= 65.0, 
                              14.0 - 1.3 * np.log(dmc_prev), 
                              6.2 * np.log(dmc_prev) - 17.2))
        
        mr = np.where(precip > 1.5, mo + 1000.0 * ra / (48.77 + b * ra), mo)
        dmc_rain = 43.43 * (5.6348 - np.log(mr - 20.0))
        
        dmc_base = np.where(precip > 1.5, np.maximum(dmc_rain, 0.0), dmc_prev)
        
        # Drying phase
        k = 1.894 * (temp_c + 1.1) * (100.0 - rh) * L * 1e-4
        return dmc_base + k

    @staticmethod
    def calculate_dc(temp, precip, dc_prev, month):
        """Drought Code (DC) - Long-term moisture (months/seasonal legacy)"""
        # Day length factor
        lfv = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]
        L = lfv[month-1]
        
        temp_c = np.maximum(temp, -2.8)
        # Rain effect
        rd = np.where(precip > 2.8, 0.83 * precip - 1.27, precip)
        qo = 800.0 * np.exp(-dc_prev / 400.0)
        qr = np.where(precip > 2.8, qo + 3.937 * rd, qo)
        dc_rain = 400.0 * np.log(800.0 / np.maximum(qr, 1e-6))
        
        dc_base = np.where(precip > 2.8, np.maximum(dc_rain, 0.0), dc_prev)
        
        # Evapotranspiration phase
        pe = (0.36 * (temp_c + 2.8) + L) * 0.5
        return dc_base + np.maximum(pe, 0.0)

    @staticmethod
    def calculate_isi(ffmc, wind):
        """Initial Spread Index (ISI)"""
        f_wind = np.exp(0.05039 * wind)
        m = 147.2 * (101.0 - ffmc) / (59.5 + ffmc)
        f_m = 91.9 * np.exp(-0.1386 * m) * (1.0 + (m**5.31) / 4.93e7)
        return 0.208 * f_wind * f_m

    @staticmethod
    def calculate_bui(dmc, dc):
        """Build-Up Index (BUI) - Total fuel available"""
        bui = np.where(dmc <= 0.4 * dc,
                       (0.8 * dmc * dc) / (dmc + 0.4 * dc),
                       dmc - (1.0 - 0.8 * dc / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc)**1.7))
        return np.maximum(bui, 0.0)

    @staticmethod
    def calculate_fwi(isi, bui):
        """Fire Weather Index (FWI) - Final intensity rating"""
        f_d = np.where(bui <= 80.0, 
                       0.626 * (bui**0.809) + 2.0, 
                       1000.0 / (25.0 + 108.64 * np.exp(-0.023 * bui)))
        b = 0.1 * isi * f_d
        fwi = np.where(b > 1.01, 
                       np.exp(2.72 * (0.434 * np.log(b))**0.647), 
                       b)
        return fwi
