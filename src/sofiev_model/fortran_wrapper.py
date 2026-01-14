import ctypes
import numpy as np
import os
import platform

# --- Load the shared library ---

# Construct the full path to the library file based on the OS
lib_name = "libsofiev.so"
if platform.system() == "Darwin":
    lib_name = "libsofiev.dylib"
elif platform.system() == "Windows":
    lib_name = "libsofiev.dll"

# The library is expected to be in the same directory as this wrapper
lib_path = os.path.join(os.path.dirname(__file__), lib_name)

if not os.path.exists(lib_path):
    raise FileNotFoundError(
        f"Shared library not found at {lib_path}. "
        "Please ensure the Fortran code has been compiled and the "
        "library is in the correct location."
    )

fortran_lib = ctypes.CDLL(lib_path)

# --- Define argument and return types for the C functions ---

# 1. plume_rise_sofiev_c
plume_rise_sofiev = fortran_lib.plume_rise_sofiev_c
plume_rise_sofiev.argtypes = [
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.POINTER(ctypes.c_double),
]
plume_rise_sofiev.restype = None

# 2. distribute_emissions_c
distribute_emissions = fortran_lib.distribute_emissions_c
distribute_emissions.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_bool,
    ctypes.c_bool,
    np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
]
distribute_emissions.restype = None

# --- Create user-friendly Python wrappers ---


def plume_rise(n2, frp, pblh):
    """
    Calculates the plume rise height using the Sofiev algorithm.

    Args:
        n2 (float): Brunt-Vaisala frequency squared (s^-2).
        frp (float): Fire Radiative Power (W).
        pblh (float): Planetary Boundary Layer height (m).

    Returns:
        float: The calculated plume top height (m).
    """
    hp = ctypes.c_double()
    plume_rise_sofiev(
        ctypes.c_double(n2),
        ctypes.c_double(frp),
        ctypes.c_double(pblh),
        ctypes.byref(hp),
    )
    return hp.value


def distribute_vertical_emissions(
    zf, u, n2, plm_hgt, base_emis, use_beta_dist=False, use_wind_adj=False
):
    """
    Distributes surface emissions into a vertical column.

    Args:
        zf (np.ndarray): 1D array of layer interface heights (m).
        u (np.ndarray): 1D array of wind speed profile (m/s).
        n2 (float): Brunt-Vaisala frequency squared (s^-2).
        plm_hgt (float): The plume top height (m).
        base_emis (float): Total surface emission rate.
        use_beta_dist (bool): Whether to use the Beta PDF for distribution.
        use_wind_adj (bool): Whether to adjust plume height for wind.

    Returns:
        np.ndarray: A 1D array of the vertical emission profile.
    """
    n_layers = len(zf)
    # Ensure arrays are C-contiguous and of the correct type (float64)
    zf_c = np.ascontiguousarray(zf, dtype=np.float64)
    u_c = np.ascontiguousarray(u, dtype=np.float64)
    emis_c = np.empty_like(zf_c)

    distribute_emissions(
        ctypes.c_int(n_layers),
        zf_c,
        u_c,
        ctypes.c_double(n2),
        ctypes.c_double(plm_hgt),
        ctypes.c_double(base_emis),
        ctypes.c_bool(use_beta_dist),
        ctypes.c_bool(use_wind_adj),
        emis_c,
    )
    return emis_c
