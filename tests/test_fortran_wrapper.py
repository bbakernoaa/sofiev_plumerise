import numpy as np
import pytest
from numpy.testing import assert_allclose

# Before importing the wrapper, ensure the project is built so the .so file exists
from sofiev_model.fortran_wrapper import (
    plume_rise,
    distribute_vertical_emissions,
)

def test_plume_rise_wrapper():
    """
    Tests the Python wrapper for the plumeRiseSofiev subroutine.
    Values are the same as in the Fortran CTest.
    """
    # Case 1: Plume within Boundary Layer
    hp1 = plume_rise(n2=1.0e-4, frp=10.0e6, pblh=1000.0)
    assert_allclose(hp1, 539.4, rtol=1e-4)

    # Case 2: Plume penetrates Free Troposphere
    hp2 = plume_rise(n2=2.0e-4, frp=500.0e6, pblh=1000.0)
    assert_allclose(hp2, 1311.84, rtol=1e-4)

    # Case 3: Numerical floor
    hp3 = plume_rise(n2=1.0e-3, frp=1.0, pblh=1.0)
    assert_allclose(hp3, 10.0, rtol=1e-4)

def test_distribute_emissions_wrapper():
    """
    Tests the Python wrapper for the distribute_emissions subroutine.
    Values are the same as in the Fortran CTest.
    """
    zf = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64)
    u = np.full_like(zf, 2.0)
    base_emis = 100.0
    n2 = 1.0e-4
    plm_hgt = 250.0

    # Test 1: Uniform distribution
    emis1 = distribute_vertical_emissions(
        zf, u, n2, plm_hgt, base_emis, use_beta_dist=False, use_wind_adj=False
    )
    expected1 = np.array([40.0, 40.0, 20.0, 0.0, 0.0])
    assert_allclose(emis1, expected1, rtol=1e-6)
    assert_allclose(np.sum(emis1), base_emis, rtol=1e-6)

    # Test 2: Beta distribution
    emis2 = distribute_vertical_emissions(
        zf, u, n2, plm_hgt, base_emis, use_beta_dist=True, use_wind_adj=False
    )
    expected2 = np.array([17.92, 64.00, 18.08, 0.0, 0.0])
    assert_allclose(emis2, expected2, rtol=1e-3)
    assert_allclose(np.sum(emis2), base_emis, rtol=1e-6)

    # Test 3: Wind adjustment
    plm_hgt_windy = 800.0
    u_windy = np.full_like(zf, 10.0)
    n2_windy = 2.5e-4
    emis3 = distribute_vertical_emissions(
        zf, u_windy, n2_windy, plm_hgt_windy, base_emis, use_beta_dist=False, use_wind_adj=True
    )
    # Expected Hp_eff = 800 * (5.0/10.0)**(0.5 * (1+1)) = 400.0
    expected3 = np.array([25.0, 25.0, 25.0, 25.0, 0.0])
    assert_allclose(emis3, expected3, rtol=1e-6)
    assert_allclose(np.sum(emis3), base_emis, rtol=1e-6)

    # Test 4: Zero plume height
    emis4 = distribute_vertical_emissions(
        zf, u, n2, 0.0, base_emis
    )
    expected4 = np.zeros_like(zf)
    assert_allclose(emis4, expected4, rtol=1e-6)
