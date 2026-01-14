program test_plumerise
  use plumerise_sofiev_mod
  implicit none

  integer :: n_failed
  n_failed = 0

  call test_find_height_index(n_failed)
  call test_plumeRiseSofiev(n_failed)
  call test_distribute_emissions(n_failed)

  if (n_failed == 0) then
    print *, "All Fortran tests passed!"
  else
    print *, n_failed, " Fortran tests failed."
  end if

  call exit(n_failed)

contains

  subroutine assert_real_equals(val, expected, test_name, n_failed, tol)
    real(rk), intent(in)    :: val, expected, tol
    character(*), intent(in) :: test_name
    integer, intent(inout) :: n_failed
    if (abs(val - expected) > tol) then
      print *, "FAIL: ", test_name
      print *, "  Expected: ", expected, " Got: ", val
      n_failed = n_failed + 1
    endif
  end subroutine assert_real_equals

  subroutine assert_int_equals(val, expected, test_name, n_failed)
    integer, intent(in)    :: val, expected
    character(*), intent(in) :: test_name
    integer, intent(inout) :: n_failed
    if (val /= expected) then
      print *, "FAIL: ", test_name
      print *, "  Expected: ", expected, " Got: ", val
      n_failed = n_failed + 1
    endif
  end subroutine assert_int_equals

  subroutine test_find_height_index(n_failed)
    integer, intent(inout) :: n_failed
    real(rk), dimension(5)     :: ZF
    integer                :: idx

    print *, "--- Running tests for find_height_index ---"
    ZF = [100.0_rk, 200.0_rk, 300.0_rk, 500.0_rk, 1000.0_rk]

    call find_height_index(ZF, 250.0_rk, idx)
    call assert_int_equals(idx, 3, "hgt within range", n_failed)

    call find_height_index(ZF, 300.0_rk, idx)
    call assert_int_equals(idx, 3, "hgt on boundary", n_failed)

    call find_height_index(ZF, 50.0_rk, idx)
    call assert_int_equals(idx, 1, "hgt below first layer", n_failed)

    call find_height_index(ZF, 1200.0_rk, idx)
    call assert_int_equals(idx, 5, "hgt above last layer", n_failed)

    call find_height_index(ZF, 0.0_rk, idx)
    call assert_int_equals(idx, 1, "hgt is zero", n_failed)
  end subroutine test_find_height_index

  subroutine test_plumeRiseSofiev(n_failed)
    integer, intent(inout) :: n_failed
    real(rk)                   :: Hp

    print *, "--- Running tests for plumeRiseSofiev ---"

    call plumeRiseSofiev(1.0e-4_rk, 10.0e6_rk, 1000.0_rk, Hp)
    call assert_real_equals(Hp, 539.4_rk, "Within BL", n_failed, 0.1_rk)

    call plumeRiseSofiev(2.0e-4_rk, 500.0e6_rk, 1000.0_rk, Hp)
    call assert_real_equals(Hp, 1311.84_rk, "Penetrates FT", n_failed, 0.1_rk)

    call plumeRiseSofiev(1.0e-3_rk, 1.0_rk, 1.0_rk, Hp)
    call assert_real_equals(Hp, 10.0_rk, "Numerical Floor", n_failed, 0.1_rk)
  end subroutine test_plumeRiseSofiev

  subroutine test_distribute_emissions(n_failed)
      integer, intent(inout) :: n_failed
      real(rk), dimension(5) :: ZF, U, emis
      type(PlumeControl) :: config
      real(rk) :: base_emis, plmHGT, N2
      real(rk), parameter :: tol = 1.0e-6_rk

      print *, "--- Running tests for distribute_emissions ---"
      ZF = [100._rk, 200._rk, 300._rk, 400._rk, 500._rk]
      U  = [2.0_rk, 2.0_rk, 2.0_rk, 2.0_rk, 2.0_rk]
      base_emis = 100.0_rk
      N2 = 1.0e-4_rk

      config%use_beta_dist = .false.
      config%use_wind_adj = .false.
      plmHGT = 250.0_rk
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      call assert_real_equals(emis(1), 40.0_rk, "Uniform dist, layer 1", n_failed, tol)
      call assert_real_equals(emis(2), 40.0_rk, "Uniform dist, layer 2", n_failed, tol)
      call assert_real_equals(emis(3), 20.0_rk, "Uniform dist, layer 3", n_failed, tol)
      call assert_real_equals(emis(4), 0.0_rk,  "Uniform dist, layer 4", n_failed, tol)
      call assert_real_equals(sum(emis), base_emis, "Uniform dist, mass conservation", n_failed, tol)

      config%use_beta_dist = .true.
      config%use_wind_adj = .false.
      plmHGT = 250.0_rk
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      call assert_real_equals(emis(1), 17.92_rk, "Beta dist, layer 1", n_failed, 0.001_rk)
      call assert_real_equals(emis(2), 64.00_rk, "Beta dist, layer 2", n_failed, 0.001_rk)
      call assert_real_equals(emis(3), 18.08_rk, "Beta dist, layer 3", n_failed, 0.001_rk)
      call assert_real_equals(sum(emis), base_emis, "Beta dist, mass conservation", n_failed, tol)

      config%use_beta_dist = .false.
      config%use_wind_adj = .true.
      plmHGT = 800.0_rk
      U = [10._rk, 10._rk, 10._rk, 10._rk, 10._rk]
      N2 = 2.5e-4_rk
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      call assert_real_equals(emis(1), 25.0_rk, "Wind adj, layer 1", n_failed, tol)
      call assert_real_equals(emis(2), 25.0_rk, "Wind adj, layer 2", n_failed, tol)
      call assert_real_equals(emis(3), 25.0_rk, "Wind adj, layer 3", n_failed, tol)
      call assert_real_equals(emis(4), 25.0_rk, "Wind adj, layer 4", n_failed, tol)
      call assert_real_equals(emis(5), 0.0_rk,  "Wind adj, layer 5", n_failed, tol)
      call assert_real_equals(sum(emis), base_emis, "Wind adj, mass conservation", n_failed, tol)

      config%use_beta_dist = .false.
      config%use_wind_adj = .false.
      plmHGT = 0.0_rk
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      call assert_real_equals(sum(emis), 0.0_rk, "Zero plume height", n_failed, tol)

  end subroutine test_distribute_emissions

end program test_plumerise
