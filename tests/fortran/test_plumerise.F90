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

  ! Helper subroutine for comparing real numbers
  subroutine assert_real_equals(val, expected, test_name, n_failed, tol)
    real, intent(in)    :: val, expected, tol
    character(*), intent(in) :: test_name
    integer, intent(inout) :: n_failed
    if (abs(val - expected) > tol) then
      print *, "FAIL: ", test_name
      print *, "  Expected: ", expected, " Got: ", val
      n_failed = n_failed + 1
    endif
  end subroutine assert_real_equals

  ! Helper subroutine for comparing integer numbers
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


  ! Test suite for find_height_index
  subroutine test_find_height_index(n_failed)
    integer, intent(inout) :: n_failed
    real, dimension(5)     :: ZF
    integer                :: idx

    print *, "--- Running tests for find_height_index ---"
    ZF = [100.0, 200.0, 300.0, 500.0, 1000.0]

    call find_height_index(ZF, 250.0, idx)
    call assert_int_equals(idx, 3, "hgt within range", n_failed)

    call find_height_index(ZF, 300.0, idx)
    call assert_int_equals(idx, 3, "hgt on boundary", n_failed)

    call find_height_index(ZF, 50.0, idx)
    call assert_int_equals(idx, 1, "hgt below first layer", n_failed)

    call find_height_index(ZF, 1200.0, idx)
    call assert_int_equals(idx, 5, "hgt above last layer", n_failed)

    call find_height_index(ZF, 0.0, idx)
    call assert_int_equals(idx, 1, "hgt is zero", n_failed)
  end subroutine test_find_height_index


  ! Test suite for plumeRiseSofiev
  subroutine test_plumeRiseSofiev(n_failed)
    integer, intent(inout) :: n_failed
    real                   :: Hp

    print *, "--- Running tests for plumeRiseSofiev ---"

    ! Case 1: Plume within Boundary Layer
    call plumeRiseSofiev(1.0e-4, 10.0e6, 1000.0, Hp)
    call assert_real_equals(Hp, 539.4, "Within BL", n_failed, 0.1)

    ! Case 2: Plume penetrates Free Troposphere
    call plumeRiseSofiev(2.0e-4, 500.0e6, 1000.0, Hp)
    call assert_real_equals(Hp, 1311.84, "Penetrates FT", n_failed, 0.1)

    ! Case 3: Numerical floor
    call plumeRiseSofiev(1.0e-3, 1.0, 1.0, Hp)
    call assert_real_equals(Hp, 10.0, "Numerical Floor", n_failed, 0.1)
  end subroutine test_plumeRiseSofiev

  ! Test suite for distribute_emissions
  subroutine test_distribute_emissions(n_failed)
      integer, intent(inout) :: n_failed
      real, dimension(5) :: ZF, U, emis
      type(PlumeControl) :: config
      real :: base_emis, plmHGT, N2
      real, parameter :: tol = 1.0e-6

      print *, "--- Running tests for distribute_emissions ---"
      ZF = [100., 200., 300., 400., 500.]
      U  = [2.0, 2.0, 2.0, 2.0, 2.0]
      base_emis = 100.0
      N2 = 1.0e-4

      ! Test 1: Uniform distribution
      config%use_beta_dist = .false.
      config%use_wind_adj = .false.
      plmHGT = 250.0
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      call assert_real_equals(emis(1), 40.0, "Uniform dist, layer 1", n_failed, tol)
      call assert_real_equals(emis(2), 40.0, "Uniform dist, layer 2", n_failed, tol)
      call assert_real_equals(emis(3), 20.0, "Uniform dist, layer 3", n_failed, tol)
      call assert_real_equals(emis(4), 0.0,  "Uniform dist, layer 4", n_failed, tol)
      call assert_real_equals(sum(emis), base_emis, "Uniform dist, mass conservation", n_failed, tol)

      ! Test 2: Beta distribution
      config%use_beta_dist = .true.
      config%use_wind_adj = .false.
      plmHGT = 250.0
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      call assert_real_equals(emis(1), 17.92, "Beta dist, layer 1", n_failed, 0.001)
      call assert_real_equals(emis(2), 64.00, "Beta dist, layer 2", n_failed, 0.001)
      call assert_real_equals(emis(3), 18.08, "Beta dist, layer 3", n_failed, 0.001)
      call assert_real_equals(sum(emis), base_emis, "Beta dist, mass conservation", n_failed, tol)

      ! Test 3: Wind adjustment (uniform dist)
      config%use_beta_dist = .false.
      config%use_wind_adj = .true.
      plmHGT = 800.0
      U = [10., 10., 10., 10., 10.] ! High wind
      N2 = 2.5e-4 ! Reference N2
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      ! Expect Hp_eff = 800 * (5.0/10.0)**(0.5 * (1+1)) = 400.0
      call assert_real_equals(emis(1), 25.0, "Wind adj, layer 1", n_failed, tol)
      call assert_real_equals(emis(2), 25.0, "Wind adj, layer 2", n_failed, tol)
      call assert_real_equals(emis(3), 25.0, "Wind adj, layer 3", n_failed, tol)
      call assert_real_equals(emis(4), 25.0, "Wind adj, layer 4", n_failed, tol)
      call assert_real_equals(emis(5), 0.0,  "Wind adj, layer 5", n_failed, tol)
      call assert_real_equals(sum(emis), base_emis, "Wind adj, mass conservation", n_failed, tol)

      ! Test 4: Zero plume height
      config%use_beta_dist = .false.
      config%use_wind_adj = .false.
      plmHGT = 0.0
      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      call assert_real_equals(sum(emis), 0.0, "Zero plume height", n_failed, tol)

  end subroutine test_distribute_emissions

end program test_plumerise
