program sofiev_driver
   use plumerise_sofiev_mod
   implicit none

   ! Profile data structure from input file
   TYPE :: profile_type
      integer :: lay
      real(rk)    :: Z
      real(rk)    :: p
      real(rk)    :: t
      real(rk)    :: q
      real(rk)    :: pbl
      real(rk)    :: psfc
      real(rk)    :: frp
   end TYPE profile_type

   type(profile_type) :: profile(35)
   type(PlumeControl) :: config

   ! Variables for plume calculation
   real(rk)    :: pblh, psfc, frp, base_emis
   real(rk)    :: N2, plmHGT
   real(rk)    :: PT1, PT2, Z1, Z2, avg_PT
   integer :: idx1, idx2
   real(rk), dimension(35) :: U_dummy
   real(rk), dimension(35) :: column_emiss

   ! File I/O
   integer :: i, i0

   ! --- Read Input Data ---
   open (9, file='input_profile.txt', status='old')
   read (9, *, iostat=i0)          ! skip headline
   do i = 1, 35
      read (9, *) profile(i)%lay, profile(i)%Z, profile(i)%p, profile(i)%t, &
                  profile(i)%q, profile(i)%pbl, profile(i)%psfc, profile(i)%frp
   end do
   close(9)

   pblh = minval(profile%pbl)
   psfc = minval(profile%psfc)
   frp = minval(profile%frp)
   base_emis = 100.0_rk
   U_dummy = 0.0_rk

   ! --- Calculate Brunt-Vaisala Frequency (N2) ---
   call find_height_index(profile%Z, 1.5_rk * pblh, idx1)
   call find_height_index(profile%Z, 2.0_rk * pblh, idx2)
   idx2 = min(idx2, size(profile%Z))
   idx1 = min(idx1, idx2)
   if (idx1 == idx2) then
       idx1 = max(1, idx1 - 1)
   endif

   Z1 = profile(idx1)%Z
   Z2 = profile(idx2)%Z
   PT1 = profile(idx1)%t * (1000.0_rk / profile(idx1)%p)**KAPPA
   PT2 = profile(idx2)%t * (1000.0_rk / profile(idx2)%p)**KAPPA
   avg_PT = (PT1 + PT2) / 2.0_rk

   if (abs(Z2 - Z1) > 1.0_rk) then
      N2 = (GRAV / avg_PT) * (PT2 - PT1) / (Z2 - Z1)
   else
      N2 = N02
   endif

   ! --- Core Plume Rise Calculation ---
   call plumeRiseSofiev(N2, frp, pblh, plmHGT)
   call distribute_emissions(profile%Z, U_dummy, N2, plmHGT, base_emis, config, column_emiss)

   write (*, '(A, F10.2)') 'Calculated Plume Height (m):', plmHGT
   write (*, '(A, F10.2)') 'SUM of total emiss:', SUM(column_emiss)

end program sofiev_driver
