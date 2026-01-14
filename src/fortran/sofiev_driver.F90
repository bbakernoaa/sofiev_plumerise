program sofiev_driver
   use plumerise_sofiev_mod
   implicit none

   ! Profile data structure from input file
   TYPE :: profile_type
      integer :: lay
      real    :: Z
      real    :: p
      real    :: t
      real    :: q
      real    :: pbl
      real    :: psfc
      real    :: frp
   end TYPE profile_type

   type(profile_type) :: profile(35)
   type(PlumeControl) :: config

   ! Variables for plume calculation
   real    :: pblh, psfc, frp, base_emis
   real    :: N2, plmHGT
   real    :: PT1, PT2, Z1, Z2, avg_PT
   integer :: idx1, idx2
   real, dimension(35) :: U_dummy
   real, dimension(35) :: column_emiss

   ! File I/O
   integer :: i, i0

   ! --- Read Input Data ---
   open (9, file='input_profile.txt', status='old')
   read (9, *, iostat=i0)          ! skip headline
   do i = 1, 35
      read (9, *) profile(i)
   end do
   close(9)

   pblh = minval(profile%pbl)
   psfc = minval(profile%psfc)
   frp = minval(profile%frp)
   base_emis = 100.0
   U_dummy = 0.0 ! Dummy wind profile as it is not in the input file

   ! --- Calculate Brunt-Vaisala Frequency (N2) ---
   ! Based on the stability of the free troposphere (above PBL)
   call find_height_index(profile%Z, 1.5 * pblh, idx1)
   call find_height_index(profile%Z, 2.0 * pblh, idx2)
   idx2 = min(idx2, size(profile%Z))
   idx1 = min(idx1, idx2)
   if (idx1 == idx2) then
       idx1 = max(1, idx1 - 1)
   endif

   ! Calculate potential temperature (PT) at the layer interfaces
   Z1 = profile(idx1)%Z
   Z2 = profile(idx2)%Z
   PT1 = profile(idx1)%t * (1000.0 / profile(idx1)%p)**KAPPA
   PT2 = profile(idx2)%t * (1000.0 / profile(idx2)%p)**KAPPA
   avg_PT = (PT1 + PT2) / 2.0

   ! Calculate N2, with a fallback for vertical layers
   if (abs(Z2 - Z1) > 1.0) then
      N2 = (GRAV / avg_PT) * (PT2 - PT1) / (Z2 - Z1)
   else
      N2 = N02 ! Use reference value if layers are too close
   endif

   ! --- Core Plume Rise Calculation ---

   ! 1. Calculate theoretical plume height using the new API
   call plumeRiseSofiev(N2, frp, pblh, plmHGT)

   ! 2. Distribute emissions vertically
   ! Note: config uses default values, so wind adjustment is off.
   call distribute_emissions(profile%Z, U_dummy, N2, plmHGT, base_emis, config, column_emiss)

   write (*, *) 'Calculated Plume Height (m):', plmHGT
   write (*, *) 'SUM of total emiss:', SUM(column_emiss)

end program sofiev_driver
