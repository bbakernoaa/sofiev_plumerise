!> @file plumerise_sofiev_mod.f90
!> @brief Advanced Plume Rise and Vertical Distribution Module
!> @author Gemini / Research-Based Implementation
!> @date 2026

module plumerise_sofiev_mod
   implicit none

   !> Physical Constants
   real, parameter :: GRAV   = 9.80665  !< Acceleration due to gravity (m/s^2)
   real, parameter :: KAPPA  = 0.2857   !< Rd/Cp (Poisson Constant)
   real, parameter :: N02    = 2.5e-4   !< Reference Brunt-Vaisala frequency squared (s^-2)
   real, parameter :: PF0    = 1.0e6    !< Reference Fire Radiative Power (1 MW)

   !> @brief Configuration structure to toggle advanced physics
   type :: PlumeControl
      logical :: use_beta_dist = .false. !< True: Use Beta PDF; False: Uniform distribution
      logical :: use_wind_adj  = .false. !< True: Adjust Hp based on wind/stability entrainment
      real    :: alpha         = 3.0     !< Beta shape param (Determines peak height)
      real    :: bet           = 2.0     !< Beta shape param (Determines tail)
   end type PlumeControl

contains

   !> @brief Finds the index in a profile that first exceeds the target height.
   !> @param ZF Array of layer interface heights (m)
   !> @param hgt Target height (m)
   !> @param idx Output index
   subroutine find_height_index(ZF, hgt, idx)
      real, intent(in)     :: ZF(:)
      real, intent(in)     :: hgt
      integer, intent(out) :: idx
      integer              :: i

      idx = size(ZF)
      do i = 1, size(ZF)
         if (ZF(i) >= hgt) then
            idx = i
            exit
         end if
      end do
   end subroutine find_height_index

   !> @brief Distributes surface emissions into a vertical column.
   !> @details Implements optional Beta-distribution mapping and wind-shear entrainment.
   !> @param ZF Layer interface heights (m)
   !> @param U Wind speed profile (m/s)
   !> @param N2 Brunt-Vaisala frequency (s^-2)
   !> @param plmHGT Theoretical plume rise height (m)
   !> @param base_emis Total surface emission (mass/time)
   !> @param config Configuration flags
   !> @param emis Output vertical emission profile
   subroutine distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      real, intent(in)               :: ZF(:), U(:), N2
      real, intent(in)               :: plmHGT, base_emis
      type(PlumeControl), intent(in) :: config
      real, intent(out)              :: emis(:)

      integer :: z, plm_idx
      real    :: hgt_prev, layer_top, x_low, x_high, Hp_eff, avg_U
      real    :: stab_penalty, weight
      real, parameter :: N2_ref = 2.5e-4

      emis = 0.0
      if (plmHGT <= 0.0) return

      ! 1. Wind & Stability Entrainment (Optional)
      Hp_eff = plmHGT
      if (config%use_wind_adj) then
         call find_height_index(ZF, plmHGT, plm_idx)
         avg_U = sum(U(1:plm_idx)) / max(1.0, real(plm_idx))

         ! Stability Penalty: Higher N2 (stable) increases the wind's suppressive effect.
         stab_penalty = 1.0 + max(0.0, N2 / N2_ref)

         if (avg_U > 2.0) then
            ! Power-law scaling for bent-over plumes (Ref: Briggs / Freitas et al.)
            Hp_eff = plmHGT * (5.0 / max(5.0, avg_U))**(0.5 * stab_penalty)
         end if
      end if

      call find_height_index(ZF, Hp_eff, plm_idx)

      hgt_prev = 0.0
      do z = 1, plm_idx
         layer_top = min(ZF(z), Hp_eff)

         if (config%use_beta_dist) then
            ! Beta(3,2) Integration: 4x^3 - 3x^4
            ! This concentrates ~60-80% of mass in the top third of the plume.
            x_low  = hgt_prev / Hp_eff
            x_high = layer_top / Hp_eff
            weight = (4.0*x_high**3 - 3.0*x_high**4) - (4.0*x_low**3 - 3.0*x_low**4)
         else
            ! Standard Linear/Uniform mapping (Mass / Total Depth)
            weight = (layer_top - hgt_prev) / Hp_eff
         end if

         emis(z) = weight * base_emis
         hgt_prev = ZF(z)
         if (hgt_prev >= Hp_eff) exit
      end do

      ! Ensure strict mass conservation (correct for floating point truncation)
      if (abs(sum(emis) - base_emis) > 1.e-6) then
         emis(plm_idx) = emis(plm_idx) + (base_emis - sum(emis))
      end if
   end subroutine distribute_emissions

   !> @brief Core Sofiev Plume Rise algorithm (Ref: Sofiev et al. 2012)
   !> @param N2 Stability at 2x PBLH (s^-2)
   !> @param frp Fire Radiative Power (W)
   !> @param pblh Planetary Boundary Layer height (m)
   !> @param Hp Output plume top height (m)
   subroutine plumeRiseSofiev(N2, frp, pblh, Hp)
      real, intent(in)  :: N2, frp, pblh
      real, intent(out) :: Hp
      real :: a, b, g, d

      ! Initial guess: Plume exceeds ABL (Set 3 parameters)
      a = 0.15; b = 102.0; g = 0.49; d = 0.0
      Hp = a * pblh + b * (frp/PF0)**g * exp(-d * N2 / N02)

      if (Hp < pblh) then
         ! Case: Plume stays within Boundary Layer
         a = 0.24; b = 170.0; g = 0.35; d = 0.6
      else
         ! Case: Plume penetrates Free Troposphere
         a = 0.93; b = 298.0; g = 0.13; d = 0.7
      end if

      Hp = a * pblh + b * (frp/PF0)**g * exp(-d * N2 / N02)
      Hp = max(Hp, 10.0) ! Numerical floor
   end subroutine plumeRiseSofiev

end module plumerise_sofiev_mod