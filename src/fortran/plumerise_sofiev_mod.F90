!> @file plumerise_sofiev_mod.f90
!> @brief Advanced Plume Rise and Vertical Distribution Module
!> @author Gemini / Research-Based Implementation
!> @date 2026

module plumerise_sofiev_mod
   use iso_c_binding, only: c_double, c_int, c_bool
   implicit none

   integer, parameter :: rk = c_double

   !> Physical Constants
   real(rk), parameter :: GRAV   = 9.80665_rk
   real(rk), parameter :: KAPPA  = 0.2857_rk
   real(rk), parameter :: N02    = 2.5e-4_rk
   real(rk), parameter :: PF0    = 1.0e6_rk

   !> @brief Configuration structure to toggle advanced physics
   type :: PlumeControl
      logical :: use_beta_dist = .false.
      logical :: use_wind_adj  = .false.
      real(rk)    :: alpha         = 3.0_rk
      real(rk)    :: bet           = 2.0_rk
   end type PlumeControl

contains

   !> @brief Finds the index in a profile that first exceeds the target height.
   subroutine find_height_index(ZF, hgt, idx)
      real(rk), intent(in)     :: ZF(:)
      real(rk), intent(in)     :: hgt
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
   subroutine distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)
      real(rk), intent(in)               :: ZF(:), U(:), N2
      real(rk), intent(in)               :: plmHGT, base_emis
      type(PlumeControl), intent(in) :: config
      real(rk), intent(out)              :: emis(:)

      integer :: z, plm_idx
      real(rk)    :: hgt_prev, layer_top, x_low, x_high, Hp_eff, avg_U
      real(rk)    :: stab_penalty, weight
      real(rk), parameter :: N2_ref = 2.5e-4_rk

      emis = 0.0_rk
      if (plmHGT <= 0.0_rk) return

      ! 1. Wind & Stability Entrainment (Optional)
      Hp_eff = plmHGT
      if (config%use_wind_adj) then
         call find_height_index(ZF, plmHGT, plm_idx)
         avg_U = sum(U(1:plm_idx)) / max(1.0_rk, real(plm_idx, rk))

         stab_penalty = 1.0_rk + max(0.0_rk, N2 / N2_ref)

         if (avg_U > 2.0_rk) then
            Hp_eff = plmHGT * (5.0_rk / max(5.0_rk, avg_U))**(0.5_rk * stab_penalty)
         end if
      end if

      call find_height_index(ZF, Hp_eff, plm_idx)

      hgt_prev = 0.0_rk
      do z = 1, plm_idx
         layer_top = min(ZF(z), Hp_eff)

         if (config%use_beta_dist) then
            x_low  = hgt_prev / Hp_eff
            x_high = layer_top / Hp_eff
            weight = (4.0_rk*x_high**3 - 3.0_rk*x_high**4) - (4.0_rk*x_low**3 - 3.0_rk*x_low**4)
         else
            weight = (layer_top - hgt_prev) / Hp_eff
         end if

         emis(z) = weight * base_emis
         hgt_prev = ZF(z)
         if (hgt_prev >= Hp_eff) exit
      end do

      if (abs(sum(emis) - base_emis) > 1.e-6_rk) then
         emis(plm_idx) = emis(plm_idx) + (base_emis - sum(emis))
      end if
   end subroutine distribute_emissions

   !> @brief Core Sofiev Plume Rise algorithm
   subroutine plumeRiseSofiev(N2, frp, pblh, Hp)
      real(rk), intent(in)  :: N2, frp, pblh
      real(rk), intent(out) :: Hp
      real(rk) :: a, b, g, d

      a = 0.15_rk; b = 102.0_rk; g = 0.49_rk; d = 0.0_rk
      Hp = a * pblh + b * (frp/PF0)**g * exp(-d * N2 / N02)

      if (Hp < pblh) then
         a = 0.24_rk; b = 170.0_rk; g = 0.35_rk; d = 0.6_rk
      else
         a = 0.93_rk; b = 298.0_rk; g = 0.13_rk; d = 0.7_rk
      end if

      Hp = a * pblh + b * (frp/PF0)**g * exp(-d * N2 / N02)
      Hp = max(Hp, 10.0_rk)
   end subroutine plumeRiseSofiev


   ! ------------------------------------------------------------------
   ! C Interoperability Wrappers
   ! ------------------------------------------------------------------
   subroutine plume_rise_sofiev_c(N2, frp, pblh, Hp) &
      bind(c, name='plume_rise_sofiev_c')
      real(c_double), value, intent(in)  :: N2, frp, pblh
      real(c_double), intent(out) :: Hp

      call plumeRiseSofiev(N2, frp, pblh, Hp)

   end subroutine plume_rise_sofiev_c

   subroutine distribute_emissions_c(n_layers, ZF, U, N2, plmHGT, base_emis, &
                                    use_beta_dist_c, use_wind_adj_c, emis) &
      bind(c, name='distribute_emissions_c')

      integer(c_int), value, intent(in) :: n_layers
      logical(c_bool), value, intent(in) :: use_beta_dist_c, use_wind_adj_c
      real(c_double), intent(in)  :: ZF(n_layers), U(n_layers)
      real(c_double), value, intent(in)  :: N2, plmHGT, base_emis
      real(c_double), intent(out) :: emis(n_layers)

      type(PlumeControl) :: config

      config%use_beta_dist = use_beta_dist_c
      config%use_wind_adj  = use_wind_adj_c

      call distribute_emissions(ZF, U, N2, plmHGT, base_emis, config, emis)

   end subroutine distribute_emissions_c

end module plumerise_sofiev_mod
