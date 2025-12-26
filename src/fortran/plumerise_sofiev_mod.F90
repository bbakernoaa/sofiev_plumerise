module plumerise_sofiev_mod

   implicit none

contains

   subroutine find_height_index(GEOHGT, hgt, index)
      ! simple function to find the height index given the target height in
      ! a profile given the geopotential height.

      ! Arguments
      real, intent(in) :: GEOHGT(:)
      real, intent(in) :: hgt
      integer, intent(out) :: index

      ! Local Variables:
      integer :: i
      real, dimension(:), allocatable :: diffs
      ! real :: diffs(:)

      allocate (diffs(size(geohgt)))

      do i = 1, size(GEOHGT)
         diffs(i) = abs(geohgt(i) - hgt)
      end do

      index = minloc(diffs, 1)
      ! function
      ! index = FINDLOC(geohgt, hgt)
      !index = FINDLOC(geohgt, hgt)

      return

   end subroutine find_height_index

   subroutine distribute_conc_linear(ZF, plmHGT, base_emiss, emis)

      ! Arguments
      real, intent(in) :: ZF(:)      ! Height of full layer (m)

      real, intent(in) :: plmHGT  ! plume rise height
      real, intent(in) :: base_emiss ! emission from file
      real, intent(out) :: emis(:) ! output column emission

      ! Local Variables:
      real :: hgt_prev     ! place holder for previous height index
      real :: dz           ! layer thickness
      real :: total_height ! height of of plume
      real :: column_frac  ! fraction of layer in total plume height
      integer :: plmHGT_index ! plume rise height index in Z
      integer :: z            ! loop index

      ! initialize emission array to zero
      emis = 0.

      ! find the plume height index
      call find_height_index(ZF, plmHGT, plmHGT_index)

      ! get the total height
      total_height = ZF(plmHGT_index)

      hgt_prev = 0.
      do z = 1, plmHGT_index
         dz = ZF(z) - hgt_prev
         column_frac = dz/total_height
         emis(z) = column_frac*base_emiss
         hgt_prev = zf(z)
      end do

      return

   end subroutine distribute_conc_linear

   subroutine sofiev_plmrise_column(Z, T, P, PBLH, psfc, frp, base_emis, plmHGT, column_emiss)
      real, intent(in) :: z(:)
      real, intent(in) :: T(:)
      real, intent(in) :: P(:)
      real, intent(in) :: PBLH
      real, intent(in) :: psfc
      real, intent(in) :: frp
      real, intent(in) :: base_emis
      real, intent(out) :: plmHGT
      real, intent(out) :: column_emiss(:)

      integer :: index
      integer :: pblx2_index
      real :: PT1, PT2, LayerDepth
      real :: hgt_prev

      !find the index of 2x the pbl
      call find_height_index(z, pblh*2, pblx2_index)

      ! now get the plume height
      PT1 = T(pblx2_index - 1)*(psfc/p(pblx2_index - 1))**(2./7.)
      PT2 = T(pblx2_index)*(psfc/p(pblx2_index))**(2./7.)
      LayerDepth = z(pblx2_index) - z(pblx2_index - 1)

      call plumeRiseSofiev(PT1, PT2, LayerDepth, frp, PBLH, plmHGT)

      call distribute_conc_linear(Z, plmHGT, base_emis, column_emiss)

      write (*, *) 'plmHGT:', plmHGT
      write (*, *) 'frp:', frp
      write (*, *) 'pblh:', PBLH
      write (*, *) 'base_emiss:', base_emis
      write (*, *) 'Z    ', 'dz    ', 'column_emiss'
      hgt_prev = 0
      do index = 1, 35
         write (*, *) z(index), z(index) - hgt_prev, column_emiss(index)
         hgt_prev = z(index)
      end do

      return
   end subroutine sofiev_plmrise_column

   subroutine plumeRiseSofiev(PT1, PT2, laydepth, frp, pblh, Hp)

!  This subroutine implements the Sofiev plume rise algorithm
!  History: 09/16/2019: Prototype by Daniel Tong (DT)
!           10/15/2019: bug fix based on feedback from M. Sofiev, DT
!            11/2020: parameterization options, Yunyao Li (YL)
!
!  Ref: M. Sofiev et al., Evaluation of the smoke-injection
!    height from wild-land fires using remote sensing data.
!    Atmos. Chem. Phys., 12, 1995-2006, 2012.

      real Hp                ! plume height (m)
      real pblh                ! PBL height (m)
      real frp                ! fire radiative power (W)
      real NFT_sq       ! N square in Free Troposphere (@ z = 2pblh)
      real PT1, PT2     ! Potential Temperature right below and above PBL height
      real laydepth        ! depth of the layer at the PBL height
      real grav                ! gravity

      real Pf0                ! reference fire power (W)
      real N0_sq        ! Brunt-Vaisala frequency (s-2)
      real alpha        ! part of ABL passed freely
      real beta                ! weights contribution of fire intensity
      real gama                ! power-law dependence on FRP
      real delta        ! dependence on stability in the FT

! ... Initial values.
! ... predefined values parameter set 3 to estimate whether hp higher
! than abl
      alpha = 0.15
      beta = 102
      gama = 0.49
      delta = 0

      Pf0 = 1000000.0
      N0_sq = 0.00025
      grav = 9.8

! ! ... calculate PT from T and P
!       PT1 = T1 * (1000/P1)**0.286
!       PT2 = T2 * (1000/P2)**0.286

! ... calculate Brunt-Vaisala frequency
      NFT_sq = grav/PT1*abs(PT1 - PT2)/laydepth

! ... calculate first guess plume rise top height
      Hp = alpha*pblh + beta*(frp/Pf0)**gama*exp(-delta*NFT_sq/N0_sq)
! ... compare Hp with ABL
      if (Hp .lt. pblh) then
         alpha = 0.24
         beta = 170
         gama = 0.35
         delta = 0.6
         Hp = alpha*pblh + beta*(frp/Pf0)**gama*exp(-delta*NFT_sq/N0_sq)
      else
         alpha = 0.93
         beta = 298
         gama = 0.13
         delta = 0.7
         Hp = alpha*pblh + beta*(frp/Pf0)**gama*exp(-delta*NFT_sq/N0_sq)
      end if

      print *, "The height of fire plume (m) is: ", Hp

   end subroutine

end module plumerise_sofiev_mod
