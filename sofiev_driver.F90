      program sofiev_driver

!  the driver to run the Sofiev plume rise algorithm
!
!  Ref: M. Sofiev et al., Evaluation of the smoke-injection
!    height from wild-land fires using remote sensing data.
!    Atmos. Chem. Phys., 12, 1995-2006, 2012.
!
!  History:
!    Prototype: Daniel Tong, 11/06/2019
!
!-------------------------------------------------------------
      use plumerise_sofiev_mod

      implicit none

      real Hp		! plume height (m)
      real pblh		! PBL height (m)
      real frp		! fire radiative power (W)
      real T1, T2	! Temperature right below and above PBL height (mbar)
      real P1, P2       ! Pressure right below and above PBL height (mbar)
      real PT1, PT2     ! Potential Temperature right below and above PBL height (K)
      real laydepth	! depth of the layer at the PBL height (m)
      real psfc

      integer num	! index of fire data points
      integer i0       ! file reading pointer
      integer :: i
      real :: base_emis
      real :: column_emiss(35)
      real :: plmHGT

      !LAY,ZF,PRES,TA,QV,PBL,PSRFC,FRP
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

! ! ... this block gives example inputs to Sofiev
! !     set or read met data either below or read from a file
! !     If values are set, these values will be overwritten, not used.
!       T1 	= 280
!       T2 	= 179
!       P1 	= 850
!       P2 	= 830
!       laydepth	= 500
!       pblh 	= 1500
!       frp	= 300



! ... read met and FRP data
      open(9,  file='input_profile.txt',  status='old')
      i0 = 0
      read(9,*,iostat=i0)  	! skip headline
      do i=1, 35
        read(9, *) profile(i)
      end do
      pblh = minval(profile%pbl)

      psfc = minval(profile%psfc)
      frp = minval(profile%frp)
      base_emis = 100.

      !open(10, file='outfile.txt', status='new')

      call sofiev_plmrise_column(profile%Z,profile%T,profile%P,PBLH,psfc,frp,base_emis, plmHGT, column_emiss)

      write(*,*) 'SUM of total emiss:' , SUM(column_emiss)

    end program sofiev_driver
