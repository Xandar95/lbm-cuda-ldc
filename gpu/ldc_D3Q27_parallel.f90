program lid_driven_cavity_D3Q27_parallel
    ! Lid driven cavity simulation using D3Q27 Lattice Boltzmann Method with parallel computing
    use iso_c_binding
    implicit none

    ! Parameters
    integer(c_int), parameter :: nx = 26, ny = 26, nz = 26 ! grid dimensions
    integer :: nstep = 10000 ! number of time steps
    integer :: i, j, k, l, t
    real(c_double) :: xl, yl, zl, dx, dy, dz, nu, Re, c2, omega, u_lid, cu, u2
    integer :: i_mid = nx / 2, j_mid = ny / 2, k_mid = nz / 2 ! mid-plane indices for streamfunction calculation
    integer :: count_start, count_end, count_rate ! for timing

    ! Variable arrays
    real(c_double), dimension(nx, ny, nz) :: rho, u, v, w, p, x, y, z
    real(c_double), target, dimension(nx, ny, nz, 0:26) :: f ! distribution function
    real(c_double), dimension(nx, ny) :: u_xz, u_xy, w_yz, psi_xz, psi_xy, psi_yz ! for streamfunction calculation

    ! Constant arrays
    real(c_double), target, dimension(0:26) :: weights
    integer(c_int), target, dimension(0:26) :: cx, cy, cz, opp

    ! --- Interfaces to CUDA kernel ---
    interface
        subroutine lbm_init_gpu(nx, ny, nz, cx, cy, cz, opp, weights) bind(C, name="lbm_init_gpu")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            integer(c_int), intent(in) :: cx(*), cy(*), cz(*), opp(*)
            real(c_double), intent(in) :: weights(*)
        end subroutine lbm_init_gpu

        subroutine lbm_copy_host_to_device(f, nx, ny, nz) bind(C, name="lbm_copy_host_to_device")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_double), intent(in) :: f(*) ! read only
        end subroutine lbm_copy_host_to_device

        subroutine lbm_run_step_gpu(nx, ny, nz, omega, u_lid) bind(C, name="lbm_run_step_gpu")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_double), value :: omega, u_lid
        end subroutine lbm_run_step_gpu

        subroutine lbm_copy_device_to_host(f, nx, ny, nz) bind(C, name="lbm_copy_device_to_host")
            use iso_c_binding
            integer(c_int), value :: nx, ny, nz
            real(c_double), intent(out) :: f(*) ! write only
        end subroutine lbm_copy_device_to_host

        subroutine lbm_free_gpu() bind(C, name="lbm_free_gpu")
            use iso_c_binding
        end subroutine lbm_free_gpu
    end interface
    
    call system_clock(count_start, count_rate) ! Start simulation timing
    call system_clock(count_start)

    ! ---- Initialization ----
    xl = 1.0d0
    yl = 1.0d0
    zl = 1.0d0
    dx = xl / (nx - 1)
    dy = yl / (ny - 1)
    dz = zl / (nz - 1)
    c2 = 1.0d0 / 3.0d0 ! lattice speed of sound squared for D3Q27 (i.e., cs = c/sqrt(3))
    u_lid = 0.05d0 ! Lid velocity in the lattice
    Re = 100.0d0 ! Desired Reynolds number
    nu = u_lid * (ny - 1.0d0) / Re ! Recalculate viscosity based on Re and lid velocity
    omega = 1.0d0 / (3.0d0 * nu + 0.5d0) ! Relaxation parameter (SRT model)
    ! Ma = u_lid / cs should be < 0.1 for incompressibility

    ! Lattice weights for D3Q27
    weights = [8.0d0/27.0d0, &                                                                             ! w0
               2.0d0/27.0d0, 2.0d0/27.0d0, 2.0d0/27.0d0, 2.0d0/27.0d0, 2.0d0/27.0, 2.0d0/27.0d0, &         ! w1-w6
               1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, &       ! w7-w18
               1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, 1.0d0/54.0d0, &
               1.0d0/216.0d0, 1.0d0/216.0d0, 1.0d0/216.0d0, 1.0d0/216.0d0, 1.0d0/216.0d0, 1.0d0/216.0d0, & ! w19-w26
               1.0d0/216.0d0, 1.0d0/216.0d0]

    ! Lattice velocities for D3Q27
    cx = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, -1, 1, 1, -1, -1, 1]
    cy = [0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -1, 1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1] ! wall normal in y
    cz = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1, 0, 0, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1] 

    ! Opposite directions for bounce-back
    opp = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25]

    ! Set lattice positions
    x(1, :, :) = 0.0d0
    y(:, 1, :) = 0.0d0
    z(:, :, 1) = 0.0d0
    do i = 1, nx-1
        x(i+1, :, :) = x(i, :, :) + dx
    end do
    do j = 1, ny-1
        y(:, j+1, :) = y(:, j, :) + dy
    end do
    do k = 1, nz-1
        z(:, :, k+1) = z(:, :, k) + dz
    end do

    ! Initial condition: rho = 1, u = 0, v = 0, w = 0
    rho = 1.0d0 ! Initial density
    u = 0.0d0
    v = 0.0d0
    w = 0.0d0
    u(:, ny, :) = u_lid ! Set lid velocity at the top boundary

    ! Initialize distributions (at equilibrium) (SOA layout)
    do l = 0, 26
        do k = 1, nz
            do j = 1, ny
                do i = 1, nx
                    cu = cx(l) * u(i, j, k) + cy(l) * v(i, j, k) + cz(l) * w(i, j, k)
                    u2 = u(i, j, k)**2 + v(i, j, k)**2 + w(i, j, k)**2
                     ! 2nd order accurate equilibrium distribution function
                    f(i, j, k, l) = weights(l) * rho(i, j, k) * (1.0d0 + 3.0d0 * cu + 4.5d0 * cu**2 - 1.5d0 * u2)
                end do
            end do
        end do
    end do

    ! Initialize GPU
    print *, 'Initializing GPU...'
    call lbm_init_gpu(nx, ny, nz, cx, cy, cz, opp, weights)

    ! Copy initial distribution to GPU
    print *, 'Copying initial data to GPU...'
    call lbm_copy_host_to_device(f, nx, ny, nz)

    ! ---- GPU Time loop ----
    print *, 'Starting time loop on GPU...'

    do t = 1, nstep
        call lbm_run_step_gpu(nx, ny, nz, omega, u_lid)
        ! progress output
        if (mod(t, 1000) == 0) then
            print *, 'Time step: ', t
        end if
    end do

    ! Copy results back to host
    print *, 'Copying results back to host...'
    call lbm_copy_device_to_host(f, nx, ny, nz)

    ! Free GPU resources
    call lbm_free_gpu()

    call system_clock(count_end) ! End simulation timing
    print *, 'Total simulation time (s): ', real(count_end - count_start) / real(count_rate)

    ! Output velocity and pressure results
    open(unit=10, file='ldc_D3Q27_parallel.dat', status='replace')
    do k = 1, nz
        do j = 1, ny
            do i = 1, nx
                ! Compute macroscopic variables
                rho(i, j, k) = 0.0d0
                u(i, j, k) = 0.0d0
                v(i, j, k) = 0.0d0
                w(i, j, k) = 0.0d0
                do l = 0, 26
                    rho(i, j, k) = rho(i, j, k) + f(i, j, k, l)
                    u(i, j, k) = u(i, j, k) + f(i, j, k, l) * cx(l)
                    v(i, j, k) = v(i, j, k) + f(i, j, k, l) * cy(l)
                    w(i, j, k) = w(i, j, k) + f(i, j, k, l) * cz(l)
                end do
                u(i, j, k) = u(i, j, k) / rho(i, j, k)
                v(i, j, k) = v(i, j, k) / rho(i, j, k)
                w(i, j, k) = w(i, j, k) / rho(i, j, k)
                p(i, j, k) = c2 * rho(i, j, k) ! equation of state for pressure in LBM

                write(10, '(7(ES16.8, 1X))') x(i, j, k), y(i, j, k), z(i, j, k), &
                                             u(i, j, k) / u_lid, v(i, j, k) / u_lid, w(i, j, k) / u_lid, & ! macroscopic velocities normalized by lid velocity
                                             p(i, j, k)
            end do
        end do
    end do
    close(10)

    ! Streamfunction calculation 
    ! Extract velocity components on mid planes
    ! XZ plane at y = ny/2
    do k = 1, nz
        do i = 1, nx
            u_xz(i, k) = u(i, j_mid, k)
        end do
    end do

    ! XY plane at z = nz/2
    do j = 1, ny
        do i = 1, nx
            u_xy(i, j) = u(i, j, k_mid)
        end do
    end do

    ! YZ plane at x = nx/2
    do k = 1, nz
        do j = 1, ny
            w_yz(j, k) = w(i_mid, j, k)
        end do
    end do

    ! Simple finite difference to compute streamfunction (2D)
    ! XZ plane
    psi_xz = 0.0d0
    do k = 2, nz
        do i = 1, nx
            psi_xz(i, k) = psi_xz(i, k-1) + u_xz(i, k-1) * dz
        end do
    end do

    ! XY plane
    psi_xy = 0.0d0
    do j = 2, ny
        do i = 1, nx
            psi_xy(i, j) = psi_xy(i, j-1) + u_xy(i, j-1) * dy
        end do
    end do

    ! YZ plane
    psi_yz = 0.0d0
    do k = 1, nz
        do j = 2, ny
            psi_yz(j, k) = psi_yz(j-1, k) + w_yz(j-1, k) * dy
        end do
    end do

    ! Output streamfunction data
    open(unit=20, file='streamfunction_XZ.dat', status='replace')
    do k = 1, nz
        do i = 1, nx
            write(20, '(7(ES16.8, 1X))') x(i, j_mid, k), z(i, j_mid, k), psi_xz(i, k)
        end do
    end do
    close(20)

    open(unit=30, file='streamfunction_XY.dat', status='replace')
    do j = 1, ny
        do i = 1, nx
            write(30, '(7(ES16.8, 1X))') x(i, j, k_mid), y(i, j, k_mid), psi_xy(i, j)
        end do
    end do
    close(30)

    open(unit=40, file='streamfunction_YZ.dat', status='replace')
    do k = 1, nz
        do j = 1, ny
            write(40, '(7(ES16.8, 1X))') y(i_mid, j, k), z(i_mid, j, k), psi_yz(j, k)
        end do
    end do
    close(40)

    print *, 'Simulation completed. Results written to ldc_D3Q27_parallel.dat & streamfunction_XY.dat & streamfunction_XZ.dat & streamfunction_YZ.dat'
    
end program lid_driven_cavity_D3Q27_parallel