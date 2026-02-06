program lid_driven_cavity_D3Q27
    ! Lid-driven cavity flow simulation using Lattice Boltzmann Method (D3Q27 model)
    
    implicit none
    integer, parameter :: nx = 26, ny = 26, nz = 26 ! Grid dimensions
    integer :: i, j, k, l, t, nstep, i_post, j_post, k_post
    real(8) :: dx, dy, dz, xl, yl, zl, nu, c2, omega, u_lid, Re, cu, u2, start_time, end_time
    real(8), dimension(nx, ny, nz) :: rho, u, v, w, x, y, z, p
    real(kind=8), dimension(0:26, nx, ny, nz) :: f, f_new ! distribution functions
    real(kind=8), dimension(0:26) :: weights, feq ! lattice weights, equilibrium distribution
    integer, dimension(0:26) :: cx, cy, cz, opp ! lattice velocities and opposite directions

    call cpu_time(start_time) ! Start simulation timing

    ! ---- Initialization ----
    xl = 1.0d0
    yl = 1.0d0
    zl = 1.0d0
    dx = xl / (nx - 1)
    dy = yl / (ny - 1)
    dz = zl / (nz - 1)
    nu = 0.005d0  ! Kinematic viscosity
    Re = 100.0d0 ! Reynolds number from macroscopic parameters
    c2 = 1.0d0 / 3.0d0 ! lattice speed of sound squared for D3Q27 (i.e., cs = c/sqrt(3))
    omega = 1.0d0 / (3.0d0 * nu + 0.5d0) ! Relaxation parameter (SRT model)
    u_lid = Re * nu / (ny - 1) ! Lid velocity in the lattice based on Reynolds number = 0.02
    ! Ma = u_lid / cs should be < 0.1 for incompressible flow
    nstep = 20000

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

    ! Initialize distributions (at equilibrium)
    do i = 1, nx
        do j = 1, ny
            do k = 1, nz
                do l = 0, 26
                    cu = cx(l) * u(i, j, k) + cy(l) * v(i, j, k) + cz(l) * w(i, j, k)
                    u2 = u(i, j, k)**2 + v(i, j, k)**2 + w(i, j, k)**2
                     ! 2nd order accurate equilibrium distribution function
                    feq(l) = weights(l) * rho(i, j, k) * (1.0d0 + 3.0d0 * cu + 4.5d0 * cu**2 - 1.5d0 * u2)
                    f(l, i, j, k) = feq(l)
                end do
            end do
        end do
    end do

    ! ---- Time loop ----
    do t = 1, nstep

        ! AOS layout for f and f_new
        ! --- Collision step ---
        do i = 1, nx
            do j = 1, ny
                do k = 1, nz
                    do l = 0, 26
                        cu = cx(l) * u(i, j, k) + cy(l) * v(i, j, k) + cz(l) * w(i, j, k)
                        u2 = u(i, j, k)**2 + v(i, j, k)**2 + w(i, j, k)**2
                        ! 2nd order accurate equilibrium distribution function
                        feq(l) = weights(l) * rho(i, j, k) * (1.0d0 + 3.0d0 * cu + 4.5d0 * cu**2 - 1.5d0 * u2)
                        f(l, i, j, k) = (1.0d0 - omega) * f(l, i, j, k) + omega * feq(l) ! SRT BGK collision
                    end do
                end do
            end do
        end do

        ! --- Streaming step ---
        ! Push style streaming
        f_new = f ! Initialize f_new

        do i = 1, nx
            do j = 1, ny
                do k = 1, nz
                    do l = 1, 26 ! skip rest particle (l=0)
                         ! Calculate new indices after streaming 
                        i_post = i + cx(l)
                        j_post = j + cy(l)
                        k_post = k + cz(l)
                        ! Only stream if within bounds (without periodicity)
                        if (i_post >= 1 .and. i_post <= nx .and. &
                            j_post >= 1 .and. j_post <= ny .and. &
                            k_post >= 1 .and. k_post <= nz) then
                            f_new(l, i_post, j_post, k_post) = f(l, i, j, k)
                        end if
                    end do
                    ! Rest particle remains the same
                    f_new(0, i, j, k) = f(0, i, j, k)
                end do
            end do
        end do

        ! Update distributions for next time step
        f = f_new

        ! --- Boundary conditions ---
        ! No-slip walls (bounce-back)
        do j = 1, ny
            do k = 1, nz
                ! West wall (i=1)
                do l = 0, 26
                    if (cx(l) == 1) then
                        f(l, 1, j, k) = f(opp(l), 1, j, k)
                    end if
                end do
                ! East wall (i=nx)
                do l = 0, 26
                    if (cx(l) == -1) then
                        f(l, nx, j, k) = f(opp(l), nx, j, k)
                    end if
                end do
            end do
        end do

        do i = 1, nx
            do j = 1, ny
                ! South wall (k=1)
                do l = 0, 26
                    if (cz(l) == 1) then
                        f(l, i, j, 1) = f(opp(l), i, j, 1)
                    end if
                end do
                ! North wall (k=nz)
                do l = 0, 26
                    if (cz(l) == -1) then
                        f(l, i, j, nz) = f(opp(l), i, j, nz)
                    end if
                end do
            end do
        end do

        do i = 1, nx
            do k = 1, nz
                ! Bottom wall (j=1)
                do l = 0, 26
                    if (cy(l) == 1) then 
                        f(l, i, 1, k) = f(opp(l), i, 1, k)
                    end if
                end do
            end do
        end do
        
        do i = 2, nx-1
            do k = 2, nz-1 ! Exclude corners to avoid double counting
                ! Top wall (moving lid) (j=ny)
                ! Calculate density at the lid
                rho(i, ny, k) = (1.0d0/(1.0d0 + (2.0d0 * u_lid / 9.0d0))) * &
                                (f(0, i, ny, k) + f(1, i, ny, k) + f(2, i, ny, k) + f(3, i, ny, k) + f(4, i, ny, k) + &
                                 f(7, i, ny, k) + f(8, i, ny, k) + f(9, i, ny, k) + f(10, i, ny, k) + &
                                 2.0d0 * (f(6, i, ny, k) + f(12, i, ny, k) + f(15, i, ny, k) + f(13, i, ny, k) + &
                                          f(19, i, ny, k) + f(22, i, ny, k) + f(17, i, ny, k) + f(24, i, ny, k) + f(26, i, ny, k)))
                do l = 0, 26
                    if (cy(l) == -1) then 
                        f(l, i, ny, k) = f(opp(l), i, ny, k) + 6.0d0 * weights(l) * rho(i, ny, k) * cx(l) * u_lid
                    end if
                end do
            end do
        end do

        ! --- Macroscopic variable update ---
        do i = 1, nx
            do j = 1, ny
                do k = 1, nz
                    rho(i, j, k) = 0.0d0 ! reset density
                    u(i, j, k) = 0.0d0
                    v(i, j, k) = 0.0d0
                    w(i, j, k) = 0.0d0
                    do l = 0, 26
                        rho(i, j, k) = rho(i, j, k) + f(l, i, j, k) ! accumulate density
                        u(i, j, k) = u(i, j, k) + f(l, i, j, k) * cx(l) ! accumulate x-momentum
                        v(i, j, k) = v(i, j, k) + f(l, i, j, k) * cy(l) ! accumulate y-momentum
                        w(i, j, k) = w(i, j, k) + f(l, i, j, k) * cz(l) ! accumulate z-momentum
                    end do
                    u(i, j, k) = u(i, j, k) / rho(i, j, k) ! normalize velocity
                    v(i, j, k) = v(i, j, k) / rho(i, j, k) ! normalize velocity
                    w(i, j, k) = w(i, j, k) / rho(i, j, k) ! normalize velocity
                    p(i, j, k) = c2 * rho(i, j, k) ! pressure from density
                end do
            end do
        end do

        ! --- Progress output ---
        if (mod(t, 1000) == 0) then
            print *, 'Time step: ', t
        end if

    end do ! End of time loop

    call cpu_time(end_time) ! End simulation timing
    print *, 'Total simulation time (s): ', end_time - start_time

    ! Output results
    open(unit=10, file='ldc_D3Q27_results.dat', status='replace')
    do i = 1, nx
        do j = 1, ny
            do k = 1, nz
                write(10,*) x(i, j, k), y(i, j, k), z(i, j, k), u(i, j, k), v(i, j, k), w(i, j, k), p(i, j, k)
            end do
        end do
    end do
    close(10)

    print *, 'Simulation completed. Results written to ldc_D3Q27_results.dat'

end program lid_driven_cavity_D3Q27