%% Lid-Driven Cavity D3Q27
clear all
clc

set(groot,'defaultAxesFontSize',10)
set(groot,'defaultAxesLabelFontSizeMultiplier',1.2)
set(groot,'defaultLineLineWidth',1.6)
set(groot,'defaultLineMarkerSize',7)
set(groot,'defaultFigureColor','w')
set(groot,'defaultTextInterpreter','latex')
set(groot,'defaultAxesTickLabelInterpreter','latex')
set(groot,'defaultLegendInterpreter','latex')

%% Load data matrices
data = readmatrix('ldc_D3Q27_parallel.dat');
str_xy = readmatrix('streamfunction_XY.dat');

% Grid size (change according to the simulation resolution)
nx = 126;
ny = 126;
nz = 126;

X = reshape(data(:,1), nx, ny, nz); 
Y = reshape(data(:,2), nx, ny, nz); 
Z = reshape(data(:,3), nx, ny, nz); 
U = reshape(data(:,4), nx, ny, nz);
V = reshape(data(:,5), nx, ny, nz);
W = reshape(data(:,6), nx, ny, nz); 
P = reshape(data(:,7), nx, ny, nz);
PSI_XY = reshape(str_xy(:,3), nx, ny);

%% 3D Vector Plot
step = 10; % to reduce density of the 3D vector plot
figure(1)
quiver3( ...
    X(1:step:end,1:step:end,1:step:end), ...
    Z(1:step:end,1:step:end,1:step:end), ...
    Y(1:step:end,1:step:end,1:step:end), ...
    U(1:step:end,1:step:end,1:step:end), ...
    W(1:step:end,1:step:end,1:step:end), ...
    V(1:step:end,1:step:end,1:step:end), ...
    2)
xlabel('x'); ylabel('z'); zlabel('y') % y is thwe wall normal direction
title('Velocity vectors (downsampled)')
axis equal

%% Velocity Contour Plots & Streamlines
% mid plane 
ix = round(nx/2);
iy = round(ny/2);
iz = round(nz/2);

% XZ Plane (y = mid)
figure(2)
X_xz = squeeze(X(:,iy,:));
Z_xz = squeeze(Z(:,iy,:));
U_xz = squeeze(U(:,iy,:));
V_xz = squeeze(V(:,iy,:));
W_xz = squeeze(W(:,iy,:));

subplot(1,3,1)
contourf(X_xz, Z_xz, U_xz, 30, 'LineColor','none')
colorbar
xlabel('x'); ylabel('z')
title('u-velocity on XZ plane (y = mid)')
axis equal tight

subplot(1,3,2)
contourf( X_xz, Z_xz, V_xz, 30, 'LineColor','none')
colorbar
title('v-velocity on XZ plane (y = mid)')
axis equal tight

subplot(1,3,3)
contourf(X_xz, Z_xz, W_xz, 30, 'LineColor','none')
colorbar
title('w-velocity on XZ plane (y = mid)')
axis equal tight

% XY Plane (z = mid)
figure(3)
X_xy = squeeze(X(:,:,iz));
Y_xy = squeeze(Y(:,:,iz));
U_xy = squeeze(U(:,:,iz));
V_xy = squeeze(V(:,:,iz));
W_xy = squeeze(W(:,:,iz));

subplot(1,3,1)
contourf(X_xy, Y_xy, U_xy, 30, 'LineColor','none')
colorbar
xlabel('x'); ylabel('y')
title('u-velocity on XY plane (z = mid)')
axis equal tight

subplot(1,3,2)
contourf(X_xy, Y_xy, V_xy, 30, 'LineColor','none')
colorbar
title('v-velocity on XY plane (z = mid)')
axis equal tight

subplot(1,3,3)
contourf(X_xy, Y_xy, W_xy, 30, 'LineColor','none')
colorbar
title('w-velocity on XY plane (z = mid)')
axis equal tight

% Streamlines XY plane
figure(4)
contour(X_xy, Y_xy, PSI_XY, 40, 'k')
hold on
xlabel('x')
ylabel('y')
xlim([0 1])
ylim([0 1])
axis equal tight

% YZ Plane (x = mid)
figure(5)
Y_yz = squeeze(Y(ix,:,:));
Z_yz = squeeze(Z(ix,:,:));
U_yz = squeeze(U(ix,:,:));
V_yz = squeeze(V(ix,:,:));
W_yz = squeeze(W(ix,:,:));

subplot(1,3,1)
contourf(Z_yz, Y_yz, U_yz, 30, 'LineColor','none')
colorbar
xlabel('z'); ylabel('y')
title('u-velocity on YZ plane (x = mid)')
axis equal tight

subplot(1,3,2)
contourf(Z_yz, Y_yz, V_yz, 30, 'LineColor','none')
colorbar
title('v-velocity on YZ plane (x = mid)')
axis equal tight

subplot(1,3,3)
contourf(Z_yz, Y_yz, W_yz, 30, 'LineColor','none')
colorbar
title('w-velocity on YZ plane (x = mid)')
axis equal tight

%% XY-plane velocity plots for validation against Ghia et. el
%  Plot U velocity profile at x=0.5
figure(6)
h_u = plot(U_xy(ix,:), Y_xy(ix,:), 'Color', 'blue', 'LineWidth', 1, ...
           'DisplayName', 'u-velocity at x/H=0.5');
ylabel('y/H')
xlabel('$u/U_{\mathrm{lid}}$')
title('$Re = 100$')
ylim([0 1])
xlim([min(U_xy(ix,:))*1.2 max(U_xy(ix,:))*1.2])
hold on

% Plot Ghia et.el reference values for U velocity
data_u_ghia = readmatrix("velocity_ghia.xlsx", 'sheet', 'u_velocity');
y_ghia = data_u_ghia(2:end,1);
u_ghia = data_u_ghia(2:end,2);

h_u_ghia = plot(u_ghia, y_ghia, '--+', 'MarkerSize', 6, ...
    'Color', 'k', 'DisplayName', 'Ghia et al. (1982)');
legend([h_u, h_u_ghia], 'Location', 'best')

% Plot V velocity profile
figure(7)
h_v = plot(X_xy(:,iy), V_xy(:,iy), 'Color', 'red', 'LineWidth', 1, ...
           'DisplayName', 'v-velocity at y/H=0.5');
xlabel('x/H')
ylabel('$v/U_{\mathrm{lid}}$')
title('$Re = 100$')
xlim([0 1])
ylim([min(V_xy(:,iy))*1.2 max(V_xy(:,iy))*1.2])
hold on

% Plot Ghia et.el reference values for V velocity
data_v_ghia = readmatrix("velocity_ghia.xlsx", 'sheet', 'v_velocity');
x_ghia = data_v_ghia(2:end,1);
v_ghia = data_v_ghia(2:end,2);

h_v_ghia = plot(x_ghia, v_ghia, '--x', 'MarkerSize', 6, ...
    'Color', 'k', 'DisplayName', 'Ghia et al. (1982)');
legend([h_v, h_v_ghia], 'Location', 'best')

%% Pressure Plot
% create a valid grid
x_vec = squeeze(X(:,1,1));
y_vec = squeeze(Y(1,:,1));
z_vec = squeeze(Z(1,1,:));
[Xg, Zg, Yg] = meshgrid(x_vec, z_vec, y_vec);

figure(8)
P_permuted = permute(P, [3,1,2]);
p_iso = patch(isosurface(Xg, Zg, Yg, P_permuted, mean(P_permuted(:))));
isonormals(Xg, Zg, Yg, P_permuted, p_iso)

set(p_iso,'FaceColor','red','EdgeColor','none','FaceAlpha',0.6)
xlabel('x'); ylabel('z'); zlabel('y')
axis equal tight
title('Pressure Isosurfaces')
view(3)
camlight
lighting gouraud

figure(9)
subplot(1,3,1)
contourf( ...
    squeeze(X(:,iy,:)), ...
    squeeze(Z(:,iy,:)), ...
    squeeze(P(:,iy,:)), 30, 'LineColor','none')
colorbar
xlabel('x'); ylabel('z')
title('Pressure contour on XZ plane (y = mid)')
axis equal tight

subplot(1,3,2)
contourf( ...
    squeeze(X(:,:,iz)), ...
    squeeze(Y(:,:,iz)), ...
    squeeze(P(:,:,iz)), 30, 'LineColor','none')
colorbar
xlabel('x'); ylabel('y')
title('Pressure contour on XY plane (z = mid)')
axis equal tight

subplot(1,3,3)
contourf( ...
    squeeze(Z(ix,:,:)), ...
    squeeze(Y(ix,:,:)), ...
    squeeze(P(ix,:,:)), 30, 'LineColor','none')
colorbar
xlabel('z'); ylabel('y')
title('Pressure contour on YZ plane (x = mid)')
axis equal tight

