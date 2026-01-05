clear; clc; close all;

%% Physical Parameters
p.M = 0.6;
p.m = 0.2;
p.l = 0.3;
p.I = 0.006;
p.b = 0.05;
p.c = 0.002;
p.g = 9.81;
S = load('metric_net.mat');

%% Simulation Parameters
dt   = 0.002;
Tend = 8.0;
t    = 0:dt:Tend;
N    = numel(t);

uMax = 40;        % actuator limit
Nint = 35;        % CCM path integral resolution

%% ref (upright equilibrium)
% theta = pi  →  (s,c) = (0,-1)
xd = [0; 0; 0; -1; 0];
ud = 0;

%% initial condition
theta0 = pi + deg2rad(15);
X = zeros(5, N);
U = zeros(1, N);
X(:,1) = [0; 0; sin(theta0); cos(theta0); 0];

% MAIN SIM LOOP
for k = 1:N-1
    x = X(:,k);

    % CCM control
    u = ccm_control_embed(x, xd, ud, p, S, Nint);
    u = max(min(u, uMax), -uMax);
    U(k) = u;

    % Integrate dynamics
    X(:,k+1) = rk4(@(xx) cartpend_embed(xx,u,p), x, dt);

    % Re-project sin/cos for visuals
    sc = X(3:4,k+1);
    nrm = norm(sc);
    if nrm > 1e-9
        X(3:4,k+1) = sc / nrm;
    else
        X(3:4,k+1) = [0; -1];
    end
end
U(end) = U(end-1);

%% Post-Processing
theta = atan2(X(3,:), X(4,:));     % (-pi,pi]
theta_err = theta - pi;
theta_err_u = unwrap(theta_err);


figure('Name','CCM stabilization diagnostics','Color','w');
subplot(4,1,1)
plot(t, theta_err_u, 'LineWidth',1.5)
grid on
ylabel('\theta_{err} (rad)')
title('CCM angle error (unwrapped)')

subplot(4,1,2)
plot(t, abs(theta_err_u), 'LineWidth',1.5)
grid on
ylabel('|\theta_{err}|')
title('Envelope (should decay quickly)')

subplot(4,1,3)
plot(t, X(1,:), 'LineWidth',1.3)
grid on
ylabel('x (m)')
title('Cart position')

subplot(4,1,4)
plot(t, U, 'LineWidth',1.3)
grid on
ylabel('u (N)')
xlabel('t (s)')
title('Control input')

%% ===================== FUNCTIONS =============================
function dx = cartpend_embed(x,u,p)
    % x = [x; xdot; s; c; thetadot]
    xdot = x(2);
    s    = x(3);
    c    = x(4);
    dth  = x(5);

    D = (p.M+p.m)*(p.I+p.m*p.l^2) - (p.m*p.l*c)^2;

    ddx = ((p.I+p.m*p.l^2)*(u - p.b*xdot + p.m*p.l*s*dth^2) ...
          + (p.m*p.l*c)*(p.m*p.g*p.l*s - p.c*dth)) / D;

    ddth = (-(p.m*p.l*c)*(u - p.b*xdot + p.m*p.l*s*dth^2) ...
            - (p.M+p.m)*(p.m*p.g*p.l*s - p.c*dth)) / D;

    ds = c*dth;
    dc = -s*dth;

    dx = [xdot; ddx; ds; dc; ddth];
end

function u = ccm_control_embed(x, xd, ud, p, S, Nint)
    dx = x - xd;
    ds = 1/Nint;
    acc = 0;

    for i = 1:Nint
        s = (i-0.5)*ds;
        xk = xd + s*dx;
        [~,B] = f_and_B_embed(xk,p);
        [~,K] = metric_MK_embed(xk,S);
        acc = acc + (K*dx)*ds;
    end

    u = ud + acc;
    % Small cart damping (no slow drift)
    u = u - 0.4*x(1) - 1.0*x(2);
end

function [f,B] = f_and_B_embed(x,p)
    xdot = x(2);
    s    = x(3);
    c    = x(4);
    dth  = x(5);

    D = (p.M+p.m)*(p.I+p.m*p.l^2) - (p.m*p.l*c)^2;

    a1 = (p.I+p.m*p.l^2)/D;
    b1 = -(p.m*p.l*c)/D;

    ddx0 = ((p.I+p.m*p.l^2)*(-p.b*xdot + p.m*p.l*s*dth^2) ...
           + (p.m*p.l*c)*(p.m*p.g*p.l*s - p.c*dth)) / D;

    ddth0 = ((p.m*p.l*c)*(p.b*xdot - p.m*p.l*s*dth^2) ...
            - (p.M+p.m)*(p.m*p.g*p.l*s - p.c*dth)) / D;

    f = [xdot; ddx0; c*dth; -s*dth; ddth0];
    B = [0; a1; 0; 0; b1];
end

function [M,K] = metric_MK_embed(x,S)
    xn = (x - S.x_mean(:)) ./ S.x_std(:);

    % M(x)
    a1 = tanh(S.M_W1*xn + S.M_b1(:));
    a2 = tanh(S.M_W2*a1 + S.M_b2(:));
    v  = S.M_W3*a2 + S.M_b3(:);

    L = zeros(5);
    k = 1;
    for r = 1:5
        for c = 1:r
            L(r,c) = v(k); k = k+1;
        end
    end
    M = L'*L + S.eps(1)*eye(5);

    % ----- Gain net -----
    g1 = tanh(S.K_W1*xn + S.K_b1(:));
    g2 = tanh(S.K_W2*g1 + S.K_b2(:));
    kv = S.K_W3*g2 + S.K_b3(:);
    K = kv(:).';
end

function xnext = rk4(f,x,h)
    k1 = f(x);
    k2 = f(x + 0.5*h*k1);
    k3 = f(x + 0.5*h*k2);
    k4 = f(x + h*k3);
    xnext = x + h*(k1 + 2*k2 + 2*k3 + k4)/6;
end
