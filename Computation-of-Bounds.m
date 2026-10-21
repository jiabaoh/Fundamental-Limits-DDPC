%% computation_of_bounds.m  --  lower bounds on the risk of finite-horizon data-driven predictive control
%%                              scalar state-space model, optimal-bias bound solved with the PDE Toolbox
%
% SETTING (Section "Illustrative example")
%   system   x_{k+1} = a x_k + b u_k + w_k,   w_k ~ N(0,sw2) i.i.d.,   x_0 ~ N(0,s02),   theta = (a,b) unknown
%   task     min E sum_{k=0}^{T-1} (q x_k^2 + r u_k^2) over state feedback, shrinking horizon, no terminal cost (P_T = 0)
%   oracle   u_k = K_k^*(theta) x_k, K_k^* from the Riccati recursion (Lemma lemma:dp);  h(theta) = [K_0^*, ..., K_{T-1}^*]
%   data     N open-loop transitions of the same system, u_k ~ N(0,su2) white, x_0 ~ N(0,s02)
%   loss     (h_hat - h)' W (h_hat - h),  W = diag(S_k Sigma_k^*),  S_k = r + b^2 P_{k+1},  Sigma_k^* = E[x_k^2] under the oracle
%   domain   D = { theta : (a-a0)^2 + (b-b0)^2 <= rho0^2 }
%
% BOUNDS (Theorem thm:bounds).  With M = I_F^{-1} = diag(Ma,Mb) and the Fisher information of the record
% (Appendix of He2026fundamental)
%       I_F = (1/sw2) diag( sum_{k=0}^{N-1} sig_k^2 , N su2 ),   sig_{k+1}^2 = a^2 sig_k^2 + b^2 su2 + sw2,   sig_0^2 = s02,
%   (1) UCRLB (pointwise in theta):      R_theta >= sum_k W_k (grad K_k^*)' M (grad K_k^*)
%   (2) optimal-bias bound (over D):     int_D R_theta dtheta >= min_b F(b),
%           F(b) = int_D f(theta,b,b') dtheta,   f = sum_k W_k [ b_k^2 + (grad b_k + grad K_k^*)' M (grad b_k + grad K_k^*) ],
%       where b = (b_0, ..., b_{T-1}) is the bias of the estimator and the minimum is over all smooth b on D,
%       with NO boundary condition.
% The gradients grad K_k^* = (dK_k^*/da, dK_k^*/db) are obtained analytically by differentiating the Riccati recursion
% (oracle_vec below).  Problem (2) is a problem of the calculus of variations; how it is solved is explained in
% Section (2) of the script.
%
% OUTPUT  printed bounds, figures, and lb_results.mat (mesh nodes A,B, nodal weights wq, bound integrands) read by compare_ddpc.m
% Requires the Partial Differential Equation Toolbox.
clear; close all;

%% ---------------- setting ----------------
q = 1; r = 0.1; T = 4;                      % cost weights, control horizon
Ndata = 20;                                 % identification sample size N
sw2 = 1; s02 = 1; su2 = 1;                  % variances of the noise w_k, of x_0 (task AND identification record), of the excitation u_k
a0 = 0.5; b0 = 0.5; rho0 = 0.4;             % domain D: centre (a0,b0), radius rho0
Hmax = 0.015;                               % maximal edge length of the finite element mesh
h_step = 1e-5;                              % finite-difference step for the divergence of the flux (see Section (2))
ngrid = 41;                                 % Cartesian grid for display only

%% ---------------- mesh of D and nodal quadrature ----------------
% D is triangulated once (the mesh is deterministic).  Every integral over D below is computed by the lumped nodal rule
%       int_D phi(theta) dtheta  ~  wq' * phi(nodes),    wq_j = (1/3) * (total area of the triangles containing node j),
% so that sum(wq) = area(D) = pi rho0^2.
model = make_model(a0,b0,rho0,Hmax);
nodes = model.Mesh.Nodes; elems = model.Mesh.Elements; nn = size(nodes,2);
[~,AE] = area(model.Mesh);                                        % element areas
wq = accumarray(elems(:), repmat(AE(:)/3,3,1), [nn 1]);           % lumped nodal weights
px = nodes(1,:)'; py = nodes(2,:)';                               % node coordinates: px = a, py = b
fprintf('mesh: %d nodes, %d elements, area %.5f (pi rho0^2 = %.5f)\n', nn, size(elems,2), sum(wq), pi*rho0^2);

%% ---------------- oracle gains, their sensitivities and the Fisher information at the nodes ----------------
[Kn,Sn,Sgn,dKan,dKbn] = oracle_vec(px,py,q,r,s02,sw2,T);         % K_k^*, S_k, Sigma_k^*, dK_k^*/da, dK_k^*/db   (nn x T)
[Man,Mbn] = fisher_inv_vec(px,py,Ndata,sw2,su2,s02);              % Ma, Mb: diagonal of M = I_F^{-1}             (nn x 1)
Wn = Sn.*Sgn;                                                     % loss weights W_k = S_k Sigma_k^*              (nn x T)
% the analytic sensitivities are checked against central finite differences at one node
k0 = 1; fd = jac_fd(@(th) first_out(@oracle_vec,th(1),th(2),q,r,s02,sw2,T), [px(k0);py(k0)], 1e-6);
fprintf('sensitivity check: %.2e\n', max(max(abs(fd - [dKan(k0,:)' dKbn(k0,:)']))));

%% ---------------- (1) UCRLB ----------------
% per step  ucr_t(:,k+1) = W_k [ Ma (dK_k^*/da)^2 + Mb (dK_k^*/db)^2 ]  (M is diagonal);  ucr = sum over k;  UCR_int = int_D ucr
ucr_t = Wn.*(Man.*dKan.^2 + Mbn.*dKbn.^2); ucr = sum(ucr_t,2); UCR_int = wq'*ucr;

%% ---------------- (2) optimal-bias bound: calculus of variations, one Neumann problem per time step ----------------
% DECOUPLING.  f is a sum over k, W and M are diagonal, and b_k enters only the k-th term; hence
%       min_b F(b) = sum_k min_{b_k} F_k(b_k),
%       F_k(b_k) = int_D W_k [ b_k^2 + Ma (b_k,a + K_k,a)^2 + Mb (b_k,b + K_k,b)^2 ] dtheta,
%   where ( ),a = d/da, ( ),b = d/db and K_k = K_k^*.  The index k is dropped below.
%
% STATIONARITY (first variation).  F is a convex quadratic functional of b, so b^* minimises F if and only if the
%   directional derivative d/de F(b^* + e v) at e = 0 vanishes for every smooth v on D (v is free on the boundary too):
%       int_D W [ b v + Ma (b,a + K,a) v,a + Mb (b,b + K,b) v,b ] dtheta = 0     for all v.          (weak form)
%
% EULER-LAGRANGE EQUATION AND NATURAL BOUNDARY CONDITION.  Integrating the weak form by parts (Green's theorem)
%   moves the derivatives off v:
%       int_D  [ -(W Ma (b,a + K,a)),a - (W Mb (b,b + K,b)),b + W b ] v dtheta
%     + int_dD [ n_a W Ma (b,a + K,a) + n_b W Mb (b,b + K,b) ] v ds = 0           for all v,
%   with n = (n_a,n_b) the outward unit normal of D.  Taking v supported inside D, then v arbitrary on dD, gives
%       -(W Ma b,a),a - (W Mb b,b),b + W b = (W Ma K,a),a + (W Mb K,b),b        in D,     (Euler-Lagrange equation)
%        n_a W Ma (b,a + K,a) + n_b W Mb (b,b + K,b) = 0                        on dD.    (natural boundary condition)
%   The boundary condition is not imposed on the admissible functions: it follows from the minimisation because b is
%   free on dD.  It is a Neumann-type condition (it fixes the normal flux of b, not its values).
%
% PDE TOOLBOX FORM.  The toolbox solves  -div(c grad u) + a u = f  in D,  n.(c grad u) + q u = g  on dD, so we set
%       u = b,   c = diag(W Ma, W Mb),   a = W,   f = (W Ma K,a),a + (W Mb K,b),b,   q = 0,   g = -(n_a W Ma K,a + n_b W Mb K,b).
%   W, Ma, Mb, K,a, K,b are smooth functions of theta evaluated on demand by the coefficient functions below (Wc, Mac,
%   Mbc, flux); the divergence in f is taken by central finite differences (step h_step) of the flux (W Ma K,a, W Mb K,b).
%   (Integrating f v by parts shows that this f and g reproduce exactly the weak form above.)
%
% FINITE ELEMENT METHOD (what solvepde does).  b is sought among the continuous functions that are linear on every
%   triangle of the mesh,  b_h = sum_j beta_j phi_j,  where phi_j is the "hat" function of node j (= 1 at node j, = 0 at
%   all other nodes) and beta_j = b_h(node j).  Inserting b_h and v = phi_i into the weak form gives the linear system
%       sum_j beta_j int_D W [ phi_i phi_j + Ma phi_i,a phi_j,a + Mb phi_i,b phi_j,b ] dtheta = int_D f phi_i dtheta + int_dD g phi_i ds,
%   i.e.  A beta = rhs  with A sparse, symmetric and positive definite (W > 0, M > 0).  Equivalently, b_h minimises F
%   over the piecewise-linear functions (Ritz method on that space), and the natural boundary condition enters only
%   through the boundary integral on the right-hand side.  The error decreases with the mesh size: F(b_h) - F(b^*) = O(Hmax^2).
%
% EVALUATION.  F_k(b_k^*) = int_D f_k(theta,b_k^*,grad b_k^*) dtheta is computed from the nodal values and gradients of
%   the finite element solution with the lumped nodal quadrature wq.
prm = struct('q',q,'r',r,'s02',s02,'sw2',sw2,'T',T,'N',Ndata,'su2',su2);
Fstar_t = zeros(T,1); f_t = zeros(nn,T); fb_t = zeros(nn,T); bstar = zeros(nn,T);
for t = 1:T-1                                                     % index t <-> time step k = t-1;  k = T-1: K_{T-1}^* = 0, so b_{T-1}^* = 0 and F_{T-1} = 0
    % flux components  W Ma K,a  and  W Mb K,b  (the source term of the Euler-Lagrange equation)
    Fa = @(x,y) flux(x,y,t,1,prm);
    Fb = @(x,y) flux(x,y,t,2,prm);
    % PDE Toolbox coefficients, evaluated at the quadrature points loc of each element (loc.x = a, loc.y = b)
    cFunc = @(loc,st) [Wc(loc.x,loc.y,t,prm).*Mac(loc.x,loc.y,prm); ...                 % c = diag(W Ma, W Mb)
                       Wc(loc.x,loc.y,t,prm).*Mbc(loc.x,loc.y,prm)];
    aFunc = @(loc,st) Wc(loc.x,loc.y,t,prm);                                             % a = W
    fFunc = @(loc,st) (Fa(loc.x+h_step,loc.y) - Fa(loc.x-h_step,loc.y))/(2*h_step) + ... % f = div(W M grad K)
                      (Fb(loc.x,loc.y+h_step) - Fb(loc.x,loc.y-h_step))/(2*h_step);
    gFunc = @(loc,st) -(Fa(loc.x,loc.y).*loc.nx + Fb(loc.x,loc.y).*loc.ny);              % g = -n.(W M grad K): natural boundary condition
    mdl = make_model(a0,b0,rho0,Hmax);                            % fresh model on the same deterministic mesh
    if size(mdl.Mesh.Nodes,2) ~= nn || max(abs(mdl.Mesh.Nodes(:) - nodes(:))) > 1e-12, error('mesh changed between calls'); end
    specifyCoefficients(mdl,'m',0,'d',0,'c',cFunc,'a',aFunc,'f',fFunc);                  % -div(c grad b) + a b = f   in D
    applyBoundaryCondition(mdl,'neumann','Edge',1:mdl.Geometry.NumEdges,'g',gFunc,'q',0); % n.(c grad b) = g           on dD
    results = solvepde(mdl);                                      % finite element solution
    b = results.NodalSolution; [ba,bb] = evaluateGradient(results);                      % b_k^* and grad b_k^* at the nodes
    b = b(:); ba = ba(:); bb = bb(:);
    f_t(:,t)  = Wn(:,t).*(b.^2 + Man.*(ba + dKan(:,t)).^2 + Mbn.*(bb + dKbn(:,t)).^2);   % integrand f_k at the minimiser
    fb_t(:,t) = Wn(:,t).*b.^2;                                                           % its bias part W_k (b_k^*)^2
    Fstar_t(t) = wq'*f_t(:,t); bstar(:,t) = b;                                           % F_k(b_k^*) by nodal quadrature
    fprintf('  t=%d: F_t(b_t^*) = %.4f   int UCRLB_t = %.4f\n', t-1, Fstar_t(t), wq'*ucr_t(:,t));
end
Fstar = sum(Fstar_t); f = sum(f_t,2); f_bias = sum(fb_t,2); f_var = f - f_bias;         % total bound, its integrand and the bias/variance split
fprintf('T=%d, N=%d:  optimal-bias bound F(b*) = %.4f,  int UCRLB = %.4f\n', T, Ndata, Fstar, UCR_int);

%% ---------------- figures ----------------
color_F_OPT = [0.80 0.29 0.24]; color_UCRLB = [0.16 0.38 0.60];
[Ac,Bc] = meshgrid(linspace(a0-rho0,a0+rho0,ngrid), linspace(b0-rho0,b0+rho0,ngrid));
inD = (Ac-a0).^2 + (Bc-b0).^2 <= rho0^2;
Fs = scatteredInterpolant(px,py,f,'linear','none');   Fgrid = Fs(Ac,Bc); Fgrid(~inD) = NaN;
Us = scatteredInterpolant(px,py,ucr,'linear','none'); Ugrid = Us(Ac,Bc); Ugrid(~inD) = NaN;
figure('Name','lower bounds (PDE toolbox)');
surf(Ac,Bc,Fgrid,'FaceColor',color_F_OPT,'EdgeColor','k','EdgeAlpha',0.3); hold on;
surf(Ac,Bc,Ugrid,'FaceColor',color_UCRLB,'FaceAlpha',0.5,'EdgeColor','k','EdgeAlpha',0.15);
xlabel('a'); ylabel('b'); zlabel('risk'); legend('f(\theta,b^*,b^{*\prime})','UCRLB');
title(sprintf('lower bounds, T=%d, N=%d: F(b^*)=%.3f, \\int UCRLB=%.3f',T,Ndata,Fstar,UCR_int));
figure('Name','per-step decomposition');
subplot(1,2,1); hb = bar([Fstar_t, (wq'*ucr_t)']); set(hb(1),'FaceColor',color_F_OPT); set(hb(2),'FaceColor',color_UCRLB);
legend('F_t(b_t^*)','\int UCRLB_t'); xlabel('t+1'); title('per-step bound');
subplot(1,2,2); Bs = scatteredInterpolant(px,py,bstar(:,1),'linear','none'); Bg = Bs(Ac,Bc); Bg(~inD) = NaN;
surf(Ac,Bc,Bg); xlabel('a'); ylabel('b'); title('optimal bias b_0^*(\theta) for K_0^*');

%% ---------------- save for compare_ddpc.m ----------------
% A, B = mesh nodes, wq = nodal quadrature weights, f/ucr = integrands of the two bounds at the nodes
A = px; B = py;
save('lb_results.mat','q','r','T','Ndata','sw2','s02','su2','a0','b0','rho0','A','B','wq','ucr','f','f_bias','f_var','Fstar','UCR_int','ucr_t','f_t','Fstar_t');

%% ======================= local functions =======================
function model = make_model(a0,b0,rho0,Hmax)
% geometry (disk of radius rho0 centred at (a0,b0)) and linear triangular mesh
model = createpde();
geometryFromEdges(model, decsg([1; a0; b0; rho0]));
generateMesh(model,'Hmax',Hmax,'GeometricOrder','linear');
end

function [K,S,Sg,dKa,dKb] = oracle_vec(a,b,q,r,s02,sw2,T)
% Oracle of the finite-horizon LQ problem (Lemma lemma:dp, P_T = 0), vectorised over the column vectors a, b (one entry
% per point theta).  Backward Riccati recursion, k = T-1, ..., 0:
%       S_k = r + b^2 P_{k+1},    K_k^* = -a b P_{k+1} / S_k,    P_k = q + a^2 r P_{k+1} / S_k
%   (equivalently P_k = q + r K_k^2 + c_k^2 P_{k+1} with c_k = a + b K_k^*, the closed-loop pole).
% Sensitivities, obtained by differentiating the recursion with respect to theta = (a,b):
%       dS_k = b^2 dP_{k+1} + 2 b P_{k+1} db
%       dK_k = -[ P_{k+1} (b da + a db) + a b dP_{k+1} + K_k dS_k ] / S_k
%       dP_k = c_k^2 dP_{k+1} + 2 c_k P_{k+1} (da + K_k db)
%   the last line by the envelope theorem: P_k = min_K [ q + r K^2 + (a + b K)^2 P_{k+1} ], so only the explicit
%   theta-dependence is differentiated and K_k is held fixed.
% Forward recursion of the oracle state variance:  Sigma_0^* = s02,  Sigma_{k+1}^* = c_k^2 Sigma_k^* + sw2.
% Outputs are n x T, column k+1 <-> time step k:  K = K_k^*, S = S_k, Sg = Sigma_k^*, dKa = dK_k^*/da, dKb = dK_k^*/db.
a = a(:); b = b(:); n = numel(a);
P = zeros(n,1); dPa = zeros(n,1); dPb = zeros(n,1);               % P_T = 0 and its derivatives
K = zeros(n,T); S = zeros(n,T); dKa = zeros(n,T); dKb = zeros(n,T);
for t = T:-1:1                                                    % t = k+1
    Pn = P; S(:,t) = r + b.^2.*Pn; K(:,t) = -a.*b.*Pn./S(:,t); c = a + b.*K(:,t);
    dSa = b.^2.*dPa;  dSb = 2*b.*Pn + b.^2.*dPb;
    dKa(:,t) = -(Pn.*b + a.*b.*dPa + dSa.*K(:,t))./S(:,t);
    dKb(:,t) = -(Pn.*a + a.*b.*dPb + dSb.*K(:,t))./S(:,t);
    dPa_n = c.^2.*dPa + 2*c.*Pn;                                  % envelope theorem
    dPb_n = c.^2.*dPb + 2*c.*Pn.*K(:,t);
    P = q + a.^2*r.*Pn./S(:,t); dPa = dPa_n; dPb = dPb_n;
end
Sg = zeros(n,T); Sg(:,1) = s02;
for t = 1:T-1, Sg(:,t+1) = (a + b.*K(:,t)).^2.*Sg(:,t) + sw2; end
end

function [Ma,Mb] = fisher_inv_vec(a,b,N,sw2,su2,s02)
% Diagonal of M = I_F^{-1} for the record of N transitions x_{k+1} = a x_k + b u_k + w_k, x_0 ~ N(0,s02),
% u_k ~ N(0,su2) white, w_k ~ N(0,sw2)  (Appendix of He2026fundamental):
%       I_F = (1/sw2) diag( sum_{k=0}^{N-1} sig_k^2 , N su2 ),    sig_{k+1}^2 = a^2 sig_k^2 + b^2 su2 + sw2,   sig_0^2 = s02,
% where sig_k^2 = E[x_k^2] is the state variance of the open-loop record (transient start, not the stationary value).
% Vectorised over the column vectors a, b.
a = a(:); b = b(:); sig2 = s02*ones(size(a)); Sx = zeros(size(a));
for k = 0:N-1, Sx = Sx + sig2; sig2 = a.^2.*sig2 + b.^2*su2 + sw2; end
Ma = sw2./Sx; Mb = (sw2/(N*su2))*ones(size(a));
end

% ---- coefficient functions for the PDE Toolbox (row vector in, row vector out, as the toolbox requires) ----
function W = Wc(x,y,t,p)
% loss weight W_k = S_k Sigma_k^* at the points theta = (x,y), time step k = t-1
[~,S,Sg] = oracle_vec(x,y,p.q,p.r,p.s02,p.sw2,p.T); W = (S(:,t).*Sg(:,t))';
end
function Ma = Mac(x,y,p), [Ma,~] = fisher_inv_vec(x,y,p.N,p.sw2,p.su2,p.s02); Ma = Ma'; end   % (1,1) entry of M
function Mb = Mbc(x,y,p), [~,Mb] = fisher_inv_vec(x,y,p.N,p.sw2,p.su2,p.s02); Mb = Mb'; end   % (2,2) entry of M
function F = flux(x,y,t,l,p)
% flux components  W Ma dK_k^*/da (l = 1)  and  W Mb dK_k^*/db (l = 2)  at the points theta = (x,y), time step k = t-1
[~,S,Sg,dKa,dKb] = oracle_vec(x,y,p.q,p.r,p.s02,p.sw2,p.T); [Ma,Mb] = fisher_inv_vec(x,y,p.N,p.sw2,p.su2,p.s02);
W = S(:,t).*Sg(:,t);
if l == 1, F = (W.*Ma.*dKa(:,t))'; else, F = (W.*Mb.*dKb(:,t))'; end
end

% ---- helpers for the finite-difference check of the sensitivities ----
function K = first_out(fun,varargin), K = fun(varargin{:}); K = K(:); end

function J = jac_fd(fun,th,ep)
% central finite-difference Jacobian of fun at th
th = th(:); n = numel(th); f0 = fun(th); J = zeros(numel(f0),n);
for i = 1:n, e = zeros(n,1); e(i) = ep; J(:,i) = (fun(th+e) - fun(th-e))/(2*ep); end
end
