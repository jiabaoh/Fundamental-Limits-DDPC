%%  --  Lower bounds for finite-horizon data-driven MPC, scalar ARX(1,1),
%%                          optimal-bias bound solved with the PDE Toolbox (explicit Neumann condition)
%
% Same problem and setting as lb_scalar_mpc.m (Ritz version):
%     y_{k+1} = a y_k + b u_k + w_{k+1},  cost sum_{k=0}^{T-1} (q y_k^2 + r u_k^2),  P_T = 0,
%     h(theta) = [K_0^* ... K_{T-1}^*],  W = diag(S_k Sigma_k^*),
%     I_F = (1/sw2) diag( sum_{k<N} sig_k^2 , N su2 ) with sig_{k+1}^2 = a^2 sig_k^2 + b^2 su2 + sw2, sig_0^2 = s02 (eq. FIM),
%     D = {(a-a0)^2 + (b-b0)^2 <= rho0^2}.
% Because W and I_F are diagonal, the optimal-bias problem decouples: for each k the bias b_k solves
% the Euler-Lagrange equation with its NATURAL boundary condition,
%     -div( W M grad b ) + W b = div( W M grad K_k^* )      in D,
%      n . ( W M grad b )     = -n . ( W M grad K_k^* )     on dD,          M = I_F^{-1} = diag(Ma,Mb),
% which is the PDE-Toolbox form  -div(c grad u) + a u = f,  n.(c grad u) + q u = g  with
%     c = [W Ma; W Mb] (diagonal),  a = W,  f = d/da(W Ma dK/da) + d/db(W Mb dK/db),
%     g = -(W Ma dK/da n_a + W Mb dK/db n_b),  q = 0.
% The gain derivatives dK_k^*/dtheta come from the analytic sensitivity recursion (vectorised over
% mesh points); the divergence in f is taken by central finite differences of the flux, as in the
% infinite-horizon script.  Outputs and the saved file lb_results.mat have the same fields as the
% Ritz version, so compare_ddmpc.m can be run unchanged afterwards.
% Requires: Partial Differential Equation Toolbox.
clear; close all;

%% ---------------- setting ----------------
q = 1; r = 0.1; T = 4;                      % cost weights, control horizon
Ndata = 20;                                 % identification sample size
sw2 = 1; s02 = 1; su2 = 1;                  % noise, initial state (task AND identification record), excitation variances
a0 = 0.5; b0 = 0.5; rho0 = 0.4;             % domain D
Hmax = 0.015;                               % mesh size
h_step = 1e-5;                              % finite-difference step for the divergence of the flux
ngrid = 41;                                 % Cartesian grid for display

%% ---------------- mesh and nodal quadrature ----------------
model = make_model(a0,b0,rho0,Hmax);
nodes = model.Mesh.Nodes; elems = model.Mesh.Elements; nn = size(nodes,2);
[~,AE] = area(model.Mesh);                                        % element areas
wq = accumarray(elems(:), repmat(AE(:)/3,3,1), [nn 1]);           % lumped nodal weights (sum = area of D)
px = nodes(1,:)'; py = nodes(2,:)';
fprintf('mesh: %d nodes, %d elements, area %.5f (pi rho0^2 = %.5f)\n', nn, size(elems,2), sum(wq), pi*rho0^2);

%% ---------------- oracle quantities, sensitivities, Fisher information at the nodes ----------------
[Kn,Sn,Sgn,dKan,dKbn] = oracle_vec(px,py,q,r,s02,sw2,T);         % nn x T
[Man,Mbn] = fisher_inv_vec(px,py,Ndata,sw2,su2,s02);              % nn x 1, transient start x_0 ~ N(0,s02)
Wn = Sn.*Sgn;                                                     % W_k = S_k Sigma_k^*
% sanity check of the analytic sensitivities against finite differences at one node
k0 = 1; fd = jac_fd(@(th) first_out(@oracle_vec,th(1),th(2),q,r,s02,sw2,T), [px(k0);py(k0)], 1e-6);
fprintf('sensitivity check: %.2e\n', max(max(abs(fd - [dKan(k0,:)' dKbn(k0,:)']))));

%% ---------------- (1) UCRLB ----------------
ucr_t = Wn.*(Man.*dKan.^2 + Mbn.*dKbn.^2); ucr = sum(ucr_t,2); UCR_int = wq'*ucr;

%% ---------------- (2) optimal-bias bound: one Neumann problem per time step ----------------
prm = struct('q',q,'r',r,'s02',s02,'sw2',sw2,'T',T,'N',Ndata,'su2',su2);
Fstar_t = zeros(T,1); f_t = zeros(nn,T); fb_t = zeros(nn,T); bstar = zeros(nn,T);
for t = 1:T-1                                                     % t = T: K_{T-1}^* = 0 -> b = 0, F = 0
    Fa = @(x,y) flux(x,y,t,1,prm);                                % W Ma dK/da
    Fb = @(x,y) flux(x,y,t,2,prm);                                % W Mb dK/db
    cFunc = @(loc,st) [Wc(loc.x,loc.y,t,prm).*Mac(loc.x,loc.y,prm); ...
                       Wc(loc.x,loc.y,t,prm).*Mbc(loc.x,loc.y,prm)];
    aFunc = @(loc,st) Wc(loc.x,loc.y,t,prm);
    fFunc = @(loc,st) (Fa(loc.x+h_step,loc.y) - Fa(loc.x-h_step,loc.y))/(2*h_step) + ...
                      (Fb(loc.x,loc.y+h_step) - Fb(loc.x,loc.y-h_step))/(2*h_step);
    gFunc = @(loc,st) -(Fa(loc.x,loc.y).*loc.nx + Fb(loc.x,loc.y).*loc.ny);   % natural BC, imposed explicitly
    mdl = make_model(a0,b0,rho0,Hmax);                            % fresh model (same deterministic mesh)
    if size(mdl.Mesh.Nodes,2) ~= nn || max(abs(mdl.Mesh.Nodes(:) - nodes(:))) > 1e-12, error('mesh changed between calls'); end
    specifyCoefficients(mdl,'m',0,'d',0,'c',cFunc,'a',aFunc,'f',fFunc);
    applyBoundaryCondition(mdl,'neumann','Edge',1:mdl.Geometry.NumEdges,'g',gFunc,'q',0);
    results = solvepde(mdl);
    b = results.NodalSolution; [ba,bb] = evaluateGradient(results);
    b = b(:); ba = ba(:); bb = bb(:);
    f_t(:,t)  = Wn(:,t).*(b.^2 + Man.*(ba + dKan(:,t)).^2 + Mbn.*(bb + dKbn(:,t)).^2);
    fb_t(:,t) = Wn(:,t).*b.^2;
    Fstar_t(t) = wq'*f_t(:,t); bstar(:,t) = b;
    fprintf('  t=%d: F_t(b_t^*) = %.4f   int UCRLB_t = %.4f\n', t-1, Fstar_t(t), wq'*ucr_t(:,t));
end
Fstar = sum(Fstar_t); f = sum(f_t,2); f_bias = sum(fb_t,2); f_var = f - f_bias;
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

% same fields as the Ritz version (A,B = mesh nodes, wq = nodal weights) so that compare_ddmpc.m runs unchanged
A = px; B = py;
save('lb_results.mat','q','r','T','Ndata','sw2','s02','su2','a0','b0','rho0','A','B','wq','ucr','f','f_bias','f_var','Fstar','UCR_int','ucr_t','f_t','Fstar_t');

%% ======================= local functions =======================
function model = make_model(a0,b0,rho0,Hmax)
model = createpde();
geometryFromEdges(model, decsg([1; a0; b0; rho0]));              % disk of radius rho0 centred at (a0,b0)
generateMesh(model,'Hmax',Hmax,'GeometricOrder','linear');
end

function [K,S,Sg,dKa,dKb] = oracle_vec(a,b,q,r,s02,sw2,T)
% vectorised finite-horizon LQ (P_T = 0): gains K_k^*, curvatures S_k, oracle covariances Sigma_k^*
% and analytic sensitivities dK_k^*/da, dK_k^*/db, for column vectors a, b.  Outputs are n x T.
a = a(:); b = b(:); n = numel(a);
P = zeros(n,1); dPa = zeros(n,1); dPb = zeros(n,1);
K = zeros(n,T); S = zeros(n,T); dKa = zeros(n,T); dKb = zeros(n,T);
for t = T:-1:1
    Pn = P; S(:,t) = r + b.^2.*Pn; K(:,t) = -a.*b.*Pn./S(:,t); c = a + b.*K(:,t);
    dSa = b.^2.*dPa;  dSb = 2*b.*Pn + b.^2.*dPb;
    dKa(:,t) = -(Pn.*b + a.*b.*dPa + dSa.*K(:,t))./S(:,t);
    dKb(:,t) = -(Pn.*a + a.*b.*dPb + dSb.*K(:,t))./S(:,t);
    dPa_n = c.^2.*dPa + 2*c.*Pn;                                  % envelope theorem: only explicit theta-dependence
    dPb_n = c.^2.*dPb + 2*c.*Pn.*K(:,t);
    P = q + a.^2*r.*Pn./S(:,t); dPa = dPa_n; dPb = dPb_n;
end
Sg = zeros(n,T); Sg(:,1) = s02;
for t = 1:T-1, Sg(:,t+1) = (a + b.*K(:,t)).^2.*Sg(:,t) + sw2; end
end

function [Ma,Mb] = fisher_inv_vec(a,b,N,sw2,su2,s02)
% diagonal of I_F^{-1} for the N transitions x_{k+1} = a x_k + b u_k + e_k with x_0 ~ N(0,s02), u_k ~ N(0,su2)
% white, e_k ~ N(0,sw2):  I_F = (1/sw2) diag( sum_{k=0}^{N-1} sig_k^2 , N su2 ),  sig_{k+1}^2 = a^2 sig_k^2 + b^2 su2 + sw2
% (eq. FIM), vectorised over column vectors a, b; the recursion equals the geometric-sum closed form.
a = a(:); b = b(:); sig2 = s02*ones(size(a)); Sx = zeros(size(a));
for k = 0:N-1, Sx = Sx + sig2; sig2 = a.^2.*sig2 + b.^2*su2 + sw2; end
Ma = sw2./Sx; Mb = (sw2/(N*su2))*ones(size(a));
end

% ---- coefficient helpers (row-vector in, row-vector out, as required by the PDE Toolbox) ----
function W = Wc(x,y,t,p)
[~,S,Sg] = oracle_vec(x,y,p.q,p.r,p.s02,p.sw2,p.T); W = (S(:,t).*Sg(:,t))';
end
function Ma = Mac(x,y,p), [Ma,~] = fisher_inv_vec(x,y,p.N,p.sw2,p.su2,p.s02); Ma = Ma'; end
function Mb = Mbc(x,y,p), [~,Mb] = fisher_inv_vec(x,y,p.N,p.sw2,p.su2,p.s02); Mb = Mb'; end
function F = flux(x,y,t,l,p)
% flux components  W Ma dK/da (l = 1)  and  W Mb dK/db (l = 2)  at time step t
[~,S,Sg,dKa,dKb] = oracle_vec(x,y,p.q,p.r,p.s02,p.sw2,p.T); [Ma,Mb] = fisher_inv_vec(x,y,p.N,p.sw2,p.su2,p.s02);
W = S(:,t).*Sg(:,t);
if l == 1, F = (W.*Ma.*dKa(:,t))'; else, F = (W.*Mb.*dKb(:,t))'; end
end

function K = first_out(fun,varargin), K = fun(varargin{:}); K = K(:); end

function J = jac_fd(fun,th,ep)
th = th(:); n = numel(th); f0 = fun(th); J = zeros(numel(f0),n);
for i = 1:n, e = zeros(n,1); e(i) = ep; J(:,i) = (fun(th+e) - fun(th-e))/(2*ep); end
end
