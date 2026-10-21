%% compare_ddpc.m  --  CE / SPC / DeePC / Bayes versus the two lower bounds, scalar state-space model, horizon T
%     Requires lb_results.mat produced by computation_of_bounds.m (same setting).
%
% SETTING (as in computation_of_bounds.m)
%   system   x_{k+1} = a x_k + b u_k + w_k,   w_k ~ N(0,sw2),   theta = (a,b) in D
%   data     one record of N open-loop transitions, u_k ~ N(0,su2) white:  x = [x_0 ... x_N]', u = [u_0 ... u_{N-1}]'
%   task     shrinking-horizon predictive control of the cost sum_{k=0}^{T-1} (q x_k^2 + r u_k^2): at time t the
%            controller minimises the cost over the remaining horizon f_t = T-1-t and applies the first input
%
% CONTROLLERS (Section "Design of data-driven predictive control").  Each one returns, for every record, an explicit
% gain sequence K_hat_0, ..., K_hat_{T-1} (u_t = K_hat_t x_t), which is compared with the oracle h = [K_0^*, ..., K_{T-1}^*]:
%   CE     theta_hat = least-squares estimate (eq. ls), Riccati recursion at theta_hat  ->  K_hat_k = K_k^*(theta_hat)
%   SPC    multi-step predictor [Gamma_f H_f] = X_f (Phi_f)' (Phi_f (Phi_f)')^{-1} (eq. spc-ls) from the data matrices
%          X0 (initial states), U_f (future inputs), X_f (future states) with a FIXED number of columns N_T = N-T+1 for
%          every horizon f: the depth-T matrices are built once per record and horizon f uses their leading rows
%          (leading sub-blocks of X_T Phi_T' and Phi_T Phi_T'); gain from Lemma "gain of a direct method" (eq. K-spc)
%   DeePC  same data matrices, with the projection-based regulariser lam*||(I - Pi_f) g||^2, Pi_f the orthogonal
%          projector onto the row space of [X0; U_f] (eq. deepc); g has length N_T at every step; gain from the
%          explicit KKT solution (eq. K-deepc)
%   Bayes  average-risk rule with uniform prior on D,
%            K_hat_t = int_D w_t(theta) K_t^*(theta) p(Z|theta) dtheta / int_D w_t(theta) p(Z|theta) dtheta,
%          w_t = S_t Sigma_t^* the loss weight, p(Z|theta) the Gaussian likelihood of the record given x_0;
%          it minimises the average risk int_D R_theta dtheta exactly (it uses D and sw2)
%
% REPORTED per controller:  risk R_theta = E[(h_hat - h)' W (h_hat - h)] (Monte Carlo average of the quadratic loss)
% and the exact expected suboptimality gap E[J_T(K_hat)] - J_T^*, both as surfaces over D and as averages over D,
% against the UCRLB and the optimal-bias bound F(b^*).
% Self-contained (local functions below), no toolboxes.
clear; close all; rng(1);
LB = load('lb_results.mat');
if ~isfield(LB,'f_bias') || ~isfield(LB,'f_var')
    error(['lb_results.mat was produced by an older bound script (no bias/variance terms). ' ...
           'Run computation_of_bounds.m once, then rerun compare_ddpc.m.']);
end
q = LB.q; r = LB.r; T = LB.T; Ndata = LB.Ndata; sw2 = LB.sw2; s02 = LB.s02; su2 = LB.su2;
a0 = LB.a0; b0 = LB.b0; rho0 = LB.rho0;

%% ---------------- setting ----------------
Mmc = 1000;                                 % Monte Carlo trials (records) per grid point
nbox = 20;                                  % Cartesian grid: nbox x nbox box, points inside D kept
lam = 10;                                   % DeePC weight of the projection regulariser lam*||(I-Pi)g||^2 (main plots)
lam_list = [0.01 0.1 1 10 100 1e3];         % DeePC regularisation sweep (average risk only); lam -> inf gives SPC
Ldepth = T;                                 % depth of the data matrices: N_T = N-Ldepth+1 columns for ALL horizons
                                            % (paper: N_T = N-T+1; the longest horizon used here is T-1)
names = {'CE','SPC','DeePC','Bayes'}; nctrl = numel(names);

%% ---------------- evaluation grid ----------------
ag = linspace(a0-rho0,a0+rho0,nbox); bg = linspace(b0-rho0,b0+rho0,nbox);
[Ag,Bg] = meshgrid(ag,bg); inD = (Ag-a0).^2+(Bg-b0).^2 <= rho0^2;
idx = find(inD); np = numel(idx);
area_w = pi*rho0^2/np;                      % quadrature weight per inside point (area-normalised)

%% ---------------- Bayes rule: oracle gains and loss weights at the quadrature nodes of D ----------------
% The integrals over D in the Bayes rule are computed on the nodes A,B with the weights wq of lb_results.mat
% (uniform prior on D <=> weights wq).
ngB = numel(LB.A); thB = [LB.A(:) LB.B(:)];              % ngB x 2 nodes theta_k
KB = zeros(ngB,T); wB = zeros(ngB,T);
for k = 1:ngB
    [Kk,Sk,Sgk] = oracle_scalar(thB(k,1),thB(k,2),q,r,s02,sw2,T); KB(k,:) = Kk'; wB(k,:) = (Sk.*Sgk)';
end
wB = wB.*LB.wq(:);                                       % fold the quadrature weights into w_t

%% ---------------- Monte Carlo ----------------
risk = zeros(np,nctrl); gap = zeros(np,nctrl); risk_lam = zeros(np,numel(lam_list));
tic;
for m = 1:np
    a = Ag(idx(m)); b = Bg(idx(m));
    [K,Sc,Sg] = oracle_scalar(a,b,q,r,s02,sw2,T); w = Sc.*Sg;   % oracle gains, curvatures S_t, variances Sigma_t^*, weights W
    acc_r = zeros(1,nctrl); acc_g = zeros(1,nctrl); acc_l = zeros(1,numel(lam_list));
    for mc = 1:Mmc
        [x,u] = gen_data(a,b,Ndata,sw2,su2);
        % ---- CE: one-step least squares (eq. ls) + Riccati recursion at theta_hat (eq. riccati-ce)
        Phi = [x(1:Ndata) u(1:Ndata)]; th = (Phi'*Phi)\(Phi'*x(2:Ndata+1));
        Kce = oracle_scalar(th(1),th(2),q,r,s02,sw2,T);
        % ---- SPC and DeePC gain sequences, shrinking horizon f_t = T-1-t, FIXED number of columns N_T = N-Ldepth+1:
        %      the data matrices are built once per record, horizon f uses their first f rows
        [X0,U,Xf,Cg,Gg] = data_matrices(x,u,Ldepth);                   % eq. hankel (fixed columns)
        Kspc = zeros(T,1); Kd = zeros(T,1); Kd_l = zeros(T,numel(lam_list));
        for t = 0:T-2                                                  % t = T-1: horizon 0, K_hat_{T-1} = 0 for both
            f = T-1-t;
            Kspc(t+1) = spc_gain(Cg,Gg,f,q,r);                          % eq. K-spc  (leading sub-blocks)
            Kd(t+1) = deepc_gain(X0,U,Xf,f,q,r,lam);                   % eq. K-deepc (first f rows)
            for il = 1:numel(lam_list), Kd_l(t+1,il) = deepc_gain(X0,U,Xf,f,q,r,lam_list(il)); end
        end
        % ---- Bayes / average-risk rule (uniform prior on D); decoupled in t since W is diagonal
        Kb = bayes_gains(Phi,x(2:Ndata+1),thB,KB,wB,sw2);
        % ---- quadratic losses and exact suboptimality gaps of this record
        Ks = {Kce, Kspc, Kd, Kb};
        for c = 1:nctrl
            acc_r(c) = acc_r(c) + sum(w.*(Ks{c}-K).^2);
            acc_g(c) = acc_g(c) + gap_exact(Ks{c},K,Sc,a,b,sw2,s02);
        end
        for il = 1:numel(lam_list), acc_l(il) = acc_l(il) + sum(w.*(Kd_l(:,il)-K).^2); end
    end
    risk(m,:) = acc_r/Mmc; gap(m,:) = acc_g/Mmc; risk_lam(m,:) = acc_l/Mmc;
end
fprintf('Monte Carlo done in %.1f s\n', toc);

%% ---------------- bounds on the evaluation grid ----------------
fI  = interp_bound(LB.A,LB.B,LB.f,     Ag(idx),Bg(idx));   % f(theta,b*,b*'): integrand of the optimal-bias bound
uI  = interp_bound(LB.A,LB.B,LB.ucr,   Ag(idx),Bg(idx));   % UCRLB
fbI = interp_bound(LB.A,LB.B,LB.f_bias,Ag(idx),Bg(idx));   % bias term of f
fvI = interp_bound(LB.A,LB.B,LB.f_var, Ag(idx),Bg(idx));   % variance term of f

%% ---------------- report ----------------
fprintf('\nAverage risk over D  (T=%d, N=%d, %d MC trials):\n', T, Ndata, Mmc);
for c = 1:nctrl, fprintf('  %-6s  risk %.4f   suboptimality gap %.4f\n', names{c}, area_w*sum(risk(:,c)), area_w*sum(gap(:,c))); end
fprintf('  optimal-bias bound F(b*) = %.4f   (grid-interpolated %.4f)\n', LB.Fstar, area_w*sum(fI));
fprintf('  int UCRLB               = %.4f   (grid-interpolated %.4f)\n', LB.UCR_int, area_w*sum(uI));
fprintf('  DeePC (projection regulariser) lambda sweep (average risk): %s\n', sprintf('lam=%g: %.4f  ', [lam_list; area_w*sum(risk_lam,1)]));
frac_below = [names; num2cell(mean(risk < uI,1))];        % 2 x nctrl cell: name / fraction
fprintf('  fraction of grid points where each controller''s risk is below the UCRLB: %s\n', sprintf('%s %.2f  ', frac_below{:}));

%% ---------------- colour and font settings (same as the infinite-horizon figures) ----------------
color_F_OPT      = [0.80 0.29 0.24];
color_UCRLB      = [0.16 0.38 0.60];
color_F_OPT_bias = [0.98 0.70 0.28];         % warm amber
color_F_OPT_var  = [0.55 0.12 0.35];         % deep plum
controller_colors = [ ...
    0.55 0.58 0.62;                          % CE
    0.89 0.60 0.20;                          % SPC
    0.42 0.34 0.74;                          % DeePC
    0.20 0.60 0.48];                         % Bayes (average-risk rule)
alpha_Risks = 0.96; FaceAlpha = 0.92; edgeAlpha = 1;
txtFS = 40; axisFS = 40; sliceLW = 3.0;
legend_names = {'$\hat h^{\mathrm{CE}}$', '$\hat h^{\mathrm{SPC}}$', '$\hat h^{\mathrm{DeePC}}$', '$\hat h^{\mathrm{A(P)}}$'};
bound_names  = {'$f(\theta,b,b^\prime)$', 'UCRLB'};
save_pdf = true;                             % export every figure to pdf

%% ---------------- data on the full grid (NaN outside D) ----------------
tofull = @(v) fill_grid(v,idx,size(Ag));
Agrid = Ag; Bgrid = Bg;
risk_F_OPT      = tofull(fI);
risk_U_derived  = tofull(uI);
risk_F_OPT_bias = tofull(fbI);
risk_F_OPT_var  = tofull(fvI);
risks = cell(1,nctrl); gaps = cell(1,nctrl);
for c = 1:nctrl, risks{c} = tofull(risk(:,c)); gaps{c} = tofull(gap(:,c)); end
a_axis = Agrid(1,:); b_axis = Bgrid(:,1);
[~,idxA5] = min(abs(a_axis - a0));           % column: slice at a = a0 (b varies)
[~,idxB5] = min(abs(b_axis - b0));           % row:    slice at b = b0 (a varies)

%% 1. Two bounds
figure('Color','w');
surf_solid(Agrid,Bgrid,risk_F_OPT,color_F_OPT,FaceAlpha,edgeAlpha,'-',0.08); hold on;
surf_solid(Agrid,Bgrid,risk_U_derived,color_UCRLB,FaceAlpha,edgeAlpha,'-',0.08);
style_3d('Risk',txtFS,axisFS);
legend(bound_names,'Interpreter','latex','Location','eastoutside','FontSize',txtFS);
export_pdf(save_pdf,'two_bounds.pdf');

%% 2. Bias and variance terms of the optimal-bias bound
figure('Color','w');
surf_solid(Agrid,Bgrid,risk_F_OPT,color_F_OPT,FaceAlpha,edgeAlpha,'-',0.08); hold on;
surf_solid(Agrid,Bgrid,risk_F_OPT_bias,color_F_OPT_bias,FaceAlpha,edgeAlpha,'--',0.08,color_F_OPT_bias*0.65);
surf_solid(Agrid,Bgrid,risk_F_OPT_var,color_F_OPT_var,FaceAlpha,edgeAlpha,':',0.08,color_F_OPT_var*0.65);
style_3d('Risk',txtFS,axisFS);
legend({'$f(\theta,b,b^\prime)$','Bias term','Variance term'},'Interpreter','latex','Location','eastoutside','FontSize',txtFS);
export_pdf(save_pdf,'bound_bias_variance.pdf');

%% 3. All controllers: risk
figure('Color','w','Position',[100 100 1100 800]); hold on;
for k = 1:nctrl, surf_ctrl(Agrid,Bgrid,risks{k},controller_colors(k,:),alpha_Risks,edgeAlpha); end
style_3d('Risk',txtFS,axisFS);
legend(legend_names,'Interpreter','latex','Location','eastoutside','FontSize',txtFS);
export_pdf(save_pdf,'all_risks.pdf');

%% 4. Each controller versus the two bounds
for k = 1:nctrl
    figure('Color','w');
    surf_solid(Agrid,Bgrid,risk_F_OPT,color_F_OPT,FaceAlpha,edgeAlpha,'-',0.08); hold on;
    surf_solid(Agrid,Bgrid,risk_U_derived,color_UCRLB,FaceAlpha,edgeAlpha,'-',0.08);
    surf_solid(Agrid,Bgrid,risks{k},controller_colors(k,:),FaceAlpha,edgeAlpha,'-',0.08);
    style_3d('Risk',txtFS,axisFS);
    legend([bound_names, legend_names(k)],'Interpreter','latex','FontSize',txtFS,'Location','best');
    export_pdf(save_pdf,sprintf('comparison_%s.pdf',names{k}));
end

%% 5. All controllers: exact suboptimality gap
figure('Color','w','Position',[100 100 1100 800]); hold on;
for k = 1:nctrl, surf_ctrl(Agrid,Bgrid,gaps{k},controller_colors(k,:),alpha_Risks,edgeAlpha); end
style_3d('Gap',txtFS,axisFS);
legend(legend_names,'Interpreter','latex','Location','eastoutside','FontSize',txtFS);
export_pdf(save_pdf,'all_gaps.pdf');

%% 6. Global slices (all controllers): risk and gap, at a = a0 and b = b0
slice_all(b_axis, cellfun(@(Z) Z(:,idxA5), risks,'UniformOutput',false), controller_colors, legend_names, '$b$', 'Risk', sliceLW, txtFS, axisFS);
export_pdf(save_pdf,'global_slice_risk_a05.pdf');
slice_all(a_axis, cellfun(@(Z) Z(idxB5,:), risks,'UniformOutput',false), controller_colors, legend_names, '$a$', 'Risk', sliceLW, txtFS, axisFS);
export_pdf(save_pdf,'global_slice_risk_b05.pdf');
slice_all(b_axis, cellfun(@(Z) Z(:,idxA5), gaps,'UniformOutput',false), controller_colors, legend_names, '$b$', 'Optimality Gap', sliceLW, txtFS, axisFS);
export_pdf(save_pdf,'global_slice_gap_a05.pdf');
slice_all(a_axis, cellfun(@(Z) Z(idxB5,:), gaps,'UniformOutput',false), controller_colors, legend_names, '$a$', 'Optimality Gap', sliceLW, txtFS, axisFS);
export_pdf(save_pdf,'global_slice_gap_b05.pdf');

%% 7. Individual slices: each controller versus the two bounds
for k = 1:nctrl
    cols = [color_F_OPT; color_UCRLB; controller_colors(k,:)];
    slice_all(b_axis, {risk_F_OPT(:,idxA5), risk_U_derived(:,idxA5), risks{k}(:,idxA5)}, cols, ...
        [bound_names, legend_names(k)], '$b$', 'Risk', sliceLW, txtFS, axisFS);
    export_pdf(save_pdf,sprintf('slice_a05_%s.pdf',names{k}));
    slice_all(a_axis, {risk_F_OPT(idxB5,:), risk_U_derived(idxB5,:), risks{k}(idxB5,:)}, cols, ...
        [bound_names, legend_names(k)], '$a$', 'Risk', sliceLW, txtFS, axisFS);
    export_pdf(save_pdf,sprintf('slice_b05_%s.pdf',names{k}));
end

%% 8. Slices of the bound: total, bias term, variance term
cols = [color_F_OPT; color_F_OPT_bias; color_F_OPT_var];
bv_names = {'$f(\theta,b,b^\prime)$','Bias term','Variance term'};
slice_all(b_axis, {risk_F_OPT(:,idxA5), risk_F_OPT_bias(:,idxA5), risk_F_OPT_var(:,idxA5)}, cols, bv_names, ...
    '$b$', 'Risk', sliceLW, txtFS, axisFS, {'-','--',':'});
export_pdf(save_pdf,'slice_bound_a05.pdf');
slice_all(a_axis, {risk_F_OPT(idxB5,:), risk_F_OPT_bias(idxB5,:), risk_F_OPT_var(idxB5,:)}, cols, bv_names, ...
    '$a$', 'Risk', sliceLW, txtFS, axisFS, {'-','--',':'});
export_pdf(save_pdf,'slice_bound_b05.pdf');

%% ======================= local functions =======================
% ---- oracle and closed-loop quantities ----
function [K,S,Sg] = oracle_scalar(a,b,q,r,s02,sw2,T)
% finite-horizon LQ oracle at theta = (a,b) (Lemma lemma:dp, P_T = 0):  backward Riccati recursion
%     S_t = r + b^2 P_{t+1},   K_t^* = -a b P_{t+1}/S_t,   P_t = q + a^2 r P_{t+1}/S_t,
% and the oracle state variance  Sigma_0^* = s02,  Sigma_{t+1}^* = (a + b K_t^*)^2 Sigma_t^* + sw2.
% Outputs (T x 1, entry t+1 <-> time step t): K = K_t^*, S = S_t, Sg = Sigma_t^*.
P = 0; K = zeros(T,1); S = zeros(T,1);
for t = T:-1:1, S(t) = r + b^2*P; K(t) = -a*b*P/S(t); P = q + a^2*r*P/S(t); end
Sg = zeros(T,1); Sg(1) = s02;
for t = 1:T-1, Sg(t+1) = (a+b*K(t))^2*Sg(t) + sw2; end
end

function g = gap_exact(Kh,K,S,a,b,sw2,s02)
% exact E[J_T(K_hat)] - J_T^* for the gain sequence K_hat (performance-difference identity):
%     sum_t S_t (K_hat_t - K_t^*)^2 Sigma_t^{K_hat},   Sigma_{t+1}^{K_hat} = (a + b K_hat_t)^2 Sigma_t^{K_hat} + sw2,
% i.e. the quadratic loss with the state variance under K_hat instead of the oracle variance Sigma_t^*.
g = 0; Sg = s02;
for t = 1:numel(K), g = g + S(t)*(Kh(t)-K(t))^2*Sg; Sg = (a+b*Kh(t))^2*Sg + sw2; end
end

% ---- identification record ----
function [x,u] = gen_data(a,b,N,sw2,su2)
% open-loop record x_0 ... x_N, u_0 ... u_{N-1} with u_k ~ N(0,su2) white and stationary start
% (for a record starting from x_0 ~ N(0,s02), as assumed by the Fisher information in computation_of_bounds.m,
%  replace the initial state by  x(1) = sqrt(s02)*randn  and pass s02 as an extra argument)
u = sqrt(su2)*randn(N,1); x = zeros(N+1,1); x(1) = sqrt((b^2*su2+sw2)/(1-a^2))*randn;
for k = 1:N, x(k+1) = a*x(k) + b*u(k) + sqrt(sw2)*randn; end
end

function [X0,U,Xf,Cg,Gg] = data_matrices(x,u,L)
% data matrices of eq. (hankel) with a FIXED number of columns N_T = N-L+1 for every horizon (L = depth):
%   X0 = [x_0 ... x_{N_T-1}]            (1 x N_T)   initial states (the same for all horizons)
%   U  = [u_k ... u_{k+L-1}]_k          (L x N_T)   future inputs,  horizon f uses the rows 1..f
%   Xf = [x_{k+1} ... x_{k+L}]_k        (L x N_T)   future states,  horizon f uses the rows 1..f
% and the products needed by SPC for all horizons at once (eq. spc-ls):
%   Cg = Xf*Phi', Gg = Phi*Phi' with Phi = [X0; U];  for horizon f use Cg(1:f,1:1+f), Gg(1:1+f,1:1+f)
% x = [x_0 ... x_N]', u = [u_0 ... u_{N-1}]'  (N transitions)
N = numel(u); Nc = N-L+1;
X0 = x(1:Nc)'; U = zeros(L,Nc); Xf = zeros(L,Nc);
for i = 1:L, U(i,:) = u(i:i+Nc-1)'; Xf(i,:) = x(i+1:i+Nc)'; end
Phi = [X0; U]; Cg = Xf*Phi'; Gg = Phi*Phi';
end

% ---- the four data-driven controllers ----
function K = spc_gain(Cg,Gg,f,q,r)
% SPC with fixed columns: [Gamma_f H_f] = X_f (Phi_f)' (Phi_f (Phi_f)')^{-1} (eq. spc-ls), where X_f (Phi_f)' and
% Phi_f (Phi_f)' are the leading sub-blocks of Cg and Gg; then the first gain of the horizon-f LQ problem with the
% predictor x_f = Gamma_f x_t + H_f u_f (Lemma "gain of a direct method"):
%     K = -e_1' (H'QH + R)^{-1} H'Q Gamma,   Q = q I_f,  R = r I_f   (eq. K-spc)
GH = Cg(1:f,1:1+f)/Gg(1:1+f,1:1+f); Gam = GH(:,1); H = GH(:,2:end);
K = -[1 zeros(1,f-1)]*((q*(H'*H) + r*eye(f))\(q*H'*Gam));
end

function K = deepc_gain(X0,U,Xf,f,q,r,lam)
% DeePC with the projection-based regulariser lam*||(I - Pi_f) g||^2 (Dorfler et al.), eq. (deepc), on the
% fixed-column matrices: g in R^{N_T} for every horizon, Phi_f = [X0; U(1:f,:)] the first 1+f rows of the
% depth-L data matrix, Pi_f = Phi_f' (Phi_f Phi_f')^{-1} Phi_f the orthogonal projector onto its row space.
% Eliminating u_f = U_f g and x_f = X_f g gives  min_g g'*Qg*g  s.t.  X0 g = x_t  (eq. deepc-g) with
%     Qg = X_f' Q X_f + U_f' R U_f + lam*(I - Pi_f),
% which is positive definite iff [U_f; X_f] Phi_f' has full column rank (a.s. for noisy data).
% Explicit KKT solution (eq. K-deepc):  g^* = Qg^{-1} X0' (X0 Qg^{-1} X0')^{-1} x_t,  K = e_1' U_f Qg^{-1} X0' / (X0 Qg^{-1} X0').
% lam -> inf recovers the SPC gain (g confined to the row space of Phi_f); lam -> 0 shrinks the gain to zero.
Nc = size(X0,2); Uf = U(1:f,:); Xff = Xf(1:f,:);
Phi = [X0; Uf]; Pi_f = Phi'*((Phi*Phi')\Phi);                    % orthogonal projector onto rowspace(Phi_f)
Qg = q*(Xff'*Xff) + r*(Uf'*Uf) + lam*(eye(Nc) - Pi_f); Qg = (Qg+Qg')/2;
v = Qg\X0';                                                      % Qg^{-1} X0'
K = (Uf(1,:)*v)/(X0*v);
end

function Kb = bayes_gains(Phi,X1,thB,KB,wB,sw2)
% average-risk rule with uniform prior on D for the loss  sum_t w_t (K_hat_t - K_t^*)^2:
%     K_hat_t = sum_k wB(k,t) K_t^*(theta_k) p(Z|theta_k) / sum_k wB(k,t) p(Z|theta_k)
% with p(Z|theta) = exp( -(1/2sw2) sum_k (x_{k+1} - a x_k - b u_k)^2 )  (conditional on x_0), evaluated on the
% quadrature nodes theta_k of D (the weights wq are already folded into wB).
% Phi = [x_k u_k]_k (N x 2) and X1 = [x_{k+1}]_k (N x 1) are the regressors and the targets of the record, so that
%     sum_k (x_{k+1} - theta' phi_k)^2 = ||X1||^2 - 2 theta' Phi' X1 + theta' Phi' Phi theta.
PhiX = Phi'*X1; PhiPhi = Phi'*Phi;
ll = -(1/(2*sw2))*(sum(X1.^2) - 2*(thB*PhiX) + sum((thB*PhiPhi).*thB,2));  % log-likelihood at every node (ngB x 1)
p = exp(ll - max(ll));                                                       % likelihood, scaled for numerical safety
num = (wB.*KB)'*p; den = wB'*p;
Kb = num./den;
end

% ---- interpolation of the bounds and plotting helpers ----
function v = interp_bound(Ap,Bp,F,xq,yq)
% interpolate a bound from the mesh nodes to the evaluation grid; nearest neighbour at the rim
v = griddata(Ap(:),Bp(:),F(:),xq,yq,'linear');
bad = isnan(v); if any(bad), v(bad) = griddata(Ap(:),Bp(:),F(:),xq(bad),yq(bad),'nearest'); end
end

function G = fill_grid(v,idx,sz)
% values at the inside points -> full grid with NaN outside D
G = NaN(sz); G(idx) = v;
end

function surf_solid(A,B,Z,col,fa,ea,ls,spec,edgecol)
% solid-colour surface (bounds, and a controller in the individual comparisons)
if nargin < 9, edgecol = [0 0 0]; end
surf(A,B,Z,'FaceColor',col,'EdgeColor',edgecol,'EdgeAlpha',ea,'LineStyle',ls, ...
    'FaceAlpha',fa,'SpecularStrength',spec);
end

function surf_ctrl(A,B,Z,col,fa,ea)
% solid-colour surface for a controller in the combined plots
surf(A,B,Z,'FaceColor',col,'EdgeColor',[0 0 0],'EdgeAlpha',ea,'LineStyle','-', ...
    'FaceAlpha',fa,'SpecularStrength',0.10,'DiffuseStrength',0.85,'AmbientStrength',0.55);
end

function style_3d(zl,txtFS,axisFS)
xlabel('$a$','Interpreter','latex','FontSize',txtFS);
ylabel('$b$','Interpreter','latex','FontSize',txtFS);
zlabel(zl,'Interpreter','latex','FontSize',txtFS);
view(135,28);
ax = gca; grid(ax,'off'); box(ax,'off'); set(ax,'Layer','top');
set(ax,'LineWidth',1.0,'FontSize',axisFS,'Color','none');
camlight('headlight'); camlight('right'); lighting gouraud; material dull;
end

function slice_all(x,Ys,cols,names,xl,yl,lw,txtFS,axisFS,styles)
% one 2-D figure with one line per entry of Ys (NaN outside D is left as a gap)
if nargin < 10, styles = repmat({'-'},1,numel(Ys)); end
figure('Color','w','Position',[100 100 950 750]); hold on;
for k = 1:numel(Ys)
    plot(x(:), Ys{k}(:), 'Color', cols(k,:), 'LineWidth', lw, 'LineStyle', styles{k});
end
xlabel(xl,'Interpreter','latex','FontSize',txtFS); ylabel(yl,'Interpreter','latex','FontSize',txtFS);
grid off; box on; set(gca,'FontSize',axisFS,'TickLabelInterpreter','latex');
legend(names,'Interpreter','latex','Location','best','FontSize',txtFS);
end

function export_pdf(flag,fname)
if ~flag, return; end
if exist('exportgraphics','file') || exist('exportgraphics','builtin')
    exportgraphics(gcf, fname, 'Resolution', 600);
else
    print(gcf, fname, '-dpdf');                 % fallback for older MATLAB / Octave
end
end
