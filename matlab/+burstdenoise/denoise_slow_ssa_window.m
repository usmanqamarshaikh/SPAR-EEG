function [y_hat, debug] = denoise_slow_ssa_window(y_in, Fs, cfg)
% denoise_slow_ssa_window
% SSA-based slow-artifact denoiser for broad low-frequency contamination.
%
% Purpose:
%   - Handle slow motion artifact / baseline sway / electrode movement drift
%   - Unlike blink-oriented EOG pass, this method does NOT require a clean
%     baseline region inside the epoch
%   - Uses SSA to identify suspicious slow RCs, builds a slow-artifact
%     estimate x_art, then subtracts it softly: y_hat = y - alpha * x_art
%
% Design:
%   Stage 1 gate (cheap, on normalized epoch):
%       - robust peak-to-peak
%       - LF power ratio
%       - half-epoch median shift
%       - trend correlation
%
%   Stage 2 gate (after SSA / x_art):
%       - variance ratio of x_art relative to epoch
%       - correlation between x_art and epoch
%
%   If both gates pass:
%       optimize alpha over [alpha_min, alpha_max]
%   else:
%       bypass the epoch unchanged
%
% INPUTS
%   y_in : [T x 1] or [1 x T]
%   Fs   : sampling frequency (Hz)
%   cfg  : struct of hyperparameters
%
% OUTPUTS
%   y_hat : denoised signal, same shape as y_in
%   debug : struct with intermediate variables

if nargin < 3, cfg = struct(); end
cfg = fill_defaults(cfg);

was_row = isrow(y_in);

% vectorize
y = y_in(:);
T = numel(y);

% -------------------- normalize --------------------
if cfg.do_normalize
    switch lower(string(cfg.scale_method))
        case "mad"
            s = mad(y,1) + cfg.scale_floor;
        case "rms"
            s = sqrt(mean(y.^2)) + cfg.scale_floor;
        otherwise
            s = mad(y,1) + cfg.scale_floor;
    end
else
    s = 1.0;
end
yn = y / s;

% -------------------- Stage 1 pre-gate --------------------
mid = floor(T/2);
trend_g1 = smoothdata(yn, 'movmean', max(3, round(cfg.g1_trend_smooth_sec * Fs)));

g1_f_p2p   = robust_ptp(yn);
g1_f_lfr   = bandpower_ratio_fft(yn, Fs, 0, cfg.g1_lf_max_Hz);
g1_f_shift = abs(median(yn(1:mid)) - median(yn(mid+1:end)));
g1_f_trcor = abs_corr_safe(yn, trend_g1);

g1_s_p2p   = ramp01(g1_f_p2p,   cfg.g1_p2p_lo,    cfg.g1_p2p_hi);
g1_s_lfr   = ramp01(g1_f_lfr,   cfg.g1_lfr_lo,    cfg.g1_lfr_hi);
g1_s_shift = ramp01(g1_f_shift, cfg.g1_shift_lo,  cfg.g1_shift_hi);
g1_s_trcor = ramp01(g1_f_trcor, cfg.g1_trcorr_lo, cfg.g1_trcorr_hi);

gate1_score = ...
    cfg.g1_w_p2p    * g1_s_p2p + ...
    cfg.g1_w_lfr    * g1_s_lfr + ...
    cfg.g1_w_shift  * g1_s_shift + ...
    cfg.g1_w_trcorr * g1_s_trcor;

gate1_pass = (gate1_score >= cfg.g1_thr);

% Optional fast skip before SSA
if cfg.fast_skip_on_gate1_fail && ~gate1_pass
    y_hat = y;
    if was_row, y_hat = y_hat.'; end

    debug = pack_debug( ...
        cfg, s, yn, y_hat, [], [], [], [], [], [], ...
        [], [], [], [], gate1_score, gate1_pass, ...
        [], [], [], [], [], [], [], false, ...
        [], [], [], [], [], [], [], []);
    return;
end

% -------------------- SSA decomposition --------------------
N = T;
L = round(cfg.L_sec * Fs);
L = max(cfg.L_min, L);
L = min(L, floor(cfg.L_max_fracN * N));
if L >= N
    L = max(cfg.L_min, floor(0.5 * N));
end

[RC0, sing0] = ssa_rcs(yn, L);   % [K x T]
K = size(RC0,1);

% order RCs by spectral centroid (low -> high)
fc0 = spectral_centroid_rows(RC0, Fs);
[fc_sorted, ord] = sort(fc0(:), 'ascend');
RC = RC0(ord,:);
sing = sing0(ord);

rc_var = var(RC, 0, 2);
var_share = rc_var / (sum(rc_var) + 1e-12);

% -------------------- score suspicious slow RCs --------------------
trend_rc = smoothdata(yn, 'movmean', max(3, round(cfg.trend_smooth_sec * Fs)));

epoch_p2p = robust_ptp(yn);
rc_p2p    = zeros(K,1);
fc_score  = zeros(K,1);
var_score = zeros(K,1);
ptp_score = zeros(K,1);
trd_score = zeros(K,1);
score     = zeros(K,1);
corr_tr   = zeros(K,1);

for k = 1:K
    rc_p2p(k) = robust_ptp(RC(k,:));

    fc_score(k)  = clamp01(1 - fc_sorted(k) / cfg.score_fc_max_Hz);
    var_score(k) = clamp01(var_share(k) / cfg.var_share_ref);
    ptp_score(k) = clamp01(rc_p2p(k) / (cfg.rc_p2p_frac_ref * epoch_p2p + 1e-12));

    corr_tr(k)   = abs_corr_safe(RC(k,:), trend_rc);
    trd_score(k) = clamp01(corr_tr(k));

    score(k) = ...
        cfg.w_fc    * fc_score(k) + ...
        cfg.w_var   * var_score(k) + ...
        cfg.w_ptp   * ptp_score(k) + ...
        cfg.w_trend * trd_score(k);
end

is_slow_band = (fc_sorted <= cfg.score_fc_max_Hz);
cand_idx = find(is_slow_band & score >= cfg.score_thr);

if ~isempty(cand_idx)
    [~, ix] = sort(score(cand_idx), 'descend');
    cand_idx = cand_idx(ix);
    cand_idx = cand_idx(1:min(cfg.max_slow_rcs, numel(cand_idx)));
end

% -------------------- build slow-artifact estimate x_art --------------------
x_art = zeros(T,1);
wk    = zeros(K,1);

if ~isempty(cand_idx)
    switch lower(string(cfg.artifact_weight_mode))
        case "uniform"
            wraw = ones(size(cand_idx));
        case "maxnorm"
            wraw = score(cand_idx);
            wraw = wraw / (max(wraw) + 1e-12);
        otherwise % "sumnorm"
            wraw = score(cand_idx);
            wraw = wraw / (sum(wraw) + 1e-12);
    end

    wk(cand_idx) = wraw;

    for ii = 1:numel(cand_idx)
        k = cand_idx(ii);
        x_art = x_art + wk(k) * RC(k,:).';
    end
end

% -------------------- Stage 2 confirmation gate --------------------
g2_varratio = var(x_art) / (var(yn) + 1e-12);
g2_corr     = abs_corr_safe(yn, x_art);

g2_s_varratio = ramp01(g2_varratio, cfg.g2_varratio_lo, cfg.g2_varratio_hi);
g2_s_corr     = ramp01(g2_corr,     cfg.g2_corr_lo,     cfg.g2_corr_hi);

gate2_score = ...
    cfg.g2_w_varratio * g2_s_varratio + ...
    cfg.g2_w_corr     * g2_s_corr;

gate2_pass = (gate2_score >= cfg.g2_thr);

% -------------------- final gating --------------------
has_artifact_est = any(abs(x_art) > cfg.x_art_eps);
do_slowpass = gate1_pass && gate2_pass && has_artifact_est;

% -------------------- optimize alpha --------------------
alpha_grid = linspace(cfg.alpha_min, cfg.alpha_max, cfg.n_alpha);
J = nan(size(alpha_grid));

if do_slowpass
    for i = 1:numel(alpha_grid)
        a = alpha_grid(i);
        ytmp = yn - a * x_art;
        J(i) = slow_obj(ytmp, yn, x_art, Fs, cfg);
    end
    [~, ibest] = min(J);
    alpha_star = alpha_grid(ibest);
else
    alpha_star = 0;
end

alpha_eff = alpha_star;

% -------------------- reconstruct --------------------
if do_slowpass
    yhat_n = yn - alpha_eff * x_art;
else
    yhat_n = yn;
end

y_hat = yhat_n * s;

if was_row
    y_hat = y_hat.';
end

% -------------------- debug --------------------
debug = pack_debug( ...
    cfg, s, yn, y_hat, RC, fc_sorted, sing, var_share, score, cand_idx, ...
    wk, x_art, alpha_grid, J, gate1_score, gate1_pass, ...
    gate2_score, gate2_pass, alpha_star, alpha_eff, ...
    g1_f_p2p, g1_f_lfr, g1_f_shift, g1_f_trcor, ...
    g2_varratio, g2_corr, do_slowpass, L, ...
    fc_score, var_score, ptp_score, trd_score);

end

%% ===================== defaults + helpers =====================

function cfg = fill_defaults(cfg)
d = struct();

% ---------- normalize ----------
d.do_normalize = true;
d.scale_method = "mad";
d.scale_floor  = 1e-6;

% ---------- SSA ----------
d.L_sec       = 0.01;
d.L_min       = 12;
d.L_max_fracN = 0.7;

% ---------- slow RC candidate selection ----------
d.score_fc_max_Hz = 3.0;
d.max_slow_rcs    = 2;
d.score_thr       = 0.5;

d.w_fc    = 0;
d.w_var   = 0.35;
d.w_ptp   = 0.35;
d.w_trend = 0.30;

d.var_share_ref   = 0.40;
d.rc_p2p_frac_ref = 0.40;
d.trend_smooth_sec = 0.35;

% ---------- Stage 1 gate ----------
d.g1_lf_max_Hz        = 4.0;
d.g1_trend_smooth_sec = 0.35;

d.g1_p2p_lo   = 3.0;
d.g1_p2p_hi   = 8.0;

d.g1_lfr_lo   = 0.9;
d.g1_lfr_hi   = 1;

d.g1_shift_lo = 0.50;
d.g1_shift_hi = 2.00;
d.g1_trcorr_lo = 0.8;
d.g1_trcorr_hi = 0.99;

d.g1_w_p2p    = 0.0;
d.g1_w_lfr    = 1.0;
d.g1_w_shift  = 0.0;
d.g1_w_trcorr = 0.0;

d.g1_thr = 0.5;

% ---------- Stage 2 gate ----------
d.g2_varratio_lo = 0.40;
d.g2_varratio_hi = 0.80;
d.g2_corr_lo     = 0.70;
d.g2_corr_hi     = 0.99;

d.g2_w_varratio = 0.50;
d.g2_w_corr     = 0.50;

d.g2_thr = 0.5;

% ---------- alpha optimization ----------
d.alpha_min = 0.00;
d.alpha_max = 1.20;
d.n_alpha   = 301;

% ---------- objective ----------
d.obj_lf_max_Hz = 4.0;
d.obj_w_lf      = 0.40;
d.obj_w_p2p     = 0.25;
d.obj_w_slope   = 0.10;
d.obj_w_corr    = 0.10;
d.obj_w_change  = 0.05;

% ---------- misc ----------
d.x_art_eps = 1e-10;
d.fast_skip_on_gate1_fail = true;   % true for runtime speed, false for richer debug
d.artifact_weight_mode    = "maxnorm"; % "sumnorm" | "maxnorm" | "uniform"

cfg = merge_structs(d, cfg);
end

function out = merge_structs(a,b)
out = a;
if isempty(b), return; end
f = fieldnames(b);
for i = 1:numel(f)
    out.(f{i}) = b.(f{i});
end
end

function [RC, sing] = ssa_rcs(y, L)
y = y(:).';
N = numel(y);
K = N - L + 1;

X = hankel(y(1:L), y(L:end));
[U,S,V] = svd(X, 'econ');
r = min(size(S,1), size(S,2));

RC = zeros(r, N);
sing = diag(S);

for i = 1:r
    Xi = S(i,i) * (U(:,i) * V(:,i).');
    RC(i,:) = hankelize(Xi);
end
end

function y_rec = hankelize(Xi)
[L,K] = size(Xi);
N = L + K - 1;
y_rec = zeros(1,N);
cnt   = zeros(1,N);

for i = 1:L
    for j = 1:K
        t = i + j - 1;
        y_rec(t) = y_rec(t) + Xi(i,j);
        cnt(t)   = cnt(t) + 1;
    end
end
y_rec = y_rec ./ max(cnt,1);
end

function fc = spectral_centroid_rows(X, Fs)
[K,T] = size(X);
Nfft = max(256, 2^nextpow2(T));
f = (0:(Nfft/2)) * (Fs/Nfft);
fc = zeros(K,1);

for k = 1:K
    x = X(k,:);
    Xf = fft(x, Nfft);
    P  = abs(Xf(1:Nfft/2+1)).^2;
    fc(k) = (f(:)' * P(:)) / (sum(P) + 1e-12);
end
end

function y = clamp01(x)
y = min(max(x,0),1);
end

function y = ramp01(x, lo, hi)
if hi <= lo
    y = double(x >= hi);
else
    y = clamp01((x - lo) / (hi - lo));
end
end

function v = robust_ptp(x)
x = x(:);
v = prctile(x,95) - prctile(x,5);
end

function r = abs_corr_safe(x, y)
x = x(:);
y = y(:);

x = x - mean(x);
y = y - mean(y);

nx = norm(x);
ny = norm(y);

if nx < 1e-12 || ny < 1e-12
    r = 0;
else
    r = abs((x' * y) / (nx * ny));
end
end

function ratio = bandpower_ratio_fft(x, Fs, f1, f2)
x = x(:).';
N = numel(x);
Nfft = max(256, 2^nextpow2(N));
Xf = fft(x, Nfft);
P  = abs(Xf(1:Nfft/2+1)).^2;
f  = (0:(Nfft/2)) * (Fs/Nfft);

idx_band = (f >= f1) & (f <= f2);
ratio = sum(P(idx_band)) / (sum(P) + 1e-12);
end

function J = slow_obj(yhat, yin, xart, Fs, cfg)
lf0   = bandpower_ratio_fft(yin,  Fs, 0, cfg.obj_lf_max_Hz);
p2p0  = robust_ptp(yin);
sl0   = sqrt(mean(diff(yin).^2));
chg0  = rms_safe(yin);

lf1   = bandpower_ratio_fft(yhat, Fs, 0, cfg.obj_lf_max_Hz);
p2p1  = robust_ptp(yhat);
sl1   = sqrt(mean(diff(yhat).^2));
corr1 = abs_corr_safe(yhat, xart);
chg1  = rms_safe(yhat - yin) / (chg0 + 1e-12);

lf_term     = lf1   / (lf0  + 1e-12);
p2p_term    = p2p1  / (p2p0 + 1e-12);
slope_term  = sl1   / (sl0  + 1e-12);
corr_term   = corr1;
change_term = chg1;

J = cfg.obj_w_lf     * lf_term + ...
    cfg.obj_w_p2p    * p2p_term + ...
    cfg.obj_w_slope  * slope_term + ...
    cfg.obj_w_corr   * corr_term + ...
    cfg.obj_w_change * change_term;
end

function r = rms_safe(x)
x = x(:);
r = sqrt(mean(x.^2));
end

function debug = pack_debug( ...
    cfg, s, yn, y_hat, RC, fc_sorted, sing, var_share, score, cand_idx, ...
    wk, x_art, alpha_grid, J, gate1_score, gate1_pass, ...
    gate2_score, gate2_pass, alpha_star, alpha_eff, ...
    g1_f_p2p, g1_f_lfr, g1_f_shift, g1_f_trcor, ...
    g2_varratio, g2_corr, do_slowpass, L, ...
    fc_score, var_score, ptp_score, trd_score)

debug = struct();

debug.cfg         = cfg;
debug.scale       = s;
debug.yn          = yn;
debug.y_hat       = y_hat;

debug.RC          = RC;
debug.fc          = fc_sorted;
debug.sing        = sing;
debug.var_share   = var_share;

debug.score       = score;
debug.cand_idx    = cand_idx;
debug.wk          = wk;
debug.x_art       = x_art;

debug.alpha_grid  = alpha_grid;
debug.J           = J;
debug.alpha_star  = alpha_star;
debug.alpha_eff   = alpha_eff;

debug.gate1_score = gate1_score;
debug.gate1_pass  = gate1_pass;
debug.gate2_score = gate2_score;
debug.gate2_pass  = gate2_pass;
debug.do_slowpass = do_slowpass;

debug.g1_features = struct( ...
    'robust_ptp', g1_f_p2p, ...
    'lf_ratio',   g1_f_lfr, ...
    'half_shift', g1_f_shift, ...
    'trend_corr', g1_f_trcor);

debug.g2_features = struct( ...
    'var_ratio', g2_varratio, ...
    'corr_xart', g2_corr);

debug.rc_subscores = struct( ...
    'fc_score',   fc_score, ...
    'var_score',  var_score, ...
    'ptp_score',  ptp_score, ...
    'trend_score', trd_score);

debug.L = L;
end