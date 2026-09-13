function [y_hat, debug] = denoise_emg_vmd_window(y_in, Fs, cfg)
% denoise_emg_vmd_window
% Demo-matched EMG denoiser:
%   VMD + common mask + per-mode region-wise attenuation optimization
%
% Patched to match demo_emg_vmd_stepbystep_PATCHED_gamma.m as closely as possible:
%   - Demo-matched defaults
%   - Constant dilation by default
%   - Gamma-shaped gain:
%         gain = 1 - (1-a) * (m.^gamma)
%         gamma = min(1, eta*a/(1-a))
%   - NO extra hard-support clipping of mrC outside (mrA > 0.5)
%   - Keeps demo logic: mrC(clean_idx) = 0
%
% INPUTS
%   y_in : [T x 1] or [1 x T]
%   Fs   : sampling frequency (Hz)
%   cfg  : struct of hyperparameters
%
% OUTPUTS
%   y_hat : denoised signal, same shape as y_in
%   debug : struct with intermediate variables

% -------------------- defaults --------------------
if nargin < 3, cfg = struct(); end
cfg = fill_defaults(cfg);

% remember input orientation
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

% -------------------- VMD decomposition --------------------
try
    [imf,~,info] = vmd(yn, ...
        NumIMFs=cfg.NumIMFs, ...
        PenaltyFactor=cfg.PenaltyFactor, ...
        InitializeMethod=string(cfg.InitMethod), ...
        MaxIterations=cfg.MaxIterations, ...
        Display=false);
catch
    [imf,~] = vmd(yn, 'NumIMFs', cfg.NumIMFs);
    info = struct();
end

% Convert to [K x T]
if size(imf,1) == T
    U0 = imf.';
else
    U0 = imf;
end
K = size(U0,1);

% -------------------- center frequencies --------------------
fc = [];
if isstruct(info)
    if isfield(info,'CentralFrequencies')
        fc = info.CentralFrequencies(:) * Fs;
    elseif isfield(info,'CenterFrequencies')
        fc = info.CenterFrequencies(:) * Fs;
    end
end
if isempty(fc) || numel(fc) ~= K
    fc = spectral_centroid_rows(U0, Fs);
end

% sort low -> high
[fc_sorted, ord] = sort(fc(:), 'ascend');
U = U0(ord,:);

% -------------------- Step 3: proxy + zscore --------------------
mask_fc_max = min(cfg.mask_fc_max_Hz, 0.99*(Fs/2));
mask_modes = find(fc_sorted >= cfg.mask_fc_min_Hz & fc_sorted <= mask_fc_max);
if isempty(mask_modes)
    mask_modes = ceil(2*K/3):K;
end

envM = zeros(numel(mask_modes), T);
for ii = 1:numel(mask_modes)
    k = mask_modes(ii);
    envM(ii,:) = abs(hilbert(U(k,:)));
end

proxy = median(envM,1);
proxy = smoothdata(proxy, 'movmean', max(1, round(cfg.env_smooth_sec*Fs)));
z = robust_zscore(proxy);

% -------------------- Step 4: common mask + regions --------------------
m0 = (z >= cfg.mask_zthr);

m_common = dilate_binary(m0, round(cfg.mask_dilate_sec_base*Fs));
m_common = smoothdata(double(m_common), 'movmean', max(1, round(cfg.mask_smooth_sec*Fs)));
m_common = min(max(m_common,0),1);

bin_mask = (m_common > 0.5);
regions = find_regions(bin_mask);

minL = max(3, round(cfg.min_burst_sec*Fs));
if ~isempty(regions)
    regions = regions((regions(:,2)-regions(:,1)+1) >= minL, :);
end
R = size(regions,1);

% -------------------- Step 5: baseline + treated modes --------------------
clean_idx = (m_common < cfg.baseline_m_thr);
minBase = max(5, round(cfg.min_baseline_sec*Fs));

treat_fc_max = min(cfg.treat_fc_max_Hz, 0.99*(Fs/2));
treat_modes = find(fc_sorted >= cfg.treat_fc_min_Hz & fc_sorted <= treat_fc_max);
if isempty(treat_modes)
    treat_modes = mask_modes;
end

% Early return exactly in spirit of demo
if R < 1 || nnz(clean_idx) < minBase
    y_hat = y;
    if was_row, y_hat = y_hat.'; end
    debug = pack_debug(cfg, s, yn, y_hat, U, U, fc_sorted, m_common, regions, ...
        [], [], clean_idx, proxy, z, mask_modes, treat_modes);
    return;
end

% -------------------- Step 8: optimize all A(k,r) + build W(k,t) --------------------
A = ones(K,R);
W = ones(K,T);

for r = 1:R
    sR = regions(r,1);
    eR = regions(r,2);

    mr_rect = zeros(1,T);
    mr_rect(sR:eR) = 1;

    for kk = treat_modes(:).'
        % Demo-style dilation helper
        dil_sec = get_dil_sec(fc_sorted(kk), cfg);

        mrA = dilate_binary(mr_rect, round(dil_sec*Fs));
        mrA = smoothdata(double(mrA), 'movmean', max(1, round(cfg.mask_smooth_sec*Fs)));
        mrA = min(max(mrA,0),1);

        burst_idx = (mrA > 0.5);
        if nnz(burst_idx) < minL
            continue;
        end

        % Mode-specific shape gating
        gk = abs(hilbert(U(kk,:)));
        gk = (gk - median(gk)) / (mad(gk,1) + 1e-12);
        gk = min(max(gk / cfg.mask_zthr, 0), cfg.shape_clip_hi);
        gk = smoothdata(gk, 'movmean', max(1, round(cfg.shape_smooth_sec*Fs)));
        gk = min(max(gk,0),1) .^ cfg.shape_power;

        mrC = mrA .* gk;

        % IMPORTANT: demo-style protection only
        mrC(clean_idx) = 0;

        % optimize scalar a in [a_min, 1]
        f = @(a) obj_energy(a, U(kk,:), mrC, clean_idx, burst_idx, cfg);
        A(kk,r) = fminbnd(f, cfg.a_min, 1.0);

        % gamma-shaped time-varying gain
        a = A(kk,r);
        if cfg.use_gamma_gain
            gamma = min(1, cfg.eta * a / (1 - a + 1e-12));
            W(kk,:) = W(kk,:) .* (1 - (1 - a) * (mrC .^ gamma));
        else
            W(kk,:) = W(kk,:) .* (1 - mrC .* (1 - a));
        end
    end
end

W = min(max(W,0),1);

% optional pushToe
if cfg.use_pushToe
    W = pushToe(W, cfg.pushToe_xp, cfg.pushToe_k);
    W = min(max(W,0),1);
end

% -------------------- reconstruct --------------------
U_att = U .* W;
yhat_n = sum(U_att,1).';
y_hat = yhat_n * s;

if was_row
    y_hat = y_hat.';
end

% -------------------- debug --------------------
debug = pack_debug(cfg, s, yn, y_hat, U, U_att, fc_sorted, m_common, regions, ...
    A, W, clean_idx, proxy, z, mask_modes, treat_modes);

end

%% ===================== local helpers =====================

function cfg = fill_defaults(cfg)
d = struct();

% ---------- normalize ----------
d.do_normalize   = true;
d.scale_method   = "mad";
d.scale_floor    = 1e-6;

% ---------- VMD hyperparams ----------
d.NumIMFs        = 8;
d.PenaltyFactor  = 1500;
d.MaxIterations  = 500;
d.InitMethod     = "peaks";   % deterministic

% ---------- proxy + common mask ----------
d.mask_fc_min_Hz   = 15;
d.mask_fc_max_Hz   = 256;
d.env_smooth_sec   = 0.03;
d.mask_zthr        = 3.0;
d.mask_smooth_sec  = 0.04;

% ---------- regions / baseline ----------
d.baseline_m_thr   = 0.10;
d.min_baseline_sec = 0.25;
d.min_burst_sec    = 0.06;

% ---------- treat modes ----------
d.treat_fc_min_Hz  = 5;
d.treat_fc_max_Hz  = 256;

% ---------- A: region blob policy ----------
d.mask_dilate_sec_base = 0.14;

% Demo patch: constant dilation by default
d.use_freq_adaptive_dilation = false;
d.fc_ref_Hz      = 30;
d.dilate_scale_p = 1.0;
d.dilate_min_sec = 0.03;
d.dilate_max_sec = 0.30;

% ---------- C: shape gating ----------
d.shape_smooth_sec = 0.02;
d.shape_power      = 1.0;
d.shape_clip_hi    = 1.0;

% ---------- optimization ----------
d.a_min          = 0.01;
d.lambda_reg     = 1e-3;
d.objective_mode = "logratio";

% ---------- gamma-shaped gain ----------
d.use_gamma_gain = true;
d.eta            = 0.9;

% ---------- optional pushToe ----------
d.use_pushToe  = false;
d.pushToe_xp   = 0.35;
d.pushToe_k    = 2;

% merge user cfg on top
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

function dil_sec = get_dil_sec(fc_hz, cfg)
% Demo-matched helper
if isfield(cfg,'use_freq_adaptive_dilation') && cfg.use_freq_adaptive_dilation
    f0 = max(fc_hz, 0.5);
    dil_sec = cfg.mask_dilate_sec_base * ...
        (cfg.fc_ref_Hz / max(f0, cfg.fc_ref_Hz))^cfg.dilate_scale_p;
    dil_sec = min(max(dil_sec, cfg.dilate_min_sec), cfg.dilate_max_sec);
else
    dil_sec = cfg.mask_dilate_sec_base;
end
end

function J = obj_energy(a, ck, mrk, clean_idx, burst_idx, cfg)
% Demo-matched scale-invariant burst/baseline energy ratio objective
gain = compute_gain(a, mrk, cfg);
x = ck(:).' .* gain;

Ebase  = mean(x(clean_idx).^2, 'omitnan');
Eburst = mean(x(burst_idx).^2, 'omitnan');

epsE = 1e-12;
switch lower(string(cfg.objective_mode))
    case "ratio"
        Jmain = ((Eburst/(Ebase+epsE)) - 1)^2;
    otherwise
        Jmain = (log((Eburst+epsE)/(Ebase+epsE)))^2;
end

J = Jmain + cfg.lambda_reg*(1 - a)^2;
end

function gain = compute_gain(a, m, cfg)
if isfield(cfg,'use_gamma_gain') && cfg.use_gamma_gain
    gamma = min(1, cfg.eta * a / (1 - a + 1e-12));
    gain  = 1 - (1 - a) * (m .^ gamma);
else
    gain  = 1 - (1 - a) * m;
end
gain = min(max(gain, a), 1);
end

function z = robust_zscore(x)
x = x(:).';
z = (x - median(x)) / (mad(x,1) + 1e-12);
end

function Wout = pushToe(W, xp, k)
Wout = (W < xp) .* (xp .* (W./xp).^k) + (W >= xp) .* W;
end

function regions = find_regions(bin_mask)
b = bin_mask(:).';
db = diff([0 b 0]);
starts = find(db==1);
ends   = find(db==-1) - 1;
regions = [starts(:) ends(:)];
end

function m = dilate_binary(m0, halfwin)
m0 = double(m0(:).');
halfwin = max(0, halfwin);
if halfwin == 0
    m = m0;
    return;
end
ker = ones(1, 2*halfwin+1);
m = conv(m0, ker, 'same') > 0;
m = double(m);
end

function fc = spectral_centroid_rows(X, Fs)
[K,T] = size(X);
Nfft = max(256, 2^nextpow2(T));
f = (0:(Nfft/2))*(Fs/Nfft);
fc = zeros(K,1);
for k = 1:K
    x = X(k,:);
    Xf = fft(x, Nfft);
    P  = abs(Xf(1:Nfft/2+1)).^2;
    fc(k) = (f(:)'*P(:)) / (sum(P)+1e-12);
end
end

function debug = pack_debug(cfg, s, yn, y_hat, U, U_att, fc_sorted, ...
    m_common, regions, A, W, clean_idx, proxy, z, mask_modes, treat_modes)

debug = struct();
debug.cfg         = cfg;
debug.scale       = s;
debug.yn          = yn;
debug.y_hat       = y_hat;
debug.U           = U;
debug.U_att       = U_att;
debug.fc          = fc_sorted;
debug.m_common    = m_common;
debug.regions     = regions;
debug.A           = A;
debug.W           = W;
debug.clean_idx   = clean_idx;
debug.proxy       = proxy;
debug.z           = z;
debug.mask_modes  = mask_modes;
debug.treat_modes = treat_modes;
end