function run_proposed_h5(dataDir, outDir, pattern, maxRecords, passPolicy, overwrite, packageParent)
%RUN_PROPOSED_H5 Shape-preserving H5 runner for the proposed MATLAB denoiser.
%
% MATLAB reads the project H5 arrays as samples x channels. Python/h5py sees
% the same datasets as channels x samples. This runner intentionally preserves
% the MATLAB-side shape when writing restored outputs, so Python sees restored
% arrays with the same shape as the original input arrays.
%
% Example:
%   run_proposed_h5('data', 'results/proposed_smoke', ...
%       '03_denoise-net_eog_-20dB.h5', 1, 'auto', true, ...
%       fullfile('..', '..', '..', 'matlab'))

if nargin < 1 || isempty(dataDir), dataDir = 'data'; end
if nargin < 2 || isempty(outDir), outDir = fullfile('results', 'proposed_restored'); end
if nargin < 3 || isempty(pattern), pattern = '*.h5'; end
if nargin < 4 || isempty(maxRecords), maxRecords = Inf; end
if nargin < 5 || isempty(passPolicy), passPolicy = 'auto'; end
if nargin < 6 || isempty(overwrite), overwrite = false; end
if nargin < 7 || isempty(packageParent)
    thisDir = fileparts(mfilename('fullpath'));
    packageParent = fullfile(thisDir, '..', '..', '..', 'matlab');
end

dataDir = char(dataDir);
outDir = char(outDir);
pattern = char(pattern);
passPolicy = char(passPolicy);
packageParent = char(packageParent);

if ischar(maxRecords) || isstring(maxRecords)
    maxRecords = str2double(maxRecords);
end
if ischar(overwrite) || isstring(overwrite)
    overwrite = parse_bool(overwrite);
end

if ~exist(outDir, 'dir')
    mkdir(outDir);
end

addpath(packageParent);

files = dir(fullfile(dataDir, pattern));
if isempty(files)
    error('No H5 files found for pattern: %s', fullfile(dataDir, pattern));
end

fprintf('Proposed MATLAB H5 runner\n');
fprintf('  dataDir      : %s\n', dataDir);
fprintf('  outDir       : %s\n', outDir);
fprintf('  pattern      : %s\n', pattern);
fprintf('  maxRecords   : %g\n', maxRecords);
fprintf('  passPolicy   : %s\n', passPolicy);
fprintf('  overwrite    : %d\n', overwrite);
fprintf('  packageParent: %s\n', packageParent);

for f = 1:numel(files)
    inFile = fullfile(files(f).folder, files(f).name);
    finalOutFile = fullfile(outDir, files(f).name);

    if exist(finalOutFile, 'file')
        if overwrite
            fprintf('Overwriting existing output after temp-file completion: %s\n', finalOutFile);
        else
            fprintf('Skipping existing output: %s\n', finalOutFile);
            continue;
        end
    end

    tempOutDir = fullfile(tempdir, 'burstdenoise_eval_h5');
    if ~exist(tempOutDir, 'dir')
        mkdir(tempOutDir);
    end
    outFile = [tempname(tempOutDir) '.h5'];

    info = h5info(inFile);
    nRec = numel(info.Groups);
    nUse = min(nRec, maxRecords);

    passOrder = get_pass_order(files(f).name, passPolicy);
    processingMode = get_processing_mode(files(f).name);
    fprintf('\n[%d/%d] %s | records=%d/%d | mode=%s | passes=%s\n', ...
        f, numel(files), files(f).name, nUse, nRec, processingMode, strjoin(passOrder, '+'));

    h5writeatt_safe(outFile, '/', 'source_file', files(f).name);
    h5writeatt_safe(outFile, '/', 'method', ['proposed_' strjoin(passOrder, '+')]);
    h5writeatt_safe(outFile, '/', 'pass_policy', passPolicy);
    h5writeatt_safe(outFile, '/', 'processing_mode', processingMode);

    for r = 1:nUse
        groupName = info.Groups(r).Name;       % e.g., /eog_0
        recordName = erase(groupName, '/');
        dsIn = [groupName '/eeg_signal'];
        dsOut = ['/' recordName];

        try
            X = double(h5read(inFile, dsIn));  % MATLAB orientation: samples x channels
            if isvector(X)
                X = X(:);
            end

            fs = read_freq(inFile, groupName);
            switch processingMode
                case 'adaptive_stream'
                    Y = adaptive_denoise_matrix_columns(X, fs, passOrder, default_stream_cfg());
                case 'whole_record'
                    Y = denoise_matrix_columns(X, fs, passOrder);
                otherwise
                    error('Unknown processing mode: %s', processingMode);
            end

            h5create(outFile, dsOut, size(Y), 'Datatype', 'double');
            h5write(outFile, dsOut, Y);
            h5writeatt_safe(outFile, dsOut, 'fs', fs);
            h5writeatt_safe(outFile, dsOut, 'method', ['proposed_' strjoin(passOrder, '+')]);
            h5writeatt_safe(outFile, dsOut, 'pass_order', strjoin(passOrder, '+'));
            h5writeatt_safe(outFile, dsOut, 'processing_mode', processingMode);
            h5writeatt_safe(outFile, dsOut, 'status', 'OK');

            fprintf('  [%d/%d] %s OK | size=%s | fs=%g\n', ...
                r, nUse, recordName, mat2str(size(Y)), fs);

        catch ME
            warning('Record failed: %s | %s', recordName, ME.message);
            try
                X = double(h5read(inFile, dsIn));
                if isvector(X), X = X(:); end
                h5create(outFile, dsOut, size(X), 'Datatype', 'double');
                h5write(outFile, dsOut, X);
                h5writeatt_safe(outFile, dsOut, 'status', ['FAIL: ' ME.message]);
            catch ME2
                warning('Could not write fallback for %s | %s', recordName, ME2.message);
            end
        end
    end

    if exist(finalOutFile, 'file')
        delete(finalOutFile);
    end
    [copyOk, copyMsg] = copyfile(outFile, finalOutFile, 'f');
    if ~copyOk
        error('Could not copy completed H5 to %s | %s', finalOutFile, copyMsg);
    end
    delete(outFile);
    fprintf('  wrote completed file: %s\n', finalOutFile);
end

fprintf('\nProposed MATLAB H5 runner finished.\n');
end

function Y = adaptive_denoise_matrix_columns(X, fs, passOrder, cfg)
X = double(X);
[nSamples, nChannels] = size(X);
Y = zeros(nSamples, nChannels);

for ch = 1:nChannels
    x = X(:, ch);
    if all(~isfinite(x)) || numel(x) < 4
        Y(:, ch) = x;
        continue;
    end

    [xn, ~] = robust_normalize_for_scoring(x);
    score_raw = compute_boundary_score(xn, fs, cfg);

    switch lower(cfg.accept_cost_mode)
        case 'percentile'
            cfg.accept_cost = prctile(score_raw, cfg.accept_percentile);
        case 'fixed'
            cfg.accept_cost = cfg.accept_cost_fixed;
        otherwise
            error('Unknown accept_cost_mode: %s', cfg.accept_cost_mode);
    end

    epochs = adaptive_epoch_signal_from_score(score_raw, fs, cfg);
    y = zeros(size(x));
    minDenoiseLen = max(4, round(cfg.min_denoise_len_sec * fs));

    for e = 1:numel(epochs)
        idx = epochs(e).idx_start:epochs(e).idx_end;
        seg = double(x(idx));
        if numel(seg) >= minDenoiseLen
            y(idx) = run_pass_order(seg, fs, passOrder);
        else
            y(idx) = seg;
        end
    end

    Y(:, ch) = y;
end
end

function Y = denoise_matrix_columns(X, fs, passOrder)
X = double(X);
[nSamples, nChannels] = size(X);
Y = zeros(nSamples, nChannels);

for ch = 1:nChannels
    y = X(:, ch);
    if all(~isfinite(y)) || numel(y) < 4
        Y(:, ch) = y;
        continue;
    end

    for p = 1:numel(passOrder)
        switch lower(passOrder{p})
            case 'emg'
                y = burstdenoise.denoise_emg_vmd_window(double(y), double(fs));
            case 'eog'
                y = burstdenoise.denoise_eog_ssa_window(double(y), double(fs));
            case 'slow'
                y = burstdenoise.denoise_slow_ssa_window(double(y), double(fs));
            otherwise
                error('Unknown pass: %s', passOrder{p});
        end
        y = double(y(:));
    end
    Y(:, ch) = y;
end
end

function y = run_pass_order(x, fs, passOrder)
y = double(x(:));
for p = 1:numel(passOrder)
    switch lower(passOrder{p})
        case 'emg'
            y = burstdenoise.denoise_emg_vmd_window(double(y), double(fs));
        case 'eog'
            y = burstdenoise.denoise_eog_ssa_window(double(y), double(fs));
        case 'slow'
            y = burstdenoise.denoise_slow_ssa_window(double(y), double(fs));
        otherwise
            error('Unknown pass: %s', passOrder{p});
    end
    y = double(y(:));
end
end

function passOrder = get_pass_order(fileName, passPolicy)
policy = lower(strtrim(passPolicy));
fname = lower(fileName);

switch policy
    case 'auto'
        if contains(fname, '01_physiobank')
            passOrder = {'slow'};
        elseif contains(fname, '02_semisimulated_eog')
            passOrder = {'eog'};
        elseif contains(fname, '03_denoise-net_eog+emg')
            passOrder = {'emg', 'eog', 'slow'};
        elseif contains(fname, '03_denoise-net_emg')
            passOrder = {'emg', 'eog', 'slow'};
        elseif contains(fname, '03_denoise-net_eog')
            passOrder = {'emg', 'eog', 'slow'};
        else
            passOrder = {'emg', 'eog', 'slow'};
        end
    case 'full'
        passOrder = {'emg', 'eog', 'slow'};
    case 'emg'
        passOrder = {'emg'};
    case 'eog'
        passOrder = {'eog'};
    case 'slow'
        passOrder = {'slow'};
    case {'emg+eog', 'emg,eog'}
        passOrder = {'emg', 'eog'};
    case {'eog+slow', 'eog,slow'}
        passOrder = {'eog', 'slow'};
    case {'emg+eog+slow', 'emg,eog,slow'}
        passOrder = {'emg', 'eog', 'slow'};
    otherwise
        error('Unknown passPolicy: %s', passPolicy);
end
end

function processingMode = get_processing_mode(fileName)
fname = lower(fileName);
if contains(fname, '01_physiobank') || contains(fname, '02_semisimulated_eog')
    processingMode = 'adaptive_stream';
else
    processingMode = 'whole_record';
end
end

function cfg = default_stream_cfg()
cfg = struct();
cfg.min_len_sec       = 2.0;
cfg.max_len_sec       = 4.0;
cfg.search_radius_sec = 2.0;
cfg.extend_step_sec   = 1.0;
cfg.rms_win_sec       = 0.10;
cfg.smooth_score_sec  = 0.03;
cfg.amp_weight        = 1.0;
cfg.slope_weight      = 0.5;
cfg.rms_weight        = 2.0;
cfg.zc_bonus          = 0.15;
cfg.accept_cost_mode  = 'percentile';
cfg.accept_percentile = 10;
cfg.accept_cost_fixed = 10.0;
cfg.min_denoise_len_sec = 0.20;
end

function [xn, info] = robust_normalize_for_scoring(x)
x = double(x(:));
medx = median(x, 'omitnan');
madx = median(abs(x - medx), 'omitnan');

if madx < eps
    sx = std(x);
    if sx < eps
        sx = 1;
    end
    mu = mean(x, 'omitnan');
    xn = (x - mu) / sx;
    info.method = 'std';
    info.center = mu;
    info.scale = sx;
else
    sc = 1.4826 * madx;
    xn = (x - medx) / sc;
    info.method = 'mad';
    info.center = medx;
    info.scale = sc;
end
xn = double(xn(:));
end

function score_raw = compute_boundary_score(xn, fs, cfg)
fs = double(fs);
xn = double(xn(:));
rms_w = max(3, round(cfg.rms_win_sec * fs));
dx = [0; diff(xn)];
local_rms = sqrt(movmean(xn.^2, rms_w, 'Endpoints', 'shrink'));
zc = false(numel(xn), 1);
zc(2:end) = sign(xn(2:end)) ~= sign(xn(1:end-1));

score_raw = cfg.amp_weight * abs(xn) + ...
            cfg.slope_weight * abs(dx) + ...
            cfg.rms_weight * local_rms - ...
            cfg.zc_bonus * double(zc);
score_raw = double(score_raw(:));
end

function epochs = adaptive_epoch_signal_from_score(score_raw, fs, cfg)
fs = double(fs);
score_raw = double(score_raw(:));
N = numel(score_raw);

min_len = max(1, round(cfg.min_len_sec * fs));
max_len = max(min_len, round(cfg.max_len_sec * fs));
search_r = max(1, round(cfg.search_radius_sec * fs));
extend_st = max(1, round(cfg.extend_step_sec * fs));

epochs = struct('idx_start', {}, 'idx_end', {}, 'nominal_end', {}, ...
                'cut_score', {}, 'accepted_direct', {});

s = 1;
ep = 0;
while s <= N
    if s + min_len - 1 > N
        ep = ep + 1;
        epochs(ep).idx_start = s;
        epochs(ep).idx_end = N;
        epochs(ep).nominal_end = N;
        epochs(ep).cut_score = score_raw(N);
        epochs(ep).accepted_direct = true;
        break;
    end

    nominal_e = s + min_len - 1;
    best_e = nominal_e;
    best_cost = inf;
    accepted_direct = false;
    cur_nominal = nominal_e;
    hard_limit = min(N, s + max_len - 1);

    while cur_nominal <= hard_limit
        lo = max(s + min_len - 1, cur_nominal - search_r);
        hi = min(hard_limit, cur_nominal + search_r);
        [cand_e, cand_cost] = find_best_boundary_in_range(score_raw, lo, hi);

        if cand_cost < best_cost
            best_cost = cand_cost;
            best_e = cand_e;
        end
        if cand_cost <= cfg.accept_cost
            accepted_direct = true;
            break;
        end
        cur_nominal = cur_nominal + extend_st;
    end

    ep = ep + 1;
    epochs(ep).idx_start = s;
    epochs(ep).idx_end = best_e;
    epochs(ep).nominal_end = nominal_e;
    epochs(ep).cut_score = best_cost;
    epochs(ep).accepted_direct = accepted_direct;
    s = best_e + 1;
end
end

function [best_idx, best_cost] = find_best_boundary_in_range(score_raw, lo, hi)
score_raw = double(score_raw(:));
[best_cost, relIdx] = min(score_raw(lo:hi));
best_idx = lo + relIdx - 1;
end

function fs = read_freq(inFile, groupName)
fs = 256;
try
    fs = double(h5readatt(inFile, groupName, 'freq'));
catch
end
end

function tf = parse_bool(x)
s = lower(strtrim(char(x)));
tf = any(strcmp(s, {'1', 'true', 'yes', 'y', 'overwrite'}));
end

function h5writeatt_safe(fileName, location, attrName, attrValue)
try
    if ~exist(fileName, 'file')
        fid = H5F.create(fileName);
        H5F.close(fid);
    end
    h5writeatt(fileName, location, attrName, attrValue);
catch
    % Attribute writing is helpful but not essential for metric evaluation.
end
end
