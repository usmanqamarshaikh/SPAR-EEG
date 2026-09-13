function run_bci_fp12_denoise_branches(inputH5, outputH5)
%RUN_BCI_FP12_DENOISE_BRANCHES Apply proposed denoising branches to FP1/FP2.
%
% Input H5 is produced by scripts/preprocess_fp12_one_run.py and contains
% /wide/data as [channels x samples] after identical wide conditioning.

    if nargin < 2
        error('Usage: run_bci_fp12_denoise_branches(inputH5, outputH5)');
    end

    thisFile = mfilename('fullpath');
    projectRoot = fullfile(fileparts(thisFile), '..', '..', '..');
    candidateRoots = { ...
        getenv('BURSTDENOISE_ROOT'), ...
        fullfile(getenv('USERPROFILE'), 'Documents', 'MATLAB', 'plugins', 'burstdenoiseMethod'), ...
        fullfile(projectRoot, 'matlab') ...
    };

    for r = 1:numel(candidateRoots)
        rootPath = candidateRoots{r};
        if ~isempty(rootPath) && exist(rootPath, 'dir') == 7
            addpath(rootPath);
        end
    end
    rehash;

    assert(has_pass('emg'), 'EMG denoiser not found on MATLAB path.');
    assert(has_pass('eog'), 'EOG denoiser not found on MATLAB path.');
    assert(has_pass('slow'), 'Slow denoiser not found on MATLAB path.');

    X = double(h5read(inputH5, '/wide/data'));
    if size(X, 1) > size(X, 2) && size(X, 2) <= 64
        X = X.'; % Python H5 [channels x samples] is read by MATLAB as [samples x channels].
    end
    fs = double(h5read(inputH5, '/wide/srate'));

    if exist(outputH5, 'file') == 2
        delete(outputH5);
    end

    write_branch(outputH5, '/baseline/data', single(X));
    h5writeatt(outputH5, '/', 'input_h5', char(inputH5));
    h5writeatt(outputH5, '/', 'wide_filter_low_hz', h5read(inputH5, '/wide/filter_low_hz'));
    h5writeatt(outputH5, '/', 'wide_filter_high_hz', h5read(inputH5, '/wide/filter_high_hz'));
    if has_h5_dataset(inputH5, '/wide/notch_freqs_hz')
        h5writeatt(outputH5, '/', 'notch_freqs_hz', h5read(inputH5, '/wide/notch_freqs_hz'));
    end
    h5writeatt(outputH5, '/', 'srate', fs);
    h5writeatt(outputH5, '/baseline', 'description', 'Wide-conditioned FP1/FP2 without proposed denoising');

    % EOG-only branch is independent because it asks a different scientific
    % question from the hierarchical EMG->EOG->slow pipeline.
    fprintf('\n[branch:eog] pass order: eog\n');
    Yeog = run_adaptive_matrix(X, fs, {'eog'}, 'eog');
    write_branch(outputH5, '/eog/data', single(Yeog));
    h5writeatt(outputH5, '/eog', 'pass_order', 'eog');

    % Hierarchical branch: compute EMG->EOG once, then apply slow to that
    % intermediate to form the full branch.
    fprintf('\n[branch:emg_eog] pass order: emg -> eog\n');
    YemgEog = run_adaptive_matrix(X, fs, {'emg', 'eog'}, 'emg_eog');
    write_branch(outputH5, '/emg_eog/data', single(YemgEog));
    h5writeatt(outputH5, '/emg_eog', 'pass_order', 'emg,eog');

    fprintf('\n[branch:full] pass order: slow applied to saved EMG->EOG intermediate\n');
    Yfull = run_adaptive_matrix(YemgEog, fs, {'slow'}, 'full');
    write_branch(outputH5, '/full/data', single(Yfull));
    h5writeatt(outputH5, '/full', 'pass_order', 'emg,eog,slow');

    branchNames = {'eog', 'emg_eog', 'full'};
    for b = 1:numel(branchNames)
        h5writeatt(outputH5, ['/' branchNames{b}], 'adaptive_min_len_sec', 2.0);
        h5writeatt(outputH5, ['/' branchNames{b}], 'adaptive_max_len_sec', 4.0);
        h5writeatt(outputH5, ['/' branchNames{b}], 'adaptive_search_radius_sec', 2.0);
        h5writeatt(outputH5, ['/' branchNames{b}], 'adaptive_extend_step_sec', 1.0);
    end

    fprintf('\nSaved denoising branches:\n%s\n', outputH5);
end

function Y = run_adaptive_matrix(X, fs, passOrder, branchName)
    cfg = adaptive_cfg(passOrder);
    Y = zeros(size(X));
    nCh = size(X, 1);
    for ch = 1:nCh
        fprintf('[branch:%s] channel %d/%d\n', branchName, ch, nCh);
        Y(ch, :) = adaptive_stream_denoise_singlecut_configured(X(ch, :).', fs, cfg).';
    end
end

function tf = has_h5_dataset(h5Path, datasetPath)
    tf = false;
    try
        h5info(h5Path, datasetPath);
        tf = true;
    catch
        tf = false;
    end
end

function write_branch(outputH5, datasetPath, data)
    dataToWrite = data.';
    h5create(outputH5, datasetPath, size(dataToWrite), 'Datatype', 'single');
    h5write(outputH5, datasetPath, dataToWrite);
end

function cfg = adaptive_cfg(passOrder)
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
    cfg.pre_bandpass      = false;
    cfg.min_denoise_len_sec = 0.20;
    cfg.pass_order = passOrder;
end

function [y_out, meta] = adaptive_stream_denoise_singlecut_configured(x_in, fs, cfg)
    fs = double(fs);
    x = double(x_in(:));
    N = numel(x);
    t = (0:N-1)' / fs;

    [xn, normInfo] = robust_normalize_for_scoring(x);
    score_raw = compute_boundary_score(xn, fs, cfg);
    smooth_w = max(1, round(cfg.smooth_score_sec * fs));
    if smooth_w > 1
        score_disp = movmean(score_raw, smooth_w);
    else
        score_disp = score_raw;
    end

    switch lower(cfg.accept_cost_mode)
        case 'percentile'
            accept_cost = prctile(score_raw, cfg.accept_percentile);
        case 'fixed'
            accept_cost = cfg.accept_cost_fixed;
        otherwise
            error('Unknown cfg.accept_cost_mode: %s', cfg.accept_cost_mode);
    end

    cfg_local = cfg;
    cfg_local.accept_cost = accept_cost;
    epochs = adaptive_epoch_signal_from_score(score_raw, fs, cfg_local);

    nSeg = numel(epochs);
    assert(nSeg >= 1, 'No segments generated.');

    segStart = arrayfun(@(s) s.idx_start, epochs);
    segEnd = arrayfun(@(s) s.idx_end, epochs);
    segDur = (segEnd - segStart + 1) / fs;

    y_out = zeros(N, 1);
    y_out(segStart(1):segEnd(1)) = x(segStart(1):segEnd(1));
    minDenoiseLen = max(4, round(cfg.min_denoise_len_sec * fs));

    for j = 2:nSeg
        right_idx = (segEnd(j-1)+1):segEnd(j);
        y_right_raw = double(x(right_idx));
        if numel(y_right_raw) >= minDenoiseLen
            y_out(right_idx) = run_pass_order(y_right_raw, fs, cfg.pass_order);
        else
            y_out(right_idx) = y_right_raw;
        end
    end

    if nSeg == 1
        warning('Only one segment generated; denoising whole channel.');
        if N >= minDenoiseLen
            y_out = run_pass_order(x, fs, cfg.pass_order);
        else
            y_out = x;
        end
    end

    meta = struct();
    meta.fs = fs;
    meta.t = t;
    meta.cfg = cfg_local;
    meta.normInfo = normInfo;
    meta.score_raw = score_raw;
    meta.score_disp = score_disp;
    meta.segment_start = segStart;
    meta.segment_end = segEnd;
    meta.segment_dur = segDur;
end

function y = run_pass_order(x, fs, pass_order)
    fs = double(fs);
    y = double(x(:));
    for p = 1:numel(pass_order)
        thisPass = lower(char(pass_order{p}));
        switch thisPass
            case 'emg'
                y = call_pass('emg', double(y), fs);
            case 'eog'
                y = call_pass('eog', double(y), fs);
            case 'slow'
                y = call_pass('slow', double(y), fs);
            otherwise
                error('Unknown denoiser pass: %s', thisPass);
        end
        y = double(y(:));
    end
end

function tf = has_pass(kind)
    switch lower(kind)
        case 'emg'
            tf = ~isempty(which('burstdenoise.denoise_emg_vmd_window')) || ...
                 exist('denoise_emg_vmd_window', 'file') == 2;
        case 'eog'
            tf = ~isempty(which('burstdenoise.denoise_eog_ssa_window')) || ...
                 exist('denoise_eog_ssa_window', 'file') == 2;
        case 'slow'
            tf = ~isempty(which('burstdenoise.denoise_slow_ssa_window')) || ...
                 exist('denoise_slow_ssa_window', 'file') == 2;
        otherwise
            tf = false;
    end
end

function y = call_pass(kind, x, fs)
    switch lower(kind)
        case 'emg'
            if ~isempty(which('burstdenoise.denoise_emg_vmd_window'))
                y = burstdenoise.denoise_emg_vmd_window(x, fs);
            else
                y = denoise_emg_vmd_window(x, fs);
            end
        case 'eog'
            if ~isempty(which('burstdenoise.denoise_eog_ssa_window'))
                y = burstdenoise.denoise_eog_ssa_window(x, fs);
            else
                y = denoise_eog_ssa_window(x, fs);
            end
        case 'slow'
            if ~isempty(which('burstdenoise.denoise_slow_ssa_window'))
                y = burstdenoise.denoise_slow_ssa_window(x, fs);
            else
                y = denoise_slow_ssa_window(x, fs);
            end
        otherwise
            error('Unknown pass: %s', kind);
    end
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
        'cut_score', {}, 'accepted_direct', {}, 'samples', {});

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
            epochs(ep).samples = s:N;
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
        epochs(ep).samples = s:best_e;
        s = best_e + 1;
    end
end

function [best_idx, best_cost] = find_best_boundary_in_range(score_raw, lo, hi)
    score_raw = double(score_raw(:));
    [best_cost, relIdx] = min(score_raw(lo:hi));
    best_idx = lo + relIdx - 1;
end
