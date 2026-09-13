function [y, meta] = spar_eeg_adaptive(x, Fs, passes, cfg)
%SPAR_EEG_ADAPTIVE Apply SPAR-EEG to adaptive 2-4 s signal segments.
%
% This wrapper reproduces the adaptive boundary policy used for continuous
% recordings in the reported experiments. Boundaries favor locally stable
% samples based on amplitude, slope, local RMS, and zero crossing terms.

if nargin < 3 || isempty(passes)
    passes = {'emg', 'eog', 'slow'};
end
if nargin < 4 || isempty(cfg)
    cfg = struct();
end

validateattributes(x, {'numeric'}, {'vector', 'real', 'finite', 'nonempty'}, mfilename, 'x');
validateattributes(Fs, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, mfilename, 'Fs');
if ischar(passes) || isstring(passes)
    passes = cellstr(passes);
end

cfg = fill_stream_defaults(cfg, passes);
wasRow = isrow(x);
x = double(x(:));
N = numel(x);

[xn, normInfo] = robust_normalize_for_scoring(x);
scoreRaw = compute_boundary_score(xn, Fs, cfg);
smoothW = max(1, round(cfg.smooth_score_sec * Fs));
scoreDisplay = movmean(scoreRaw, smoothW);

switch lower(cfg.accept_cost_mode)
    case 'percentile'
        acceptCost = prctile(scoreRaw, cfg.accept_percentile);
    case 'fixed'
        acceptCost = cfg.accept_cost_fixed;
    otherwise
        error('spar_eeg_adaptive:UnknownCostMode', ...
            'Unknown accept_cost_mode: %s', cfg.accept_cost_mode);
end

epochs = find_adaptive_epochs(scoreRaw, Fs, cfg, acceptCost);
segmentStart = arrayfun(@(s) s.idx_start, epochs);
segmentEnd = arrayfun(@(s) s.idx_end, epochs);
segmentDuration = (segmentEnd - segmentStart + 1) / Fs;

y = zeros(N, 1);
y(segmentStart(1):segmentEnd(1)) = x(segmentStart(1):segmentEnd(1));
minDenoiseLength = max(4, round(cfg.min_denoise_len_sec * Fs));

for k = 2:numel(epochs)
    idx = (segmentEnd(k - 1) + 1):segmentEnd(k);
    if numel(idx) >= minDenoiseLength
        y(idx) = spar_eeg_denoise(x(idx), Fs, passes, cfg.pass_cfg);
    else
        y(idx) = x(idx);
    end
end

if numel(epochs) == 1
    if N >= minDenoiseLength
        y = spar_eeg_denoise(x, Fs, passes, cfg.pass_cfg);
    else
        y = x;
    end
end

meta = struct();
meta.fs = Fs;
meta.cfg = cfg;
meta.accept_cost = acceptCost;
meta.normalization = normInfo;
meta.score_raw = scoreRaw;
meta.score_display = scoreDisplay;
meta.segment_start = segmentStart;
meta.segment_end = segmentEnd;
meta.segment_duration_sec = segmentDuration;

if wasRow
    y = y.';
end
end

function cfg = fill_stream_defaults(cfg, passes)
defaults = struct();
defaults.min_len_sec = 2.0;
defaults.max_len_sec = 4.0;
defaults.search_radius_sec = 2.0;
defaults.extend_step_sec = 1.0;
defaults.rms_win_sec = 0.10;
defaults.smooth_score_sec = 0.03;
defaults.amp_weight = 1.0;
defaults.slope_weight = 0.5;
defaults.rms_weight = 2.0;
defaults.zc_bonus = 0.15;
defaults.accept_cost_mode = 'percentile';
defaults.accept_percentile = 10;
defaults.accept_cost_fixed = 10.0;
defaults.min_denoise_len_sec = 0.20;
defaults.pass_cfg = struct();
defaults.pass_order = passes;

names = fieldnames(defaults);
for k = 1:numel(names)
    if ~isfield(cfg, names{k}) || isempty(cfg.(names{k}))
        cfg.(names{k}) = defaults.(names{k});
    end
end
end

function [xn, info] = robust_normalize_for_scoring(x)
center = median(x, 'omitnan');
madValue = median(abs(x - center), 'omitnan');
if madValue < eps
    scale = std(x);
    if scale < eps, scale = 1; end
    center = mean(x, 'omitnan');
    info.method = 'std';
else
    scale = 1.4826 * madValue;
    info.method = 'mad';
end
xn = (x - center) / scale;
info.center = center;
info.scale = scale;
end

function score = compute_boundary_score(x, Fs, cfg)
rmsWindow = max(3, round(cfg.rms_win_sec * Fs));
slope = [0; diff(x)];
localRms = sqrt(movmean(x.^2, rmsWindow, 'Endpoints', 'shrink'));
zeroCrossing = false(numel(x), 1);
zeroCrossing(2:end) = sign(x(2:end)) ~= sign(x(1:end - 1));
score = cfg.amp_weight * abs(x) + ...
    cfg.slope_weight * abs(slope) + ...
    cfg.rms_weight * localRms - ...
    cfg.zc_bonus * double(zeroCrossing);
end

function epochs = find_adaptive_epochs(score, Fs, cfg, acceptCost)
N = numel(score);
minLength = max(1, round(cfg.min_len_sec * Fs));
maxLength = max(minLength, round(cfg.max_len_sec * Fs));
searchRadius = max(1, round(cfg.search_radius_sec * Fs));
extendStep = max(1, round(cfg.extend_step_sec * Fs));
epochs = struct('idx_start', {}, 'idx_end', {}, 'nominal_end', {}, ...
    'cut_score', {}, 'accepted_direct', {});

startIndex = 1;
epochIndex = 0;
while startIndex <= N
    if startIndex + minLength - 1 > N
        epochIndex = epochIndex + 1;
        epochs(epochIndex) = make_epoch(startIndex, N, N, score(N), true);
        break;
    end

    nominalEnd = startIndex + minLength - 1;
    candidateEnd = nominalEnd;
    candidateCost = inf;
    acceptedDirect = false;
    currentNominal = nominalEnd;
    hardLimit = min(N, startIndex + maxLength - 1);

    while currentNominal <= hardLimit
        lo = max(startIndex + minLength - 1, currentNominal - searchRadius);
        hi = min(hardLimit, currentNominal + searchRadius);
        [trialEnd, trialCost] = best_boundary(score, lo, hi);
        if trialCost < candidateCost
            candidateEnd = trialEnd;
            candidateCost = trialCost;
        end
        if trialCost <= acceptCost
            acceptedDirect = true;
            break;
        end
        currentNominal = currentNominal + extendStep;
    end

    epochIndex = epochIndex + 1;
    epochs(epochIndex) = make_epoch(startIndex, candidateEnd, nominalEnd, ...
        candidateCost, acceptedDirect);
    startIndex = candidateEnd + 1;
end
end

function epoch = make_epoch(startIndex, endIndex, nominalEnd, cost, accepted)
epoch = struct('idx_start', startIndex, 'idx_end', endIndex, ...
    'nominal_end', nominalEnd, 'cut_score', cost, ...
    'accepted_direct', accepted);
end

function [index, cost] = best_boundary(score, lo, hi)
[cost, relativeIndex] = min(score(lo:hi));
index = lo + relativeIndex - 1;
end
