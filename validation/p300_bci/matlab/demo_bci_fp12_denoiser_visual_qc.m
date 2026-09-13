%% demo_bci_fp12_denoiser_visual_qc.m
% Step-by-step MATLAB visual QC for the Won P300 FP1/FP2 preprocessing.
%
% Run this script section-by-section in MATLAB.
%
% Purpose:
%   1. Inspect raw FP1/FP2 signal and PSD.
%   2. Inspect wide conditioning and notch choices.
%   3. Run proposed denoiser branches on a selected window.
%   4. Visually compare baseline vs EOG vs EMG+EOG vs full.
%
% Notes:
%   - This is a debugging/demo script, not the final batch pipeline.
%   - The final decoder will still apply the official narrow P300 filter
%     after denoising.

clear; clc; close all;

%% =========================
% USER CONFIG
% =========================

thisDir = fileparts(mfilename('fullpath'));
projectRoot = fullfile(thisDir, '..', '..', '..');
datasetRoot = fullfile(projectRoot, 'data', 'Won2022_BIDS');
denoiseLocalRoot = fullfile(projectRoot, 'matlab');
denoisePluginRoot = getenv('BURSTDENOISE_ROOT');

subject = 'sub-002';
taskName = 'P300trainrun1';
runNo = 6;
channelsWanted = {'FP1', 'FP2'};

% Detailed visual window. Change these and rerun from Section "Extract window".
startSec = 40;
windowSec = 12;

% Shared conditioning before proposed denoising.
wideLowHz = 0.5;
wideHighHz = 70;

% Change these interactively after looking at raw/wide PSD.
notchFreqsHz = 50;
notchWidthHz = 2.0;       % total stopband width around each center frequency

% Official-style narrow P300 filter for visual reference only.
p300LowHz = 0.5;
p300HighHz = 10;

outDir = fullfile(projectRoot, 'results', 'p300_bci', 'matlab_visual_qc');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% =========================
% ADD DENOISER PATHS
% =========================

if exist(denoisePluginRoot, 'dir') == 7
    addpath(denoisePluginRoot);
end
if exist(denoiseLocalRoot, 'dir') == 7
    addpath(denoiseLocalRoot);
end
rehash;

fprintf('Denoiser path checks:\n');
fprintf('  package EMG: %d\n', ~isempty(which('burstdenoise.denoise_emg_vmd_window')));
fprintf('  local EMG:   %d | %s\n', exist('denoise_emg_vmd_window', 'file'), which('denoise_emg_vmd_window'));
fprintf('  local EOG:   %d | %s\n', exist('denoise_eog_ssa_window', 'file'), which('denoise_eog_ssa_window'));
fprintf('  local slow:  %d | %s\n', exist('denoise_slow_ssa_window', 'file'), which('denoise_slow_ssa_window'));

%% =========================
% LOAD ONE EEGLAB/BIDS RUN
% =========================

stem = sprintf('%s_task-%s_run-%d', subject, taskName, runNo);
eegDir = fullfile(datasetRoot, subject, 'eeg');
setPath = fullfile(eegDir, [stem '_eeg.set']);
fdtPath = fullfile(eegDir, [stem '_eeg.fdt']);

S = load(setPath, '-mat');

fs = double(S.srate);

% fs = double(1024);


nCh = double(S.nbchan);
nT = double(S.pnts);
labels = string({S.chanlocs.labels});

fid = fopen(fdtPath, 'rb');
assert(fid > 0, 'Could not open .fdt file: %s', fdtPath);
Xall = fread(fid, [nCh nT], 'float32=>double');
fclose(fid);

chIdx = zeros(1, numel(channelsWanted));
for i = 1:numel(channelsWanted)
    hit = find(strcmpi(labels, channelsWanted{i}), 1);
    assert(~isempty(hit), 'Channel not found: %s', channelsWanted{i});
    chIdx(i) = hit;
end

Xraw = Xall(chIdx, :);
t = (0:size(Xraw,2)-1) / fs;

fprintf('\nLoaded %s\n', setPath);
fprintf('Raw selected data: %d channels x %d samples | fs = %.1f Hz | duration = %.2f s\n', ...
    size(Xraw,1), size(Xraw,2), fs, size(Xraw,2)/fs);
fprintf('Selected channels: %s\n', strjoin(channelsWanted, ', '));

% clear Xall;

%% =========================
% RAW TIME SERIES AND PSD
% =========================

plot_time_compare({Xraw}, {'Raw'}, fs, channelsWanted, startSec, windowSec, ...
    sprintf('%s raw FP1/FP2', stem));
saveas(gcf, fullfile(outDir, [stem '_01_raw_timeseries.png']));

plot_psd_compare({Xraw}, {'Raw'}, fs, channelsWanted, 90, notchFreqsHz, ...
    sprintf('%s raw PSD', stem));
saveas(gcf, fullfile(outDir, [stem '_01_raw_psd.png']));

print_top_psd_bins(Xraw, fs, 'raw', 35, 90, 15);

%% =========================
% WIDE BANDPASS ONLY
% =========================

Xwide = zero_phase_bandpass_iir(Xraw, fs, wideLowHz, wideHighHz);

plot_time_compare({Xraw, Xwide}, {'Raw', sprintf('Wide %.1f-%.1f Hz', wideLowHz, wideHighHz)}, ...
    fs, channelsWanted, startSec, windowSec, sprintf('%s raw vs wide bandpass', stem));
saveas(gcf, fullfile(outDir, [stem '_02_raw_vs_wide_timeseries.png']));

plot_psd_compare({Xraw, Xwide}, {'Raw', 'Wide'}, fs, channelsWanted, 90, notchFreqsHz, ...
    sprintf('%s raw vs wide PSD', stem));
saveas(gcf, fullfile(outDir, [stem '_02_raw_vs_wide_psd.png']));

print_top_psd_bins(Xwide, fs, 'wide', 35, 90, 15);

%% =========================
% NOTCH AFTER WIDE BANDPASS
% =========================

XwideNotch = Xwide;
for nf = notchFreqsHz
    XwideNotch = zero_phase_notch_iir(XwideNotch, fs, nf, notchWidthHz);
end

plot_time_compare({Xwide, XwideNotch}, {'Wide', 'Wide + notch'}, ...
    fs, channelsWanted, startSec, windowSec, sprintf('%s wide vs notched', stem));
saveas(gcf, fullfile(outDir, [stem '_03_wide_vs_notch_timeseries.png']));

plot_psd_compare({Xwide, XwideNotch}, {'Wide', 'Wide + notch'}, fs, channelsWanted, 90, notchFreqsHz, ...
    sprintf('%s wide vs notch PSD', stem));
saveas(gcf, fullfile(outDir, [stem '_03_wide_vs_notch_psd.png']));

print_top_psd_bins(XwideNotch, fs, 'wide+notch', 35, 90, 15);

%% =========================
% OPTIONAL NOTCH EXPERIMENT
% =========================
% Change tryNotchFreqsHz and tryNotchWidthHz, then rerun this section.

tryNotchFreqsHz = 50;
tryNotchWidthHz = 4.0;

XwideNotchTry = Xwide;
for nf = tryNotchFreqsHz
    XwideNotchTry = zero_phase_notch_iir(XwideNotchTry, fs, nf, tryNotchWidthHz);
end

plot_psd_compare({Xwide, XwideNotch, XwideNotchTry}, ...
    {'Wide', sprintf('Notch %.1f Hz width', notchWidthHz), sprintf('Try notch %.1f Hz width', tryNotchWidthHz)}, ...
    fs, channelsWanted, 90, tryNotchFreqsHz, sprintf('%s notch-width experiment PSD', stem));
saveas(gcf, fullfile(outDir, [stem '_04_notch_experiment_psd.png']));

plot_time_compare({Xwide, XwideNotch, XwideNotchTry}, ...
    {'Wide', sprintf('Notch %.1f Hz width', notchWidthHz), sprintf('Try notch %.1f Hz width', tryNotchWidthHz)}, ...
    fs, channelsWanted, startSec, windowSec, sprintf('%s notch-width experiment time series', stem));
saveas(gcf, fullfile(outDir, [stem '_04_notch_experiment_timeseries.png']));

print_top_psd_bins(XwideNotchTry, fs, 'wide+try-notch', 35, 90, 15);

%% =========================
% EXTRACT WINDOW FOR FAST DENOISER DEBUGGING
% =========================
% This keeps denoiser debugging fast. Once it looks correct, run full signal.

i1 = max(1, round(startSec * fs) + 1);
i2 = min(size(XwideNotch, 2), i1 + round(windowSec * fs) - 1);
Xdemo = XwideNotch(:, i1:i2);

fprintf('\nDenoiser demo window: %.2f-%.2f s | %d samples\n', ...
    (i1-1)/fs, (i2-1)/fs, size(Xdemo,2));

%% =========================
% RUN PROPOSED DENOISER BRANCHES ON WINDOW
% =========================

tic;
Yeog = run_branch_matrix(Xdemo, fs, {'eog'});
YemgEog = run_branch_matrix(Xdemo, fs, {'emg', 'eog'});
Yfull = run_branch_matrix(Xdemo, fs, {'emg', 'eog', 'slow'});
toc;

fprintf('\nWindow RMS delta vs baseline:\n');
fprintf('  EOG:     %.4f uV\n', rms(Yeog(:) - Xdemo(:)));
fprintf('  EMG+EOG: %.4f uV\n', rms(YemgEog(:) - Xdemo(:)));
fprintf('  Full:    %.4f uV\n', rms(Yfull(:) - Xdemo(:)));

%% =========================
% PLOT DENOISER WINDOW OUTPUTS
% =========================

plot_time_compare({Xdemo, Yeog, YemgEog, Yfull}, ...
    {'Baseline wide+notch', 'EOG', 'EMG+EOG', 'Full'}, ...
    fs, channelsWanted, 0, size(Xdemo,2)/fs, sprintf('%s proposed denoiser window output', stem));
saveas(gcf, fullfile(outDir, [stem '_05_denoiser_window_timeseries.png']));

plot_psd_compare({Xdemo, Yeog, YemgEog, Yfull}, ...
    {'Baseline wide+notch', 'EOG', 'EMG+EOG', 'Full'}, ...
    fs, channelsWanted, 90, notchFreqsHz, sprintf('%s proposed denoiser window PSD', stem));
saveas(gcf, fullfile(outDir, [stem '_05_denoiser_window_psd.png']));

%% =========================
% NARROW P300 FILTER VISUAL REFERENCE
% =========================

Xnarrow = zero_phase_bandpass_iir(XwideNotch, fs, p300LowHz, p300HighHz);
XdemoNarrow = Xnarrow(:, i1:i2);

plot_time_compare({Xdemo, XdemoNarrow}, ...
    {'Wide+notch denoiser input', sprintf('Narrow %.1f-%.1f Hz', p300LowHz, p300HighHz)}, ...
    fs, channelsWanted, 0, size(Xdemo,2)/fs, sprintf('%s narrow P300 filter visual reference', stem));
saveas(gcf, fullfile(outDir, [stem '_06_wide_notch_vs_narrow_timeseries.png']));

plot_psd_compare({XwideNotch, Xnarrow}, {'Wide+notch', 'Narrow P300'}, ...
    fs, channelsWanted, 90, notchFreqsHz, sprintf('%s narrow P300 filter PSD reference', stem));
saveas(gcf, fullfile(outDir, [stem '_06_wide_notch_vs_narrow_psd.png']));

fprintf('\nSaved figures to:\n%s\n', outDir);

%% ========================================================================
% LOCAL FUNCTIONS
% ========================================================================

function Y = zero_phase_bandpass_iir(X, fs, loHz, hiHz)
    fs = double(fs);
    Wn = [loHz hiHz] / (fs/2);
    [b, a] = butter(4, Wn, 'bandpass');
    Y = filtfilt(b, a, double(X).').';
end

function Y = zero_phase_notch_iir(X, fs, centerHz, widthHz)
    fs = double(fs);
    lo = max(0.01, centerHz - widthHz/2);
    hi = min(fs/2 - 0.01, centerHz + widthHz/2);
    Wn = [lo hi] / (fs/2);
    [b, a] = butter(2, Wn, 'stop');
    Y = filtfilt(b, a, double(X).').';
end

function Y = run_branch_matrix(X, fs, passOrder)
    Y = zeros(size(X));
    for ch = 1:size(X,1)
        y = double(X(ch,:).');
        for p = 1:numel(passOrder)
            passName = lower(passOrder{p});
            switch passName
                case 'emg'
                    y = call_denoise_pass('emg', y, fs);
                case 'eog'
                    y = call_denoise_pass('eog', y, fs);
                case 'slow'
                    y = call_denoise_pass('slow', y, fs);
                otherwise
                    error('Unknown pass: %s', passName);
            end
            y = double(y(:));
        end
        Y(ch,:) = y.';
    end
end

function y = call_denoise_pass(kind, x, fs)
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

function plot_time_compare(Xcell, names, fs, chanLabels, startSec, windowSec, figTitle)
    startIdx = max(1, round(startSec * fs) + 1);
    stopIdx = min(size(Xcell{1},2), startIdx + round(windowSec * fs) - 1);
    tt = ((startIdx:stopIdx) - startIdx) / fs;

    figure('Color', 'w', 'Position', [100 100 1250 620]);
    tiledlayout(numel(chanLabels), 1, 'TileSpacing', 'compact', 'Padding', 'compact');

    for ch = 1:numel(chanLabels)
        nexttile;
        hold on;
        allVals = [];
        for k = 1:numel(Xcell)
            y = Xcell{k}(ch, startIdx:stopIdx);
            plot(tt, y, 'LineWidth', 0.9);
            allVals = [allVals; y(:)]; %#ok<AGROW>
        end
        yl = prctile(allVals, [0.5 99.5]);
        pad = max(1, 0.08 * diff(yl));
        ylim([yl(1)-pad yl(2)+pad]);
        grid on;
        ylabel(sprintf('%s (uV)', chanLabels{ch}));
        if ch == 1
            title(figTitle, 'Interpreter', 'none');
            legend(names, 'Location', 'northeast', 'Box', 'off');
        end
        if ch == numel(chanLabels)
            xlabel('Time within window (s)');
        end
    end
end

function plot_psd_compare(Xcell, names, fs, chanLabels, fmax, notchFreqsHz, figTitle)
    figure('Color', 'w', 'Position', [120 120 1250 470]);
    tiledlayout(1, numel(chanLabels), 'TileSpacing', 'compact', 'Padding', 'compact');

    for ch = 1:numel(chanLabels)
        nexttile;
        hold on;
        for k = 1:numel(Xcell)
            x = double(Xcell{k}(ch,:));
            nwin = min(numel(x), 8192);
            [pxx, f] = pwelch(x, hamming(nwin), [], nwin, fs);
            keep = f <= fmax;
            semilogy(f(keep), pxx(keep), 'LineWidth', 1.0);
        end
        for nf = notchFreqsHz
            xline(nf, '--', 'Color', [0.55 0.55 0.55]);
        end
        grid on;
        title(chanLabels{ch});
        xlabel('Frequency (Hz)');
        if ch == 1
            ylabel('PSD (uV^2/Hz)');
            legend(names, 'Location', 'northeast', 'Box', 'off');
        end
    end
    sgtitle(figTitle, 'Interpreter', 'none');
end

function print_top_psd_bins(X, fs, name, loHz, hiHz, topN)
    nwin = min(size(X,2), 8192);
    P = [];
    for ch = 1:size(X,1)
        [pxx, f] = pwelch(double(X(ch,:)), hamming(nwin), [], nwin, fs);
        P(:,ch) = pxx; %#ok<AGROW>
    end
    meanP = mean(P, 2);
    keep = f >= loHz & f <= hiHz;
    fb = f(keep);
    pb = meanP(keep);
    [~, ord] = sort(pb, 'descend');
    ord = ord(1:min(topN, numel(ord)));

    fprintf('\nTop PSD bins for %s between %.1f-%.1f Hz:\n', name, loHz, hiHz);
    for i = 1:numel(ord)
        fprintf('  %8.3f Hz : %.6g\n', fb(ord(i)), pb(ord(i)));
    end
end
