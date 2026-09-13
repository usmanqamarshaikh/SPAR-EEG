function run_clean_input_activation_h5(dataFile, restoredRoot, outDir, maxRecords, overwrite, packageParent)
%RUN_CLEAN_INPUT_ACTIVATION_H5 Collect pass diagnostics on clean EEG epochs.
%
% The five deployed configurations are evaluated through a shared branch:
%   clean -> EMG -> EOG -> slow
%      |       |
%      +-> EOG +-> saved EMG+EOG
%      +-> slow
%
% This avoids recomputing the EMG decomposition for the sequential outputs.
% Existing restored H5 files are treated as immutable references and are used
% to verify that the diagnostic rerun reproduces the submitted outputs.

if nargin < 1 || isempty(dataFile)
    dataFile = fullfile('results', 'clean_preservation', 'data', '04_denoise-net_clean.h5');
end
if nargin < 2 || isempty(restoredRoot)
    restoredRoot = fullfile('results', 'clean_preservation', 'restored');
end
if nargin < 3 || isempty(outDir)
    outDir = fullfile('results', 'clean_preservation', 'activation_diagnostics');
end
if nargin < 4 || isempty(maxRecords), maxRecords = Inf; end
if nargin < 5 || isempty(overwrite), overwrite = false; end
if nargin < 6 || isempty(packageParent)
    thisDir = fileparts(mfilename('fullpath'));
    packageParent = fullfile(thisDir, '..', '..', '..', 'matlab');
end

dataFile = char(dataFile);
restoredRoot = char(restoredRoot);
outDir = char(outDir);
packageParent = char(packageParent);
if ischar(maxRecords) || isstring(maxRecords), maxRecords = str2double(maxRecords); end
if ischar(overwrite) || isstring(overwrite)
    overwrite = any(strcmpi(strtrim(char(overwrite)), {'1', 'true', 'yes', 'y'}));
end

if ~exist(dataFile, 'file'), error('Missing clean input file: %s', dataFile); end
if ~exist(outDir, 'dir'), mkdir(outDir); end
outFile = fullfile(outDir, 'clean_input_activation_record_diagnostics.csv');
if exist(outFile, 'file') && ~overwrite
    fprintf('Skipping existing diagnostic file: %s\n', outFile);
    return;
end

addpath(packageParent);
assert_package_file(packageParent, 'denoise_emg_vmd_window.m');
assert_package_file(packageParent, 'denoise_eog_ssa_window.m');
assert_package_file(packageParent, 'denoise_slow_ssa_window.m');

saved = struct( ...
    'emg_from_clean', fullfile(restoredRoot, 'proposed_emg', file_name(dataFile)), ...
    'eog_from_clean', fullfile(restoredRoot, 'proposed_eog', file_name(dataFile)), ...
    'slow_from_clean', fullfile(restoredRoot, 'proposed_slow', file_name(dataFile)), ...
    'eog_after_emg', fullfile(restoredRoot, 'proposed_emg_eog', file_name(dataFile)), ...
    'slow_after_emg_eog', fullfile(restoredRoot, 'proposed_full', file_name(dataFile)));
savedFields = fieldnames(saved);
for i = 1:numel(savedFields)
    if ~exist(saved.(savedFields{i}), 'file')
        error('Missing saved reference output: %s', saved.(savedFields{i}));
    end
end

info = h5info(dataFile);
nUse = min(numel(info.Groups), maxRecords);
if nUse < 1, error('No records found in %s.', dataFile); end

rows = repmat(empty_row(), nUse * 5, 1);
rowIdx = 0;
nFailed = 0;
progressStep = max(1, ceil(nUse / 20));

fprintf('SPAR-EEG clean-input activation diagnostic\n');
fprintf('  data file      : %s\n', dataFile);
fprintf('  records        : %d/%d\n', nUse, numel(info.Groups));
fprintf('  package parent : %s\n', packageParent);
fprintf('  output         : %s\n', outFile);

for r = 1:nUse
    groupName = info.Groups(r).Name;
    recordName = erase(groupName, '/');
    dsIn = [groupName '/eeg_reference'];
    if ~dataset_exists(dataFile, dsIn), dsIn = [groupName '/eeg_signal']; end

    try
        x = double(h5read(dataFile, dsIn));
        if isvector(x), x = x(:); end
        if size(x, 2) ~= 1
            error('Expected one channel, found size %s.', mat2str(size(x)));
        end
        fs = read_freq(dataFile, groupName);

        ticId = tic;
        [yEmg, dEmg] = burstdenoise.denoise_emg_vmd_window(x, fs);
        tEmg = toc(ticId);

        ticId = tic;
        [yEog, dEog] = burstdenoise.denoise_eog_ssa_window(x, fs);
        tEog = toc(ticId);

        ticId = tic;
        [ySlow, dSlow] = burstdenoise.denoise_slow_ssa_window(x, fs);
        tSlow = toc(ticId);

        ticId = tic;
        [yEmgEog, dEmgEog] = burstdenoise.denoise_eog_ssa_window(yEmg, fs);
        tEmgEog = toc(ticId);

        ticId = tic;
        [yFull, dFull] = burstdenoise.denoise_slow_ssa_window(yEmgEog, fs);
        tFull = toc(ticId);

        branches = { ...
            'emg_from_clean', 'EMG', 'Clean', x, yEmg, dEmg, tEmg; ...
            'eog_from_clean', 'EOG', 'Clean', x, yEog, dEog, tEog; ...
            'slow_from_clean', 'Slow', 'Clean', x, ySlow, dSlow, tSlow; ...
            'eog_after_emg', 'EOG', 'EMG output', yEmg, yEmgEog, dEmgEog, tEmgEog; ...
            'slow_after_emg_eog', 'Slow', 'EMG+EOG output', yEmgEog, yFull, dFull, tFull};

        recordRows = repmat(empty_row(), 5, 1);
        for b = 1:size(branches, 1)
            savedY = read_saved(saved.(branches{b,1}), recordName);
            recordRows(b) = summarize_stage( ...
                recordName, branches{b,1}, branches{b,2}, branches{b,3}, ...
                x, branches{b,4}, branches{b,5}, branches{b,6}, ...
                fs, branches{b,7}, savedY);
        end
        rows(rowIdx + (1:5)) = recordRows;
        rowIdx = rowIdx + 5;

        if r == 1 || r == nUse || mod(r, progressStep) == 0
            lastRows = rows(rowIdx-4:rowIdx);
            maxVerify = max([lastRows.verification_relative_error]);
            fprintf('  [%d/%d] %s OK | max verification error %.3g\n', ...
                r, nUse, recordName, maxVerify);
        end
    catch ME
        nFailed = nFailed + 1;
        warning('Record failed: %s | %s', recordName, ME.message);
        for b = 1:5
            rowIdx = rowIdx + 1;
            rows(rowIdx) = empty_row();
            rows(rowIdx).record = string(recordName);
            rows(rowIdx).status = "FAIL: " + string(ME.message);
        end
    end
end

rows = rows(1:rowIdx);
result = struct2table(rows);
tempFile = [tempname(tempdir) '.csv'];
writetable(result, tempFile);
if exist(outFile, 'file'), delete(outFile); end
[ok, msg] = copyfile(tempFile, outFile, 'f');
if ~ok, error('Could not copy diagnostic CSV: %s', msg); end
delete(tempFile);

fprintf('Wrote %s\n', outFile);
fprintf('Rows: %d | failed records: %d\n', height(result), nFailed);
end

function row = summarize_stage(recordName, node, passName, inputSource, cleanX, stageX, y, debug, fs, elapsedSec, savedY)
cleanX = double(cleanX(:));
stageX = double(stageX(:));
y = double(y(:));
savedY = double(savedY(:));
if numel(y) ~= numel(stageX) || numel(y) ~= numel(cleanX) || numel(y) ~= numel(savedY)
    error('Length mismatch while summarizing %s/%s.', recordName, node);
end

row = empty_row();
row.record = string(recordName);
row.node = string(node);
row.pass_name = string(passName);
row.stage_input = string(inputSource);
row.fs = fs;
row.samples = numel(y);
row.status = "OK";
row.elapsed_sec = elapsedSec;
row.stage_input_rrmse_percent = 100 * relative_change(stageX, y);
row.clean_input_rrmse_percent = 100 * relative_change(cleanX, y);
row.output_modified = double(row.stage_input_rrmse_percent > 1e-10);
row.modified_sample_fraction_vs_stage_input = modified_sample_fraction(stageX, y);
row.modified_sample_fraction_vs_clean = modified_sample_fraction(cleanX, y);
row.verification_relative_error = relative_error(savedY, y);
row.verification_max_abs_error = max(abs(savedY - y));

if strcmpi(passName, 'EMG') || strcmpi(passName, 'EOG')
    row.detected_regions = size(debug.regions, 1);
    row.detector_positive = double(row.detected_regions > 0);
    row.initial_exceedance_fraction = mean(debug.z(:) >= debug.cfg.mask_zthr);
    row.mask_fraction = mean(debug.m_common(:) > 0.5);
    row.baseline_fraction = mean(debug.clean_idx(:));
    minBaseline = max(5, round(debug.cfg.min_baseline_sec * fs));
    row.bypass_no_region = double(row.detected_regions < 1);
    row.bypass_insufficient_baseline = double(row.detected_regions >= 1 && nnz(debug.clean_idx) < minBaseline);
    [row.attenuation_applied, row.attenuated_sample_fraction, ...
        row.attenuated_component_count, row.min_gain, ...
        row.mean_gain_reduction_active] = summarize_weights(debug);
    if strcmpi(passName, 'EMG')
        row.treated_component_count = numel(debug.treat_modes);
        row.realized_depth = size(debug.U, 1);
    else
        row.treated_component_count = numel(debug.treat_rcs);
        row.realized_depth = debug.L;
    end
else
    row.gate1_score = scalar_or_nan(debug.gate1_score);
    row.gate1_pass = double(logical(debug.gate1_pass));
    row.detector_positive = row.gate1_pass;
    row.gate2_score = scalar_or_nan(debug.gate2_score);
    row.gate2_pass = logical_or_nan(debug.gate2_pass);
    row.slow_candidate_count = numel(debug.cand_idx);
    row.alpha_eff = scalar_or_nan(debug.alpha_eff);
    row.attenuation_applied = double( ...
        logical_or_false(debug.do_slowpass) && isfinite(row.alpha_eff) && row.alpha_eff > 1e-12);
    row.attenuated_sample_fraction = slow_support_fraction(debug);
    row.bypass_gate1 = double(~logical(debug.gate1_pass));
    if logical(debug.gate1_pass)
        row.bypass_gate2 = double(~logical_or_false(debug.gate2_pass));
    end
    row.realized_depth = scalar_or_nan(debug.L);
end
end

function row = empty_row()
row = struct( ...
    'record', "", 'node', "", 'pass_name', "", 'stage_input', "", ...
    'fs', NaN, 'samples', NaN, 'status', "", 'elapsed_sec', NaN, ...
    'detector_positive', NaN, 'detected_regions', NaN, ...
    'initial_exceedance_fraction', NaN, 'mask_fraction', NaN, ...
    'baseline_fraction', NaN, 'bypass_no_region', NaN, ...
    'bypass_insufficient_baseline', NaN, 'gate1_score', NaN, ...
    'gate1_pass', NaN, 'gate2_score', NaN, 'gate2_pass', NaN, ...
    'bypass_gate1', NaN, 'bypass_gate2', NaN, ...
    'slow_candidate_count', NaN, 'alpha_eff', NaN, ...
    'attenuation_applied', NaN, 'attenuated_sample_fraction', NaN, ...
    'attenuated_component_count', NaN, 'treated_component_count', NaN, ...
    'min_gain', NaN, 'mean_gain_reduction_active', NaN, ...
    'realized_depth', NaN, 'stage_input_rrmse_percent', NaN, ...
    'clean_input_rrmse_percent', NaN, 'output_modified', NaN, ...
    'modified_sample_fraction_vs_stage_input', NaN, ...
    'modified_sample_fraction_vs_clean', NaN, ...
    'verification_relative_error', NaN, 'verification_max_abs_error', NaN);
end

function [applied, sampleFraction, componentCount, minGain, meanReduction] = summarize_weights(debug)
if ~isfield(debug, 'W') || isempty(debug.W)
    applied = 0;
    sampleFraction = 0;
    componentCount = 0;
    minGain = 1;
    meanReduction = 0;
    return;
end
active = debug.W < (1 - 1e-12);
applied = double(any(active(:)));
sampleFraction = mean(any(active, 1));
componentCount = sum(any(active, 2));
minGain = min(debug.W(:));
if any(active(:))
    meanReduction = mean(1 - debug.W(active));
else
    meanReduction = 0;
end
end

function frac = slow_support_fraction(debug)
if ~isfield(debug, 'do_slowpass') || ~logical_or_false(debug.do_slowpass) || ...
        ~isfield(debug, 'x_art') || isempty(debug.x_art) || ...
        ~isfield(debug, 'alpha_eff') || isempty(debug.alpha_eff)
    frac = 0;
    return;
end
delta = abs(double(debug.alpha_eff) .* double(debug.x_art(:)));
tol = 1e-12 * max(1, sqrt(mean(double(debug.yn(:)).^2)));
frac = mean(delta > tol);
end

function frac = modified_sample_fraction(x, y)
x = double(x(:));
y = double(y(:));
tol = 1e-12 * max(1, sqrt(mean(x.^2)));
frac = mean(abs(y - x) > tol);
end

function value = relative_change(x, y)
den = norm(double(x(:)));
if den <= eps
    value = NaN;
else
    value = norm(double(y(:)) - double(x(:))) / den;
end
end

function value = relative_error(reference, observed)
den = norm(double(reference(:)));
if den <= eps
    value = norm(double(observed(:)) - double(reference(:)));
else
    value = norm(double(observed(:)) - double(reference(:))) / den;
end
end

function y = read_saved(filePath, recordName)
y = double(h5read(filePath, ['/' recordName]));
if isvector(y), y = y(:); end
if size(y, 2) ~= 1
    error('Expected one saved channel for %s, found %s.', recordName, mat2str(size(y)));
end
end

function fs = read_freq(inFile, groupName)
fs = 256;
try
    fs = double(h5readatt(inFile, groupName, 'freq'));
catch
end
end

function tf = dataset_exists(filePath, datasetPath)
tf = true;
try
    h5info(filePath, datasetPath);
catch
    tf = false;
end
end

function value = scalar_or_nan(x)
if isempty(x), value = NaN; else, value = double(x(1)); end
end

function value = logical_or_nan(x)
if isempty(x), value = NaN; else, value = double(logical(x(1))); end
end

function value = logical_or_false(x)
if isempty(x), value = false; else, value = logical(x(1)); end
end

function name = file_name(pathValue)
[~, base, ext] = fileparts(pathValue);
name = [base ext];
end

function assert_package_file(packageParent, fileName)
pathValue = fullfile(packageParent, '+burstdenoise', fileName);
if ~exist(pathValue, 'file')
    error('Required package file is not available: %s', pathValue);
end
end
