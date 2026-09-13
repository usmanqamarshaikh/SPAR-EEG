function run_spar_erp_preservation_branches(inputFile, outDir, maxTrials, overwrite, packageParent)
%RUN_SPAR_ERP_PRESERVATION_BRANCHES Apply frozen SPAR-EEG passes to ERP epochs.

if nargin < 1 || isempty(inputFile), inputFile = 'processedEEG.mat'; end
if nargin < 2 || isempty(outDir), outDir = fullfile('results', 'spar_erp_preservation'); end
if nargin < 3 || isempty(maxTrials), maxTrials = Inf; end
if nargin < 4 || isempty(overwrite), overwrite = false; end
if nargin < 5 || isempty(packageParent)
    thisDir = fileparts(mfilename('fullpath'));
    packageParent = fullfile(thisDir, '..', '..', '..', 'matlab');
end

inputFile = char(inputFile);
outDir = char(outDir);
packageParent = char(packageParent);
if ischar(maxTrials) || isstring(maxTrials), maxTrials = str2double(maxTrials); end
if ischar(overwrite) || isstring(overwrite)
    overwrite = any(strcmpi(strtrim(char(overwrite)), {'1', 'true', 'yes', 'y'}));
end
if ~exist(inputFile, 'file'), error('Missing input file: %s', inputFile); end
if ~exist(outDir, 'dir'), mkdir(outDir); end

bundleFile = fullfile(outDir, 'spar_erp_branch_data.mat');
diagnosticFile = fullfile(outDir, 'spar_erp_branch_diagnostics.csv');
if exist(bundleFile, 'file') && exist(diagnosticFile, 'file') && ~overwrite
    fprintf('Skipping existing branch outputs in %s\n', outDir);
    return;
end

addpath(packageParent);
assert_package_file(packageParent, 'denoise_emg_vmd_window.m');
assert_package_file(packageParent, 'denoise_eog_ssa_window.m');
assert_package_file(packageParent, 'denoise_slow_ssa_window.m');

loaded = load(inputFile);
if ~isfield(loaded, 'EEG'), error('Input MAT file does not contain EEG.'); end
EEG = loaded.EEG;
if ndims(EEG.data) ~= 3
    error('Expected epoched EEG [channels x samples x trials], found %s.', mat2str(size(EEG.data)));
end

nChannels = size(EEG.data, 1);
nSamples = size(EEG.data, 2);
nUse = min(size(EEG.data, 3), maxTrials);
if nUse < 1, error('No trials selected.'); end

data_reference = single(EEG.data(:, :, 1:nUse));
data_emg = zeros(size(data_reference), 'single');
data_eog = zeros(size(data_reference), 'single');
data_emg_eog = zeros(size(data_reference), 'single');
data_full = zeros(size(data_reference), 'single');

rows = repmat(empty_row(), nChannels * nUse * 4, 1);
rowIndex = 0;
nFailed = 0;
progressStep = max(1, ceil(nChannels * nUse / 20));
processed = 0;

fprintf('SPAR-EEG low-artifact ERP preservation branches\n');
fprintf('  input        : %s\n', inputFile);
fprintf('  data size    : %s\n', mat2str(size(EEG.data)));
fprintf('  trials       : %d/%d\n', nUse, EEG.trials);
fprintf('  sampling rate: %g Hz\n', EEG.srate);
fprintf('  package      : %s\n', packageParent);

for trial = 1:nUse
    for channel = 1:nChannels
        processed = processed + 1;
        x = double(data_reference(channel, :, trial)).';
        label = string(EEG.chanlocs(channel).labels);

        try
            ticId = tic;
            [yEmg, dEmg] = burstdenoise.denoise_emg_vmd_window(x, EEG.srate);
            tEmg = toc(ticId);

            ticId = tic;
            [yEog, dEog] = burstdenoise.denoise_eog_ssa_window(x, EEG.srate);
            tEog = toc(ticId);

            ticId = tic;
            [yEmgEog, dEmgEog] = burstdenoise.denoise_eog_ssa_window(yEmg, EEG.srate);
            tEmgEog = toc(ticId);

            ticId = tic;
            [yFull, dFull] = burstdenoise.denoise_slow_ssa_window(yEmgEog, EEG.srate);
            tFull = toc(ticId);

            data_emg(channel, :, trial) = single(yEmg(:));
            data_eog(channel, :, trial) = single(yEog(:));
            data_emg_eog(channel, :, trial) = single(yEmgEog(:));
            data_full(channel, :, trial) = single(yFull(:));

            stageRows = [ ...
                summarize_stage(trial, channel, label, "emg_from_reference", "EMG", ...
                    "Reference", x, x, yEmg, dEmg, EEG.srate, tEmg); ...
                summarize_stage(trial, channel, label, "eog_from_reference", "EOG", ...
                    "Reference", x, x, yEog, dEog, EEG.srate, tEog); ...
                summarize_stage(trial, channel, label, "eog_after_emg", "EOG", ...
                    "EMG output", x, yEmg, yEmgEog, dEmgEog, EEG.srate, tEmgEog); ...
                summarize_stage(trial, channel, label, "slow_after_emg_eog", "Slow", ...
                    "EMG+EOG output", x, yEmgEog, yFull, dFull, EEG.srate, tFull)];
            rows(rowIndex + (1:4)) = stageRows;
            rowIndex = rowIndex + 4;
        catch ME
            nFailed = nFailed + 1;
            data_emg(channel, :, trial) = single(x);
            data_eog(channel, :, trial) = single(x);
            data_emg_eog(channel, :, trial) = single(x);
            data_full(channel, :, trial) = single(x);
            for stage = 1:4
                rowIndex = rowIndex + 1;
                rows(rowIndex) = empty_row();
                rows(rowIndex).trial = trial;
                rows(rowIndex).channel_index = channel;
                rows(rowIndex).channel_label = label;
                rows(rowIndex).status = "FAIL: " + string(ME.message);
            end
            warning('Trial %d channel %s failed: %s', trial, label, ME.message);
        end

        if processed == 1 || processed == nChannels * nUse || mod(processed, progressStep) == 0
            fprintf('  [%d/%d] trial=%d channel=%s | failures=%d\n', ...
                processed, nChannels * nUse, trial, label, nFailed);
        end
    end
end

times = double(EEG.times(:).');
srate = double(EEG.srate);
channel_labels = {EEG.chanlocs.labels};
channel_theta = double([EEG.chanlocs.theta]);
channel_radius = double([EEG.chanlocs.radius]);
trial_indices = 1:nUse;
metadata = struct();
metadata.source_file = inputFile;
metadata.source_setname = EEG.setname;
metadata.processing_unit = 'independent channel-trial epoch';
metadata.adaptive_streaming = false;
metadata.internal_defaults = 'frozen deployed defaults';
metadata.pass_branches = {'reference', 'emg', 'eog', 'emg+eog', 'emg+eog+slow'};
metadata.total_source_trials = EEG.trials;
metadata.processed_trials = nUse;
metadata.failed_channel_trials = nFailed;

atomic_save_bundle(bundleFile, data_reference, data_emg, data_eog, ...
    data_emg_eog, data_full, times, srate, channel_labels, channel_theta, ...
    channel_radius, trial_indices, metadata);

diagnostics = struct2table(rows(1:rowIndex));
atomic_write_table(diagnosticFile, diagnostics);

if nUse == EEG.trials
    branchDir = fullfile(outDir, 'eeglab_branches');
    if ~exist(branchDir, 'dir'), mkdir(branchDir); end
    save_eeg_branch(fullfile(branchDir, 'processedEEG_reference.mat'), EEG, ...
        data_reference, 'Low-artifact reference', 'reference', false);
    save_eeg_branch(fullfile(branchDir, 'processedEEG_emg.mat'), EEG, ...
        data_emg, 'Low-artifact reference + SPAR EMG', 'emg', true);
    save_eeg_branch(fullfile(branchDir, 'processedEEG_eog.mat'), EEG, ...
        data_eog, 'Low-artifact reference + SPAR EOG', 'eog', true);
    save_eeg_branch(fullfile(branchDir, 'processedEEG_emg_eog.mat'), EEG, ...
        data_emg_eog, 'Low-artifact reference + SPAR EMG+EOG', 'emg+eog', true);
    save_eeg_branch(fullfile(branchDir, 'processedEEG_full.mat'), EEG, ...
        data_full, 'Low-artifact reference + full SPAR sequence', 'emg+eog+slow', true);
else
    fprintf('Smoke/subset run: individual full EEGLAB branch files were not written.\n');
end

fprintf('Wrote bundle: %s\n', bundleFile);
fprintf('Wrote diagnostics: %s\n', diagnosticFile);
fprintf('Channel-trials: %d | failures: %d\n', nChannels * nUse, nFailed);
end

function row = summarize_stage(trial, channel, label, node, passName, inputSource, reference, stageInput, output, debug, fs, elapsedSec)
reference = double(reference(:));
stageInput = double(stageInput(:));
output = double(output(:));
row = empty_row();
row.trial = trial;
row.channel_index = channel;
row.channel_label = label;
row.node = node;
row.pass_name = passName;
row.stage_input = inputSource;
row.status = "OK";
row.elapsed_sec = elapsedSec;
row.detector_positive = 0;
row.attenuation_applied = 0;
row.stage_input_rrmse_percent = 100 * relative_change(stageInput, output);
row.reference_rrmse_percent = 100 * relative_change(reference, output);
row.modified_sample_fraction_vs_stage_input = modified_sample_fraction(stageInput, output);
row.modified_sample_fraction_vs_reference = modified_sample_fraction(reference, output);

if passName == "EMG" || passName == "EOG"
    row.detected_regions = size(debug.regions, 1);
    row.detector_positive = double(row.detected_regions > 0);
    row.initial_exceedance_fraction = mean(debug.z(:) >= debug.cfg.mask_zthr);
    row.mask_fraction = mean(debug.m_common(:) > 0.5);
    row.baseline_fraction = mean(debug.clean_idx(:));
    minBaseline = max(5, round(debug.cfg.min_baseline_sec * fs));
    row.bypass_no_region = double(row.detected_regions < 1);
    row.bypass_insufficient_baseline = double(row.detected_regions >= 1 && nnz(debug.clean_idx) < minBaseline);
    [row.attenuation_applied, row.attenuated_sample_fraction, row.min_gain] = summarize_weights(debug);
else
    row.gate1_score = scalar_or_nan(debug.gate1_score);
    row.gate1_pass = logical_or_nan(debug.gate1_pass);
    row.detector_positive = logical_or_false(debug.gate1_pass);
    row.gate2_score = scalar_or_nan(debug.gate2_score);
    row.gate2_pass = logical_or_nan(debug.gate2_pass);
    row.alpha_eff = scalar_or_nan(debug.alpha_eff);
    row.attenuation_applied = double(logical_or_false(debug.do_slowpass) && ...
        isfinite(row.alpha_eff) && row.alpha_eff > 1e-12);
    row.attenuated_sample_fraction = slow_support_fraction(debug);
end
end

function row = empty_row()
row = struct( ...
    'trial', NaN, 'channel_index', NaN, 'channel_label', "", ...
    'node', "", 'pass_name', "", 'stage_input', "", 'status', "", ...
    'elapsed_sec', NaN, 'detector_positive', NaN, 'detected_regions', NaN, ...
    'initial_exceedance_fraction', NaN, 'mask_fraction', NaN, ...
    'baseline_fraction', NaN, 'bypass_no_region', NaN, ...
    'bypass_insufficient_baseline', NaN, 'gate1_score', NaN, ...
    'gate1_pass', NaN, 'gate2_score', NaN, 'gate2_pass', NaN, ...
    'alpha_eff', NaN, 'attenuation_applied', NaN, ...
    'attenuated_sample_fraction', NaN, 'min_gain', NaN, ...
    'stage_input_rrmse_percent', NaN, 'reference_rrmse_percent', NaN, ...
    'modified_sample_fraction_vs_stage_input', NaN, ...
    'modified_sample_fraction_vs_reference', NaN);
end

function [applied, fraction, minGain] = summarize_weights(debug)
if ~isfield(debug, 'W') || isempty(debug.W)
    applied = 0; fraction = 0; minGain = 1; return;
end
active = debug.W < (1 - 1e-12);
applied = double(any(active(:)));
fraction = mean(any(active, 1));
minGain = min(debug.W(:));
end

function fraction = slow_support_fraction(debug)
if ~isfield(debug, 'do_slowpass') || ~logical_or_false(debug.do_slowpass) || ...
        ~isfield(debug, 'x_art') || isempty(debug.x_art) || ...
        ~isfield(debug, 'alpha_eff') || isempty(debug.alpha_eff)
    fraction = 0; return;
end
delta = abs(double(debug.alpha_eff) .* double(debug.x_art(:)));
tol = 1e-12 * max(1, sqrt(mean(double(debug.yn(:)).^2)));
fraction = mean(delta > tol);
end

function value = relative_change(x, y)
den = norm(double(x(:)));
if den <= eps, value = NaN; else, value = norm(double(y(:)) - double(x(:))) / den; end
end

function fraction = modified_sample_fraction(x, y)
x = double(x(:)); y = double(y(:));
tol = 1e-12 * max(1, sqrt(mean(x.^2)));
fraction = mean(abs(y - x) > tol);
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

function atomic_save_bundle(filePath, data_reference, data_emg, data_eog, data_emg_eog, data_full, times, srate, channel_labels, channel_theta, channel_radius, trial_indices, metadata)
tempFile = [tempname(tempdir) '.mat'];
save(tempFile, 'data_reference', 'data_emg', 'data_eog', 'data_emg_eog', ...
    'data_full', 'times', 'srate', 'channel_labels', 'channel_theta', ...
    'channel_radius', 'trial_indices', 'metadata', '-v7');
atomic_copy(tempFile, filePath);
end

function atomic_write_table(filePath, value)
tempFile = [tempname(tempdir) '.csv'];
writetable(value, tempFile);
atomic_copy(tempFile, filePath);
end

function save_eeg_branch(filePath, template, data, setname, branchName, clearIca)
EEG = template;
EEG.data = single(data);
EEG.setname = setname;
EEG.filename = file_name(filePath);
EEG.filepath = fileparts(filePath);
EEG.saved = 'no';
if clearIca
    EEG.icaact = [];
    EEG.icawinv = [];
    EEG.icasphere = [];
    EEG.icaweights = [];
    EEG.icachansind = [];
    if isfield(EEG.etc, 'ic_classification')
        EEG.etc.pre_spar_ic_classification = EEG.etc.ic_classification;
        EEG.etc = rmfield(EEG.etc, 'ic_classification');
    end
end
EEG.etc.spar_erp_preservation = struct( ...
    'branch', branchName, ...
    'processing_unit', 'independent channel-trial epoch', ...
    'adaptive_streaming', false, ...
    'internal_defaults', 'frozen deployed defaults');
EEG.history = sprintf('%s\n%% SPAR-EEG preservation branch: %s; frozen defaults; channel-trial epochs.', ...
    EEG.history, branchName);
tempFile = [tempname(tempdir) '.mat'];
save(tempFile, 'EEG', '-v7');
atomic_copy(tempFile, filePath);
end

function atomic_copy(tempFile, finalFile)
if exist(finalFile, 'file'), delete(finalFile); end
[ok, message] = copyfile(tempFile, finalFile, 'f');
if ~ok, error('Could not write %s: %s', finalFile, message); end
delete(tempFile);
end

function name = file_name(pathValue)
[~, base, ext] = fileparts(pathValue); name = [base ext];
end

function assert_package_file(packageParent, fileName)
pathValue = fullfile(packageParent, '+burstdenoise', fileName);
if ~exist(pathValue, 'file'), error('Required package file is unavailable: %s', pathValue); end
end
