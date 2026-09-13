function run_artifact_geometry_h5(dataFile, outFile, diagnosticsFile, overwrite, packageParent)
%RUN_ARTIFACT_GEOMETRY_H5 Process paired EMG geometry variants with frozen defaults.
%
% Saves the EMG-only, EMG+EOG, and full EMG+EOG+slow branches. The
% synthesis/evaluation masks are never supplied to the denoising functions.

if nargin < 1 || isempty(dataFile)
    dataFile = fullfile('results', 'artifact_geometry_emg_minus10_n500', 'data', ...
        'emg_geometry_minus10.h5');
end
if nargin < 2 || isempty(outFile)
    outFile = fullfile('results', 'artifact_geometry_emg_minus10_n500', 'restored', ...
        'emg_geometry_minus10_restored.h5');
end
if nargin < 3 || isempty(diagnosticsFile)
    diagnosticsFile = fullfile('results', 'artifact_geometry_emg_minus10_n500', ...
        'analysis', 'artifact_geometry_stage_diagnostics.csv');
end
if nargin < 4 || isempty(overwrite), overwrite = false; end
if nargin < 5 || isempty(packageParent)
    thisDir = fileparts(mfilename('fullpath'));
    packageParent = fullfile(thisDir, '..', '..', '..', 'matlab');
end

dataFile = char(dataFile);
outFile = char(outFile);
diagnosticsFile = char(diagnosticsFile);
packageParent = char(packageParent);
if ischar(overwrite) || isstring(overwrite)
    overwrite = any(strcmpi(strtrim(char(overwrite)), {'1', 'true', 'yes', 'y'}));
end
if ~exist(dataFile, 'file'), error('Missing geometry input file: %s', dataFile); end
if exist(outFile, 'file') && exist(diagnosticsFile, 'file') && ~overwrite
    fprintf('Skipping existing geometry outputs:\n  %s\n  %s\n', outFile, diagnosticsFile);
    return;
end

addpath(packageParent);
assert_package_file(packageParent, 'denoise_emg_vmd_window.m');
assert_package_file(packageParent, 'denoise_eog_ssa_window.m');
assert_package_file(packageParent, 'denoise_slow_ssa_window.m');

ensure_parent(outFile);
ensure_parent(diagnosticsFile);
tempDir = fullfile(tempdir, 'spar_artifact_geometry');
if ~exist(tempDir, 'dir'), mkdir(tempDir); end
tempOut = [tempname(tempDir) '.h5'];
tempCsv = [tempname(tempDir) '.csv'];

info = h5info(dataFile);
nRecords = numel(info.Groups);
if nRecords < 1, error('No geometry records found in %s.', dataFile); end
rows = repmat(empty_row(), nRecords * 3, 1);
rowIdx = 0;
nFailed = 0;
progressStep = max(1, ceil(nRecords / 40));

h5writeatt_safe(tempOut, '/', 'source_file', dataFile);
h5writeatt_safe(tempOut, '/', 'processing', 'frozen EMG -> EOG -> slow defaults');
h5writeatt_safe(tempOut, '/', 'mask_use', 'none during denoising');
h5writeatt_safe(tempOut, '/', 'record_count', nRecords);

fprintf('SPAR-EEG EMG artifact-geometry robustness\n');
fprintf('  input          : %s\n', dataFile);
fprintf('  records        : %d\n', nRecords);
fprintf('  package parent : %s\n', packageParent);

for r = 1:nRecords
    groupName = info.Groups(r).Name;
    recordName = erase(groupName, '/');
    geometry = string(h5readatt(dataFile, groupName, 'geometry'));
    recordId = double(h5readatt(dataFile, groupName, 'record_id'));
    fs = read_freq(dataFile, groupName);
    try
        x = double(h5read(dataFile, [groupName '/eeg_signal']));
        if isvector(x), x = x(:); end
        if size(x, 2) ~= 1
            error('Expected one channel, found size %s.', mat2str(size(x)));
        end

        ticId = tic;
        [yEmg, dEmg] = burstdenoise.denoise_emg_vmd_window(x, fs);
        tEmg = toc(ticId);
        ticId = tic;
        [yEmgEog, dEog] = burstdenoise.denoise_eog_ssa_window(yEmg, fs);
        tEog = toc(ticId);
        ticId = tic;
        [yFull, dSlow] = burstdenoise.denoise_slow_ssa_window(yEmgEog, fs);
        tSlow = toc(ticId);

        yEmg = double(yEmg(:));
        yEmgEog = double(yEmgEog(:));
        yFull = double(yFull(:));
        write_dataset(tempOut, [groupName '/emg'], yEmg);
        write_dataset(tempOut, [groupName '/emg_eog'], yEmgEog);
        write_dataset(tempOut, [groupName '/full'], yFull);
        h5writeatt_safe(tempOut, groupName, 'status', 'OK');
        h5writeatt_safe(tempOut, groupName, 'geometry', char(geometry));
        h5writeatt_safe(tempOut, groupName, 'record_id', recordId);
        h5writeatt_safe(tempOut, groupName, 'fs', fs);

        branches = { ...
            'emg', 'EMG', 'Noisy input', x, yEmg, dEmg, tEmg; ...
            'eog_after_emg', 'EOG', 'EMG output', yEmg, yEmgEog, dEog, tEog; ...
            'slow_after_emg_eog', 'Slow', 'EMG+EOG output', yEmgEog, yFull, dSlow, tSlow};
        for b = 1:size(branches, 1)
            rowIdx = rowIdx + 1;
            rows(rowIdx) = summarize_stage(recordName, geometry, recordId, ...
                branches{b,1}, branches{b,2}, branches{b,3}, ...
                branches{b,4}, branches{b,5}, branches{b,6}, fs, branches{b,7});
        end

        if r == 1 || r == nRecords || mod(r, progressStep) == 0
            fprintf('  [%d/%d] %s OK | EMG %.3f s | EOG %.3f s | slow %.3f s\n', ...
                r, nRecords, recordName, tEmg, tEog, tSlow);
        end
    catch ME
        nFailed = nFailed + 1;
        warning('Record failed: %s | %s', recordName, ME.message);
        try
            write_dataset(tempOut, [groupName '/emg'], x);
            write_dataset(tempOut, [groupName '/emg_eog'], x);
            write_dataset(tempOut, [groupName '/full'], x);
            h5writeatt_safe(tempOut, groupName, 'status', ['FAIL: ' ME.message]);
        catch ME2
            warning('Could not write fallback for %s | %s', recordName, ME2.message);
        end
        for b = 1:3
            rowIdx = rowIdx + 1;
            rows(rowIdx) = empty_row();
            rows(rowIdx).record = string(recordName);
            rows(rowIdx).geometry = geometry;
            rows(rowIdx).record_id = recordId;
            rows(rowIdx).status = "FAIL: " + string(ME.message);
        end
    end
end

rows = rows(1:rowIdx);
writetable(struct2table(rows), tempCsv);
h5writeatt_safe(tempOut, '/', 'failed_record_count', nFailed);

if exist(outFile, 'file'), delete(outFile); end
[ok, msg] = copyfile(tempOut, outFile, 'f');
if ~ok, error('Could not copy restored H5: %s', msg); end
if exist(diagnosticsFile, 'file'), delete(diagnosticsFile); end
[ok, msg] = copyfile(tempCsv, diagnosticsFile, 'f');
if ~ok, error('Could not copy diagnostic CSV: %s', msg); end
delete(tempOut);
delete(tempCsv);

fprintf('Wrote %s\n', outFile);
fprintf('Wrote %s\n', diagnosticsFile);
fprintf('Records: %d | diagnostic rows: %d | failed: %d\n', nRecords, rowIdx, nFailed);
end

function row = summarize_stage(recordName, geometry, recordId, node, passName, ...
    inputSource, x, y, debug, fs, elapsedSec)
x = double(x(:));
y = double(y(:));
row = empty_row();
row.record = string(recordName);
row.geometry = string(geometry);
row.record_id = recordId;
row.node = string(node);
row.pass_name = string(passName);
row.stage_input = string(inputSource);
row.fs = fs;
row.samples = numel(y);
row.status = "OK";
row.elapsed_sec = elapsedSec;
row.stage_input_rrmse_percent = 100 * relative_change(x, y);
row.output_modified = double(row.stage_input_rrmse_percent > 1e-10);

if strcmpi(passName, 'EMG') || strcmpi(passName, 'EOG')
    row.detected_regions = size(debug.regions, 1);
    row.detector_positive = double(row.detected_regions > 0);
    row.initial_exceedance_fraction = mean(debug.z(:) >= debug.cfg.mask_zthr);
    row.mask_fraction = mean(debug.m_common(:) > 0.5);
    row.baseline_fraction = mean(debug.clean_idx(:));
    minBaseline = max(5, round(debug.cfg.min_baseline_sec * fs));
    row.min_baseline_required = minBaseline;
    row.bypass_no_region = double(row.detected_regions < 1);
    row.bypass_insufficient_baseline = double( ...
        row.detected_regions >= 1 && nnz(debug.clean_idx) < minBaseline);
    [row.attenuation_applied, row.attenuated_sample_fraction, ...
        row.min_gain] = summarize_weights(debug);
    if strcmpi(passName, 'EMG')
        row.realized_depth = size(debug.U, 1);
        row.treated_component_count = numel(debug.treat_modes);
    else
        row.realized_depth = debug.L;
        row.treated_component_count = numel(debug.treat_rcs);
    end
else
    row.gate1_score = scalar_or_nan(debug.gate1_score);
    row.gate1_pass = double(logical(debug.gate1_pass));
    row.gate2_score = scalar_or_nan(debug.gate2_score);
    row.gate2_pass = logical_or_nan(debug.gate2_pass);
    row.alpha_eff = scalar_or_nan(debug.alpha_eff);
    row.attenuation_applied = double(logical_or_false(debug.do_slowpass) && ...
        isfinite(row.alpha_eff) && row.alpha_eff > 1e-12);
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
    'record', "", 'geometry', "", 'record_id', NaN, 'node', "", ...
    'pass_name', "", 'stage_input', "", 'fs', NaN, 'samples', NaN, ...
    'status', "", 'elapsed_sec', NaN, 'detector_positive', NaN, ...
    'detected_regions', NaN, 'initial_exceedance_fraction', NaN, ...
    'mask_fraction', NaN, 'baseline_fraction', NaN, ...
    'min_baseline_required', NaN, 'bypass_no_region', NaN, ...
    'bypass_insufficient_baseline', NaN, 'gate1_score', NaN, ...
    'gate1_pass', NaN, 'gate2_score', NaN, 'gate2_pass', NaN, ...
    'bypass_gate1', NaN, 'bypass_gate2', NaN, 'alpha_eff', NaN, ...
    'attenuation_applied', NaN, 'attenuated_sample_fraction', NaN, ...
    'treated_component_count', NaN, 'min_gain', NaN, 'realized_depth', NaN, ...
    'stage_input_rrmse_percent', NaN, 'output_modified', NaN);
end

function [applied, sampleFraction, minGain] = summarize_weights(debug)
if ~isfield(debug, 'W') || isempty(debug.W)
    applied = 0; sampleFraction = 0; minGain = 1; return;
end
active = debug.W < (1 - 1e-12);
applied = double(any(active(:)));
sampleFraction = mean(any(active, 1));
minGain = min(debug.W(:));
end

function frac = slow_support_fraction(debug)
if ~isfield(debug, 'do_slowpass') || ~logical_or_false(debug.do_slowpass) || ...
        ~isfield(debug, 'x_art') || isempty(debug.x_art) || ...
        ~isfield(debug, 'alpha_eff') || isempty(debug.alpha_eff)
    frac = 0; return;
end
delta = abs(double(debug.alpha_eff) .* double(debug.x_art(:)));
tol = 1e-12 * max(1, sqrt(mean(double(debug.yn(:)).^2)));
frac = mean(delta > tol);
end

function value = relative_change(x, y)
den = norm(double(x(:)));
if den <= eps, value = NaN; else, value = norm(double(y(:)) - double(x(:))) / den; end
end

function value = scalar_or_nan(x)
if isempty(x), value = NaN; else, value = double(x(1)); end
end

function value = logical_or_nan(x)
if isempty(x), value = NaN; else, value = double(logical(x(1))); end
end

function value = logical_or_false(x)
value = ~isempty(x) && logical(x(1));
end

function fs = read_freq(filePath, groupName)
fs = 256;
try, fs = double(h5readatt(filePath, groupName, 'freq')); catch, end
end

function write_dataset(filePath, datasetPath, values)
values = double(values(:));
h5create(filePath, datasetPath, size(values), 'Datatype', 'double');
h5write(filePath, datasetPath, values);
end

function ensure_parent(filePath)
parent = fileparts(filePath);
if ~isempty(parent) && ~exist(parent, 'dir'), mkdir(parent); end
end

function assert_package_file(packageParent, functionFile)
expected = fullfile(packageParent, '+burstdenoise', functionFile);
if ~exist(expected, 'file'), error('Missing deployed package function: %s', expected); end
end

function h5writeatt_safe(fileName, location, attrName, attrValue)
try
    if ~exist(fileName, 'file')
        fid = H5F.create(fileName); H5F.close(fid);
    end
    h5writeatt(fileName, location, attrName, attrValue);
catch ME
    warning('Could not write attribute %s at %s | %s', attrName, location, ME.message);
end
end
