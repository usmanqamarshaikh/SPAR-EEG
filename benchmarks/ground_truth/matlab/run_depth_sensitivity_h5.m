function run_depth_sensitivity_h5(dataDir, outRoot, manifestPath, depths, overwrite, packageParent, snrLevels)
%RUN_DEPTH_SENSITIVITY_H5 Isolated decomposition-depth sensitivity runner.
%
% Varies only the VMD mode count for the EMG pass and only the realized SSA
% embedding length for the EOG pass. All other internal defaults are left
% unchanged. The selected record IDs are shared across both noise domains.

if nargin < 1 || isempty(dataDir), dataDir = 'data'; end
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile('results', 'restored_depth_sensitivity_exploratory');
end
if nargin < 3 || isempty(manifestPath)
    error('A record-selection manifest is required.');
end
if nargin < 4 || isempty(depths), depths = [4 8 12]; end
if nargin < 5 || isempty(overwrite), overwrite = false; end
if nargin < 6 || isempty(packageParent)
    thisDir = fileparts(mfilename('fullpath'));
    packageParent = fullfile(thisDir, '..', '..', '..', 'matlab');
end
if nargin < 7 || isempty(snrLevels), snrLevels = -10; end

dataDir = char(dataDir);
outRoot = char(outRoot);
manifestPath = char(manifestPath);
packageParent = char(packageParent);
depths = unique(double(depths(:).'), 'stable');
snrLevels = unique(double(snrLevels(:).'), 'stable');

if ischar(overwrite) || isstring(overwrite)
    overwrite = any(strcmpi(strtrim(char(overwrite)), {'1', 'true', 'yes', 'y'}));
end
if any(~isfinite(depths) | depths < 2 | depths ~= round(depths))
    error('Depths must be finite integers greater than or equal to 2.');
end
if any(~isfinite(snrLevels))
    error('SNR levels must be finite.');
end

selection = readtable(manifestPath);
if ~ismember('record_id', selection.Properties.VariableNames)
    error('Selection manifest must contain a record_id column.');
end
recordIds = double(selection.record_id(:));
if isempty(recordIds) || any(~isfinite(recordIds) | recordIds < 0 | recordIds ~= round(recordIds))
    error('record_id values must be non-negative integers.');
end

addpath(packageParent);
if ~exist(outRoot, 'dir'), mkdir(outRoot); end

specs = struct( ...
    'family', {'vmd', 'ssa'}, ...
    'prefix', {'emg', 'eog'});

fprintf('SPAR-EEG exploratory decomposition-depth sensitivity\n');
fprintf('  selected records: %d\n', numel(recordIds));
fprintf('  depths          : %s\n', mat2str(depths));
fprintf('  SNR levels      : %s dB\n', mat2str(snrLevels));
fprintf('  package parent  : %s\n', packageParent);

for s = 1:numel(specs)
    for n = 1:numel(snrLevels)
        snrDb = snrLevels(n);
        inputName = sensitivity_file_name(specs(s).prefix, snrDb);
        inFile = fullfile(dataDir, inputName);
        if ~exist(inFile, 'file')
            error('Missing input file: %s', inFile);
        end

        for d = 1:numel(depths)
        depth = depths(d);
        settingKey = setting_key(specs(s).family, depth);
        outDir = fullfile(outRoot, settingKey);
        if ~exist(outDir, 'dir'), mkdir(outDir); end
        finalOutFile = fullfile(outDir, inputName);

        if exist(finalOutFile, 'file') && ~overwrite
            fprintf('Skipping existing output: %s\n', finalOutFile);
            continue;
        end

        tempOutDir = fullfile(tempdir, 'spar_depth_sensitivity');
        if ~exist(tempOutDir, 'dir'), mkdir(tempOutDir); end
        outFile = [tempname(tempOutDir) '.h5'];

        h5writeatt_safe(outFile, '/', 'source_file', inputName);
        h5writeatt_safe(outFile, '/', 'test_family', specs(s).family);
        h5writeatt_safe(outFile, '/', 'configured_depth', depth);
        h5writeatt_safe(outFile, '/', 'nominal_snr_db', snrDb);
        h5writeatt_safe(outFile, '/', 'selection_manifest', manifestPath);
        h5writeatt_safe(outFile, '/', 'selected_record_count', numel(recordIds));
        h5writeatt_safe(outFile, '/', 'other_defaults', 'frozen_internal_defaults');

        fprintf('\n%s | SNR=%g dB | depth=%d | records=%d\n', ...
            upper(specs(s).family), snrDb, depth, numel(recordIds));
        nFailed = 0;
        progressStep = max(1, ceil(numel(recordIds) / 20));
        for r = 1:numel(recordIds)
            recordName = sprintf('%s_%d', specs(s).prefix, recordIds(r));
            groupName = ['/' recordName];
            dsIn = [groupName '/eeg_signal'];
            dsOut = ['/' recordName];

            try
                x = double(h5read(inFile, dsIn));
                if isvector(x), x = x(:); end
                if size(x, 2) ~= 1
                    error('Expected a single-channel epoch, found size %s.', mat2str(size(x)));
                end
                fs = read_freq(inFile, groupName);

                ticId = tic;
                if strcmp(specs(s).family, 'vmd')
                    cfg = struct('NumIMFs', depth);
                    [y, debug] = burstdenoise.denoise_emg_vmd_window(x, fs, cfg);
                    realizedDepth = size(debug.U, 1);
                else
                    cfg = struct('L_min', depth);
                    [y, debug] = burstdenoise.denoise_eog_ssa_window(x, fs, cfg);
                    realizedDepth = debug.L;
                end
                elapsedSec = toc(ticId);
                y = double(y(:));

                h5create(outFile, dsOut, size(y), 'Datatype', 'double');
                h5write(outFile, dsOut, y);
                h5writeatt_safe(outFile, dsOut, 'status', 'OK');
                h5writeatt_safe(outFile, dsOut, 'fs', fs);
                h5writeatt_safe(outFile, dsOut, 'test_family', specs(s).family);
                h5writeatt_safe(outFile, dsOut, 'configured_depth', depth);
                h5writeatt_safe(outFile, dsOut, 'realized_depth', realizedDepth);
                h5writeatt_safe(outFile, dsOut, 'detected_regions', size(debug.regions, 1));
                h5writeatt_safe(outFile, dsOut, 'clean_samples', nnz(debug.clean_idx));
                h5writeatt_safe(outFile, dsOut, 'mask_fraction', mean(debug.m_common(:) > 0.5));
                h5writeatt_safe(outFile, dsOut, 'attenuated_sample_fraction', attenuation_fraction(debug));
                h5writeatt_safe(outFile, dsOut, 'output_change_rrmse', relative_change(x, y));
                h5writeatt_safe(outFile, dsOut, 'elapsed_sec', elapsedSec);

                if r == 1 || r == numel(recordIds) || mod(r, progressStep) == 0
                    fprintf('  [%d/%d] %s OK | realized=%d | regions=%d | %.3f s\n', ...
                        r, numel(recordIds), recordName, realizedDepth, ...
                        size(debug.regions, 1), elapsedSec);
                end
            catch ME
                nFailed = nFailed + 1;
                warning('Record failed: %s | %s', recordName, ME.message);
                try
                    x = double(h5read(inFile, dsIn));
                    if isvector(x), x = x(:); end
                    h5create(outFile, dsOut, size(x), 'Datatype', 'double');
                    h5write(outFile, dsOut, x);
                    h5writeatt_safe(outFile, dsOut, 'status', ['FAIL: ' ME.message]);
                catch ME2
                    warning('Could not write fallback for %s | %s', recordName, ME2.message);
                end
            end
        end

        h5writeatt_safe(outFile, '/', 'failed_record_count', nFailed);
        if exist(finalOutFile, 'file'), delete(finalOutFile); end
        [copyOk, copyMsg] = copyfile(outFile, finalOutFile, 'f');
        if ~copyOk
            error('Could not copy completed H5 to %s | %s', finalOutFile, copyMsg);
        end
        delete(outFile);
        fprintf('Wrote: %s | failures=%d\n', finalOutFile, nFailed);
        end
    end
end

fprintf('\nDepth-sensitivity restoration finished.\n');
end

function fileName = sensitivity_file_name(prefix, snrDb)
if abs(snrDb - round(snrDb)) < 1e-12
    snrText = sprintf('%d', round(snrDb));
else
    snrText = sprintf('%g', snrDb);
end
fileName = sprintf('03_denoise-net_%s_%sdB.h5', prefix, snrText);
end

function key = setting_key(family, depth)
if strcmp(family, 'vmd')
    key = sprintf('vmd_k%d', depth);
else
    key = sprintf('ssa_l%d', depth);
end
end

function frac = attenuation_fraction(debug)
if ~isfield(debug, 'W') || isempty(debug.W)
    frac = 0;
    return;
end
frac = mean(any(debug.W < (1 - 1e-12), 1));
end

function value = relative_change(x, y)
den = norm(double(x(:)));
if den <= eps
    value = NaN;
else
    value = norm(double(y(:)) - double(x(:))) / den;
end
end

function fs = read_freq(inFile, groupName)
fs = 256;
try
    fs = double(h5readatt(inFile, groupName, 'freq'));
catch
end
end

function h5writeatt_safe(fileName, location, attrName, attrValue)
try
    if ~exist(fileName, 'file')
        fid = H5F.create(fileName);
        H5F.close(fid);
    end
    h5writeatt(fileName, location, attrName, attrValue);
catch ME
    warning('Could not write attribute %s at %s | %s', attrName, location, ME.message);
end
end
