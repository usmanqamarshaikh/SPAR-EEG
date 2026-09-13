function run_dilation_sensitivity_h5(dataDir, outRoot, manifestPath, multipliers, overwrite, packageParent, snrLevels)
%RUN_DILATION_SENSITIVITY_H5 Isolated temporal-margin sensitivity runner.

if nargin < 1 || isempty(dataDir), dataDir = 'data'; end
if nargin < 2 || isempty(outRoot)
    outRoot = fullfile('results', 'restored_dilation_sensitivity');
end
if nargin < 3 || isempty(manifestPath)
    error('A record-selection manifest is required.');
end
if nargin < 4 || isempty(multipliers), multipliers = [0 1 2]; end
if nargin < 5 || isempty(overwrite), overwrite = false; end
if nargin < 6 || isempty(packageParent)
    thisDir = fileparts(mfilename('fullpath'));
    packageParent = fullfile(thisDir, '..', '..', '..', 'matlab');
end
if nargin < 7 || isempty(snrLevels), snrLevels = [-20 -10 0]; end

dataDir = char(dataDir);
outRoot = char(outRoot);
manifestPath = char(manifestPath);
packageParent = char(packageParent);
multipliers = unique(double(multipliers(:).'), 'stable');
snrLevels = unique(double(snrLevels(:).'), 'stable');
if ischar(overwrite) || isstring(overwrite)
    overwrite = any(strcmpi(strtrim(char(overwrite)), {'1', 'true', 'yes', 'y'}));
end
if any(~isfinite(multipliers) | multipliers < 0)
    error('Dilation multipliers must be finite and non-negative.');
end
if any(~isfinite(snrLevels)), error('SNR levels must be finite.'); end

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
    'prefix', {'emg', 'eog'}, ...
    'depth', {8, 12}, ...
    'defaultDilationSec', {0.14, 0.12});

fprintf('SPAR-EEG dilation sensitivity\n');
fprintf('  selected records : %d\n', numel(recordIds));
fprintf('  multipliers      : %s\n', mat2str(multipliers));
fprintf('  SNR levels       : %s dB\n', mat2str(snrLevels));
fprintf('  package parent   : %s\n', packageParent);

for s = 1:numel(specs)
    for n = 1:numel(snrLevels)
        snrDb = snrLevels(n);
        inputName = sensitivity_file_name(specs(s).prefix, snrDb);
        inFile = fullfile(dataDir, inputName);
        if ~exist(inFile, 'file'), error('Missing input file: %s', inFile); end

        for q = 1:numel(multipliers)
            multiplier = multipliers(q);
            dilationSec = specs(s).defaultDilationSec * multiplier;
            settingKey = setting_key(specs(s).family, multiplier);
            outDir = fullfile(outRoot, settingKey);
            if ~exist(outDir, 'dir'), mkdir(outDir); end
            finalOutFile = fullfile(outDir, inputName);
            if exist(finalOutFile, 'file') && ~overwrite
                fprintf('Skipping existing output: %s\n', finalOutFile);
                continue;
            end

            tempOutDir = fullfile(tempdir, 'spar_dilation_sensitivity');
            if ~exist(tempOutDir, 'dir'), mkdir(tempOutDir); end
            outFile = [tempname(tempOutDir) '.h5'];

            h5writeatt_safe(outFile, '/', 'source_file', inputName);
            h5writeatt_safe(outFile, '/', 'test_family', specs(s).family);
            h5writeatt_safe(outFile, '/', 'dilation_multiplier', multiplier);
            h5writeatt_safe(outFile, '/', 'configured_dilation_sec', dilationSec);
            h5writeatt_safe(outFile, '/', 'submitted_default_dilation_sec', specs(s).defaultDilationSec);
            h5writeatt_safe(outFile, '/', 'fixed_depth', specs(s).depth);
            h5writeatt_safe(outFile, '/', 'fixed_zthr', 3);
            h5writeatt_safe(outFile, '/', 'nominal_snr_db', snrDb);
            h5writeatt_safe(outFile, '/', 'selection_manifest', manifestPath);
            h5writeatt_safe(outFile, '/', 'selected_record_count', numel(recordIds));
            h5writeatt_safe(outFile, '/', 'other_defaults', 'frozen_internal_defaults');

            fprintf('\n%s | SNR=%g dB | dilation=%g s (%gx) | records=%d\n', ...
                upper(specs(s).family), snrDb, dilationSec, multiplier, numel(recordIds));
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
                        cfg = struct('NumIMFs', specs(s).depth, ...
                            'mask_zthr', 3, 'mask_dilate_sec_base', dilationSec);
                        [y, debug] = burstdenoise.denoise_emg_vmd_window(x, fs, cfg);
                        realizedDepth = size(debug.U, 1);
                    else
                        cfg = struct('L_min', specs(s).depth, ...
                            'mask_zthr', 3, 'mask_dilate_sec_base', dilationSec);
                        [y, debug] = burstdenoise.denoise_eog_ssa_window(x, fs, cfg);
                        realizedDepth = debug.L;
                    end
                    elapsedSec = toc(ticId);
                    y = double(y(:));

                    nRegions = size(debug.regions, 1);
                    nClean = nnz(debug.clean_idx);
                    minBaseline = max(5, round(debug.cfg.min_baseline_sec * fs));
                    bypassNoRegion = nRegions < 1;
                    bypassInsufficientBaseline = nRegions >= 1 && nClean < minBaseline;

                    h5create(outFile, dsOut, size(y), 'Datatype', 'double');
                    h5write(outFile, dsOut, y);
                    h5writeatt_safe(outFile, dsOut, 'status', 'OK');
                    h5writeatt_safe(outFile, dsOut, 'fs', fs);
                    h5writeatt_safe(outFile, dsOut, 'test_family', specs(s).family);
                    h5writeatt_safe(outFile, dsOut, 'dilation_multiplier', multiplier);
                    h5writeatt_safe(outFile, dsOut, 'configured_dilation_sec', dilationSec);
                    h5writeatt_safe(outFile, dsOut, 'submitted_default_dilation_sec', specs(s).defaultDilationSec);
                    h5writeatt_safe(outFile, dsOut, 'fixed_depth', specs(s).depth);
                    h5writeatt_safe(outFile, dsOut, 'realized_depth', realizedDepth);
                    h5writeatt_safe(outFile, dsOut, 'fixed_zthr', debug.cfg.mask_zthr);
                    h5writeatt_safe(outFile, dsOut, 'detected_regions', nRegions);
                    h5writeatt_safe(outFile, dsOut, 'clean_samples', nClean);
                    h5writeatt_safe(outFile, dsOut, 'min_baseline_required', minBaseline);
                    h5writeatt_safe(outFile, dsOut, 'bypass_no_region', double(bypassNoRegion));
                    h5writeatt_safe(outFile, dsOut, 'bypass_insufficient_baseline', double(bypassInsufficientBaseline));
                    h5writeatt_safe(outFile, dsOut, 'initial_exceedance_fraction', ...
                        mean(debug.z(:) >= debug.cfg.mask_zthr));
                    h5writeatt_safe(outFile, dsOut, 'mask_fraction', mean(debug.m_common(:) > 0.5));
                    h5writeatt_safe(outFile, dsOut, 'attenuated_sample_fraction', attenuation_fraction(debug));
                    h5writeatt_safe(outFile, dsOut, 'minimum_realized_gain', minimum_gain(debug));
                    h5writeatt_safe(outFile, dsOut, 'output_change_rrmse', relative_change(x, y));
                    h5writeatt_safe(outFile, dsOut, 'elapsed_sec', elapsedSec);

                    if r == 1 || r == numel(recordIds) || mod(r, progressStep) == 0
                        fprintf('  [%d/%d] %s OK | regions=%d | support=%.3f | %.3f s\n', ...
                            r, numel(recordIds), recordName, nRegions, ...
                            attenuation_fraction(debug), elapsedSec);
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
            if ~copyOk, error('Could not copy completed H5 to %s | %s', finalOutFile, copyMsg); end
            delete(outFile);
            fprintf('Wrote: %s | failures=%d\n', finalOutFile, nFailed);
        end
    end
end

fprintf('\nDilation-sensitivity restoration finished.\n');
end

function fileName = sensitivity_file_name(prefix, snrDb)
if abs(snrDb - round(snrDb)) < 1e-12
    snrText = sprintf('%d', round(snrDb));
else
    snrText = sprintf('%g', snrDb);
end
fileName = sprintf('03_denoise-net_%s_%sdB.h5', prefix, snrText);
end

function key = setting_key(family, multiplier)
text = strrep(strrep(sprintf('%g', multiplier), '-', 'm'), '.', 'p');
key = sprintf('%s_d%s', family, text);
end

function frac = attenuation_fraction(debug)
if ~isfield(debug, 'W') || isempty(debug.W)
    frac = 0;
else
    frac = mean(any(debug.W < (1 - 1e-12), 1));
end
end

function value = minimum_gain(debug)
if ~isfield(debug, 'W') || isempty(debug.W)
    value = 1;
else
    value = min(debug.W(:));
end
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
