function benchmark_proposed_cost(dataDir, outDir, pattern, maxRecords, passPolicies, repeats, packageParent)
%BENCHMARK_PROPOSED_COST Measure algorithm-only runtime for proposed passes.
%
% Timed sections exclude H5 output writing and metric computation. MATLAB is
% already running when timing starts, so MATLAB startup cost is also excluded.

if nargin < 1 || isempty(dataDir), dataDir = 'data'; end
if nargin < 2 || isempty(outDir), outDir = fullfile('results', 'computation_cost'); end
if nargin < 3 || isempty(pattern), pattern = '03_denoise-net_*_-10dB.h5'; end
if nargin < 4 || isempty(maxRecords), maxRecords = 100; end
if nargin < 5 || isempty(passPolicies)
    passPolicies = {'emg', 'eog', 'slow', 'emg+eog', 'full'};
end
if nargin < 6 || isempty(repeats), repeats = 3; end
if nargin < 7 || isempty(packageParent)
    thisDir = fileparts(mfilename('fullpath'));
    packageParent = fullfile(thisDir, '..', '..', '..', 'matlab');
end

dataDir = char(dataDir);
outDir = char(outDir);
pattern = char(pattern);
packageParent = char(packageParent);
if ischar(maxRecords) || isstring(maxRecords), maxRecords = str2double(maxRecords); end
if ischar(repeats) || isstring(repeats), repeats = str2double(repeats); end
passPolicies = normalize_policy_list(passPolicies);

if ~exist(outDir, 'dir'), mkdir(outDir); end
addpath(packageParent);

files = dir(fullfile(dataDir, pattern));
if isempty(files)
    error('No H5 files found for pattern: %s', fullfile(dataDir, pattern));
end

fprintf('Proposed computation-cost benchmark\n');
fprintf('  dataDir      : %s\n', dataDir);
fprintf('  outDir       : %s\n', outDir);
fprintf('  pattern      : %s\n', pattern);
fprintf('  maxRecords   : %g per file\n', maxRecords);
fprintf('  repeats      : %g\n', repeats);
fprintf('  passPolicies : %s\n', strjoin(passPolicies, ', '));
fprintf('  packageParent: %s\n', packageParent);

engine = {};
methodKey = {};
methodLabel = {};
sourceFile = {};
recordName = {};
repeatNum = [];
status = {};
elapsedSec = [];
msPerEpoch = [];
durationSec = [];
realtimeFactor = [];
fsCol = [];
nChannels = [];
nSamples = [];
outShape = {};
passOrderCol = {};

for f = 1:numel(files)
    inFile = fullfile(files(f).folder, files(f).name);
    info = h5info(inFile);
    nUse = min(numel(info.Groups), maxRecords);

    for p = 1:numel(passPolicies)
        policy = passPolicies{p};
        [passOrder, key, label] = pass_policy_to_order(policy);
        fprintf('\n%s | records=%d/%d | policy=%s | passes=%s\n', ...
            files(f).name, nUse, numel(info.Groups), policy, strjoin(passOrder, '+'));

        if nUse > 0
            try
                Xwarm = read_record_matrix(inFile, info.Groups(1).Name);
                fsWarm = read_freq(inFile, info.Groups(1).Name);
                denoise_matrix_columns(Xwarm, fsWarm, passOrder);
            catch ME
                warning('Warm-up failed for %s/%s: %s', files(f).name, policy, ME.message);
            end
        end

        for r = 1:nUse
            groupName = info.Groups(r).Name;
            recName = erase(groupName, '/');
            X = read_record_matrix(inFile, groupName);
            fs = read_freq(inFile, groupName);
            dur = size(X, 1) / double(fs);

            for rep = 1:repeats
                thisStatus = 'OK';
                thisShape = '';
                tStart = tic;
                try
                    Y = denoise_matrix_columns(X, fs, passOrder);
                    thisShape = mat2str(size(Y));
                catch ME
                    thisStatus = ['FAIL: ' ME.message];
                end
                dt = toc(tStart);

                engine{end+1, 1} = 'matlab'; %#ok<AGROW>
                methodKey{end+1, 1} = key; %#ok<AGROW>
                methodLabel{end+1, 1} = label; %#ok<AGROW>
                sourceFile{end+1, 1} = files(f).name; %#ok<AGROW>
                recordName{end+1, 1} = recName; %#ok<AGROW>
                repeatNum(end+1, 1) = rep; %#ok<AGROW>
                status{end+1, 1} = thisStatus; %#ok<AGROW>
                elapsedSec(end+1, 1) = dt; %#ok<AGROW>
                msPerEpoch(end+1, 1) = 1000 * dt; %#ok<AGROW>
                durationSec(end+1, 1) = dur; %#ok<AGROW>
                realtimeFactor(end+1, 1) = dt / dur; %#ok<AGROW>
                fsCol(end+1, 1) = fs; %#ok<AGROW>
                nChannels(end+1, 1) = size(X, 2); %#ok<AGROW>
                nSamples(end+1, 1) = size(X, 1); %#ok<AGROW>
                outShape{end+1, 1} = thisShape; %#ok<AGROW>
                passOrderCol{end+1, 1} = strjoin(passOrder, '+'); %#ok<AGROW>

                fprintf('  [%d/%d] %s rep %d/%d %.4f s\n', r, nUse, recName, rep, repeats, dt);
            end
        end
    end
end

raw = table(engine, methodKey, methodLabel, sourceFile, recordName, repeatNum, ...
    status, elapsedSec, msPerEpoch, durationSec, realtimeFactor, fsCol, ...
    nChannels, nSamples, outShape, passOrderCol, ...
    'VariableNames', {'engine', 'method_key', 'method', 'source_file', 'record', ...
    'repeat', 'status', 'elapsed_sec', 'ms_per_epoch', 'duration_sec', ...
    'realtime_factor', 'fs', 'n_channels', 'n_samples', 'out_shape', 'pass_order'});

rawPath = fullfile(outDir, 'computation_cost_proposed_raw.csv');
summaryPath = fullfile(outDir, 'computation_cost_proposed_summary.csv');
writetable(raw, rawPath);
writetable(summarize_timings(raw), summaryPath);

fprintf('\nSaved raw timings: %s\n', rawPath);
fprintf('Saved summary    : %s\n', summaryPath);
end

function policies = normalize_policy_list(passPolicies)
if ischar(passPolicies) || isstring(passPolicies)
    txt = char(passPolicies);
    if contains(txt, ',')
        parts = split(string(txt), ',');
        policies = cellstr(strtrim(parts));
    else
        policies = {strtrim(txt)};
    end
elseif iscell(passPolicies)
    policies = cellfun(@char, passPolicies, 'UniformOutput', false);
else
    error('Unsupported passPolicies type.');
end
policies = policies(~cellfun(@isempty, policies));
end

function X = read_record_matrix(inFile, groupName)
X = double(h5read(inFile, [groupName '/eeg_signal']));
if isvector(X), X = X(:); end
end

function Y = denoise_matrix_columns(X, fs, passOrder)
X = double(X);
[nSamp, nChan] = size(X);
Y = zeros(nSamp, nChan);
for ch = 1:nChan
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

function [passOrder, key, label] = pass_policy_to_order(policy)
policy = lower(strtrim(char(policy)));
switch policy
    case 'emg'
        passOrder = {'emg'};
        key = 'proposed_emg';
        label = 'Proposed EMG only';
    case 'eog'
        passOrder = {'eog'};
        key = 'proposed_eog';
        label = 'Proposed EOG only';
    case 'slow'
        passOrder = {'slow'};
        key = 'proposed_slow';
        label = 'Proposed slow only';
    case {'emg+eog', 'emg,eog'}
        passOrder = {'emg', 'eog'};
        key = 'proposed_emg_eog';
        label = 'Proposed EMG+EOG';
    case {'full', 'emg+eog+slow', 'emg,eog,slow'}
        passOrder = {'emg', 'eog', 'slow'};
        key = 'proposed_full';
        label = 'Proposed full';
    otherwise
        error('Unknown pass policy: %s', policy);
end
end

function fs = read_freq(inFile, groupName)
fs = 256;
try
    fs = double(h5readatt(inFile, groupName, 'freq'));
catch
end
end

function summary = summarize_timings(raw)
ok = raw(strcmp(raw.status, 'OK'), :);
if isempty(ok)
    summary = table();
    return;
end

[groups, keys, labels, engines] = findgroups(ok.method_key, ok.method, ok.engine);
nG = max(groups);
rows = cell(nG, 1);
for g = 1:nG
    idx = groups == g;
    ms = ok.ms_per_epoch(idx);
    rtf = ok.realtime_factor(idx);
    recPairs = strcat(ok.source_file(idx), '/', ok.record(idx));
    rows{g} = {engines{g}, keys{g}, labels{g}, sum(idx), numel(unique(recPairs)), ...
        median(ms, 'omitnan'), prctile(ms, 25), prctile(ms, 75), mean(ms, 'omitnan'), ...
        median(rtf, 'omitnan'), prctile(rtf, 25), prctile(rtf, 75)};
end

summary = cell2table(vertcat(rows{:}), 'VariableNames', ...
    {'engine', 'method_key', 'method', 'n_timed_runs', 'n_records', ...
    'median_ms_per_epoch', 'q1_ms_per_epoch', 'q3_ms_per_epoch', ...
    'mean_ms_per_epoch', 'median_realtime_factor', 'q1_realtime_factor', ...
    'q3_realtime_factor'});
end
