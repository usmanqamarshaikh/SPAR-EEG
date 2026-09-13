function [y, debug] = spar_eeg_denoise(x, Fs, passes, cfg)
%SPAR_EEG_DENOISE Apply selected SPAR-EEG passes to one signal window.
%
%   y = spar_eeg_denoise(x, Fs)
%   y = spar_eeg_denoise(x, Fs, {'emg','eog','slow'})
%   [y, debug] = spar_eeg_denoise(x, Fs, passes, cfg)
%
% x must be a single-channel vector. cfg may contain optional emg, eog, and
% slow sub-structures that override the corresponding frozen pass defaults.

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

wasRow = isrow(x);
y = double(x(:));
debug = struct('pass_order', {passes}, 'passes', {cell(numel(passes), 1)});

for k = 1:numel(passes)
    passName = lower(char(passes{k}));
    passCfg = struct();
    if isfield(cfg, passName)
        passCfg = cfg.(passName);
    end

    switch passName
        case 'emg'
            [y, passDebug] = burstdenoise.denoise_emg_vmd_window(y, Fs, passCfg);
        case 'eog'
            [y, passDebug] = burstdenoise.denoise_eog_ssa_window(y, Fs, passCfg);
        case 'slow'
            [y, passDebug] = burstdenoise.denoise_slow_ssa_window(y, Fs, passCfg);
        otherwise
            error('spar_eeg_denoise:UnknownPass', 'Unknown pass: %s', passName);
    end
    y = double(y(:));
    debug.passes{k} = passDebug;
end

if wasRow
    y = y.';
end
end
