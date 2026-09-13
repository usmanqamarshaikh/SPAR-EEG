repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'matlab'));

assert(~isempty(which('burstdenoise.denoise_emg_vmd_window')));
assert(~isempty(which('burstdenoise.denoise_eog_ssa_window')));
assert(~isempty(which('burstdenoise.denoise_slow_ssa_window')));
assert(exist('spar_eeg_denoise', 'file') == 2);
assert(exist('spar_eeg_adaptive', 'file') == 2);

Fs = 256;
rng(3);
x = 0.2 * randn(1, 2 * Fs) + sin(2 * pi * 10 * (0:(2 * Fs - 1)) / Fs);
for passName = {'emg', 'eog', 'slow'}
    y = spar_eeg_denoise(x, Fs, passName);
    assert(isequal(size(x), size(y)));
    assert(all(isfinite(y)));
end

yFull = spar_eeg_denoise(x, Fs);
assert(isequal(size(x), size(yFull)));
assert(all(isfinite(yFull)));

xContinuous = repmat(x, 1, 3);
[yAdaptive, meta] = spar_eeg_adaptive(xContinuous, Fs, {'eog'});
assert(isequal(size(xContinuous), size(yAdaptive)));
assert(all(isfinite(yAdaptive)));
assert(~isempty(meta.segment_duration_sec));
fprintf('MATLAB smoke test passed for all passes and adaptive streaming.\n');
