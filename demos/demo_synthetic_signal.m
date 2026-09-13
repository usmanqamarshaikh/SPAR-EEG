%DEMO_SYNTHETIC_SIGNAL Dataset-free SPAR-EEG demonstration.

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'matlab'));

Fs = 256;
t = (0:(8 * Fs - 1)) / Fs;
rng(7);
clean = 0.7 * sin(2 * pi * 10 * t) + 0.2 * randn(size(t));
blink = 5.0 * exp(-0.5 * ((t - 3.0) / 0.12).^2);
burstMask = t >= 5.2 & t <= 5.8;
emg = zeros(size(t));
emg(burstMask) = 2.0 * randn(1, nnz(burstMask));
noisy = clean + blink + emg;

[cleaned, debug] = spar_eeg_denoise(noisy, Fs);

figure('Color', 'w');
tiledlayout(2, 1, 'TileSpacing', 'compact');
nexttile;
plot(t, noisy, 'Color', [0.65 0.25 0.20]); hold on;
plot(t, cleaned, 'Color', [0.10 0.35 0.65]);
xlabel('Time (s)'); ylabel('Amplitude');
legend('Input', 'SPAR-EEG', 'Location', 'best');
title('Synthetic single-channel example');
nexttile;
plot(t, clean, 'Color', [0.20 0.20 0.20]); hold on;
plot(t, cleaned, 'Color', [0.10 0.35 0.65]);
xlabel('Time (s)'); ylabel('Amplitude');
legend('Known clean signal', 'SPAR-EEG', 'Location', 'best');
title(sprintf('Applied passes: %s', strjoin(debug.pass_order, ' -> ')));
