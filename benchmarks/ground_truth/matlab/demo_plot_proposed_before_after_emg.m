%% demo_plot_proposed_before_after_emg.m
% Plot original noisy EEG, clean reference, and proposed restored output
% for selected DenoiseNet EMG -20 dB examples.

clear; clc; close all;

%% Paths
rootDir = fileparts(fileparts(mfilename('fullpath')));

origFile = fullfile(rootDir, 'data', '03_denoise-net_emg_-20dB.h5');
restFile = fullfile(rootDir, 'results', 'proposed_restored_primary', '03_denoise-net_emg_-20dB.h5');

%% Records to plot
recordIds = [0 1 2 3 4];   % change as needed
fs = 256;

%% Plot
figure('Color','w','Position',[100 100 1200 800]);

for i = 1:numel(recordIds)
    recName = sprintf('emg_%d', recordIds(i));

    noisy = h5read(origFile, ['/' recName '/eeg_signal']);
    clean = h5read(origFile, ['/' recName '/eeg_reference']);
    mask  = h5read(origFile, ['/' recName '/artifacts']);
    restored = h5read(restFile, ['/' recName]);

    % Ensure row vectors
    noisy = double(noisy(:))';
    clean = double(clean(:))';
    restored = double(restored(:))';
    % mask = logical(mask(:))';
    % mask = logical(boolean(mask))';

    n = numel(noisy);
    t = (0:n-1) / fs;

    subplot(numel(recordIds), 1, i);
    hold on;

    % Shade artifact mask
    yl0 = [min([noisy clean restored]) max([noisy clean restored])];
    pad = 0.1 * max(diff(yl0), eps);
    yl = [yl0(1)-pad yl0(2)+pad];

    % runs = mask_to_runs(mask);
    % for r = 1:size(runs,1)
    %     x1 = t(runs(r,1));
    %     x2 = t(runs(r,2));
    %     patch([x1 x2 x2 x1], [yl(1) yl(1) yl(2) yl(2)], ...
    %           [1.0 0.85 0.25], ...
    %           'FaceAlpha', 0.25, ...
    %           'EdgeColor', 'none');
    % end

    plot(t, clean, 'b', 'LineWidth', 1.2);
    plot(t, noisy, 'r', 'LineWidth', 1.0);
    plot(t, restored, 'k', 'LineWidth', 1.1);

    inputSnr = h5readatt(origFile, ['/' recName], 'measured_snr_db');

    title(sprintf('%s | input SNR = %.2f dB', recName, inputSnr), ...
          'Interpreter','none');

    ylabel('Amplitude');
    ylim(yl);
    grid on;

    if i == 1
        % legend({'Artifact mask','Clean reference','Noisy input','Proposed restored'}, ...
               % 'Location','best');
        legend({'Clean reference','Noisy input','Proposed restored'}, ...
               'Location','best');
    end

    if i == numel(recordIds)
        xlabel('Time (s)');
    end
end

sgtitle('DenoiseNet EMG -20 dB: Proposed Denoising Before/After');

%% Helper function
function runs = mask_to_runs(mask)
    mask = logical(mask(:)');
    edges = diff([false mask false]);
    starts = find(edges == 1);
    ends = find(edges == -1) - 1;
    runs = [starts(:) ends(:)];
end
