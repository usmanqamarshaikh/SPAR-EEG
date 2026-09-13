function EEGout = demo_eeglab_struct(EEG, channelIndices, useAdaptive)
%DEMO_EEGLAB_STRUCT Apply SPAR-EEG to selected channels of an EEGLAB struct.
%
% EEGout = demo_eeglab_struct(EEG, [1 2], true)

if nargin < 2 || isempty(channelIndices)
    channelIndices = 1:size(EEG.data, 1);
end
if nargin < 3 || isempty(useAdaptive)
    useAdaptive = true;
end
assert(isfield(EEG, 'data') && isfield(EEG, 'srate'), ...
    'Input must be an EEGLAB EEG structure with data and srate fields.');
assert(ndims(EEG.data) == 2, ...
    'This demonstration expects continuous channels-by-samples EEG data.');

repoRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoRoot, 'matlab'));
EEGout = EEG;

for channelIndex = channelIndices(:).'
    x = double(EEG.data(channelIndex, :));
    if useAdaptive
        y = spar_eeg_adaptive(x, EEG.srate);
    else
        y = spar_eeg_denoise(x, EEG.srate);
    end
    EEGout.data(channelIndex, :) = cast(y, 'like', EEG.data);
end

if isfield(EEGout, 'setname')
    EEGout.setname = [char(EEGout.setname), '_SPAR_EEG'];
end
end
