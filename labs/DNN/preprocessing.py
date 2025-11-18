from progressbar import printProgressBar

import os
from glob import glob
from dataclasses import dataclass

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import librosa
import librosa.display
import IPython.display as ipd
from itertools import cycle

sns.set_theme(style="white", palette=None)
color_pal = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])


# File Paths

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
# Other Project Paths
root_dir = os.path.join(script_dir, '..', '..')
training_data_dir = os.path.join(root_dir, 'training data')
categories_dir = os.path.join(training_data_dir, 'categories')
cremad_dir = os.path.join(training_data_dir, 'files', 'CREMA-D')
emogator_dir = os.path.join(training_data_dir, 'files', 'Emogator', 'data', 'mp3')
datasheeet_path = os.path.join(categories_dir, 'data.xlsx')


# Read Data
audio_files = []
data_raw_df = pd.read_excel(datasheeet_path)
headers = data_raw_df.columns.values.tolist()
data_raw_noheaders_df = data_raw_df.values
data_df = pd.DataFrame(data_raw_noheaders_df)

# Extract Targets
emotion_target_categories = headers[3:11]
intensity_target_categories = headers[12:]
selected_emotion_targets_df = data_df.iloc[:, [i for i in range(3, 11)]]
selected_intensity_targets_df = data_df.iloc[:, [i for i in range(12, 15)]]
emotion_targets = selected_emotion_targets_df.to_numpy()
intensity_targets = selected_intensity_targets_df.to_numpy()

# Load the audio files
datasets = data_raw_df['Dataset'].values
files = data_raw_df['File'].values
num_loaded = 0
for dataset, file in zip(datasets, files):
    if dataset == 'CREMA-D':
        file_path = os.path.join(cremad_dir, file)
        audio_files.append(file_path)
        num_loaded += 1
    elif dataset == 'EmoGator':
        file_path = os.path.join(emogator_dir, file)
        audio_files.append(file_path)
        num_loaded += 1
print(f'Loaded {num_loaded} files from dataset.')

# Convert audio file to features (log mel spectogram and log mel delta)
num_loaded = 0
target_frames = 86  # ≈ 2 seconds with librosa defaults (sr=22050, hop_length=512)
printProgressBar(0, len(audio_files), prefix='Progress:', suffix='Complete', length=50)

input_features = np.empty((len(audio_files), 2), dtype=object)

for afile in audio_files:
    # Extract y = (the raw data), and sr = (integer value of sample rate)
    y, sr = librosa.load(afile)
    # Apply STFT
    D = librosa.stft(y)
    # Retreive Mel
    S = librosa.feature.melspectrogram(y=y,
                                       sr=sr,
                                       n_mels=128 * 2,)
    S_decible_mel = librosa.amplitude_to_db(S, ref=np.max)
    # Extract Log Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # Extract Delta Mel spectrogram
    delta_log_mel_spectrogram = librosa.feature.delta(log_mel_spectrogram)
    input_features[num_loaded, 0] = log_mel_spectrogram
    input_features[num_loaded, 1] = delta_log_mel_spectrogram
    # sample_features = np.stack([(log_mel_spectrogram.T, delta_log_mel_spectrogram.T)], axis=-1)
    # input_features.append(sample_features)
    num_loaded += 1
    printProgressBar(
        num_loaded,
        len(audio_files),
        prefix='Progress:',
        suffix=f'  [{num_loaded}/{len(audio_files)}]',
        length=50
    )


# Save file to avoid preprocessing more
# np.savetxt('input_features.json', input_features, delimiter=',', fmt='%d', comments='')
np.savetxt('emotion_targets.csv', emotion_targets, delimiter=',', fmt='%d', comments='')
np.savetxt('intensity_targets.csv', intensity_targets, delimiter=',', fmt='%d', comments='')
input_features_array = np.array(input_features, dtype=object)
np.save('input_features.npy', input_features_array)
print(f'Wrote numpy arrays to file for training in {script_dir}')
print('Completed preprocessing!')
