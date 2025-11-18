import os
import numpy as np

script_dir = os.path.dirname(__file__)
inputs_features = np.load(os.path.join(script_dir, 'input_features.npy'), allow_pickle=True)
intensity_labels = np.load(os.path.join(script_dir, 'intensity_targets.csv'), allow_pickle=True)
