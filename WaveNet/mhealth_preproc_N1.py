import argparse
import numpy as np
import os
import pandas as pd

import os, torch 
import pandas as pd
import numpy as np
mhealth_dataset = []
for path, dir, file in os.walk("data/MHEALTHDATASET"):
    for fileNames in file:
        if fileNames.endswith("log"):
            fileName = str(os.path.join(path,fileNames))
            tmpData = pd.read_csv(fileName, sep='\t', engine='python')
            np.shape(tmpData.to_numpy()) # time x variables
            mhealth_dataset.append(tmpData.to_numpy())

            tcat = []
for i in range(len(mhealth_dataset)):
  # Truncate all time series to the same length
  t = torch.from_numpy(mhealth_dataset[i][:98303,:])
  tcat.append(t)
tcat = torch.stack(tcat, dim=0)
tcat.size() # subjects x time x var
mhealth = tcat.numpy()
print("Raw 3D time series shape: {0}".format(mhealth.shape))
mhealth_reshaped = np.transpose(mhealth, (1, 0, 2))
print("New 3D time series shape: {0}".format(mhealth_reshaped.shape))

# Truncate the dataset
timepoints_to_keep = 30000 # 3/5 of a min
nsubjects = 10
features_to_keep = 5
truncated_mhealth = mhealth_reshaped[:timepoints_to_keep, :nsubjects, :features_to_keep]
print("Truncated time series shape: {0}".format(truncated_mhealth.shape))

# Rearrange to time x features x subjects

truncated_mhealth_reshaped = np.transpose(truncated_mhealth, axes=[0, 2, 1])
print("Truncated mhealth reshaped: {0}".format(truncated_mhealth_reshaped.shape))

def generate_graph_seq2seq_io_data(df, x_offsets, y_offsets, scaler=None):
    """
    Generate samples from
    :param df:
    :param x_offsets:
    :param y_offsets:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """

    num_samples, n, f = df.shape
    num_nodes = n*f
    data = df
    feature_list = [data]

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

# Set up parameters (this should be as similar as the input to STEP as possible)

# *** NOTE: WAVENET NEEDS SYMMETRICAL SEQ LENGTHS! ***

seq_length_x = 12
seq_length_y = 12
y_start = 1
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 1 - train_ratio - valid_ratio
output_dir = 'data/mhealth_trun'
name = ''

df = truncated_mhealth 
x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))
x, y = generate_graph_seq2seq_io_data(
    df,
    x_offsets=x_offsets,
    y_offsets=y_offsets
)
print("x shape: ", x.shape, ", y shape: ", y.shape)

# Write the data into npz file.
num_samples = x.shape[0]
num_test = round(num_samples * test_ratio)
num_train = round(num_samples * train_ratio)
num_val = num_samples - num_test - num_train
x_train, y_train = x[:num_train], y[:num_train]
x_val, y_val = (
    x[num_train: num_train + num_val],
    y[num_train: num_train + num_val],
)
x_test, y_test = x[-num_test:], y[-num_test:]

for cat in ["train", "val", "test"]:
    _x, _y = locals()["x_" + cat], locals()["y_" + cat]
    print(cat, "x: ", _x.shape, "y:", _y.shape)
    filename = cat
    np.savez_compressed(
        os.path.join(output_dir, f"{filename}.npz"),
        x=_x,
        y=_y,
        x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
        y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
    )