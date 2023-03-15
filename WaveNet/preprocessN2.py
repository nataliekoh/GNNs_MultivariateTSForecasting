# Set up parameters (this should be as similar as the input to STEP as possible)

# *** NOTE: WAVENET NEEDS SYMMETRICAL SEQ LENGTHS! ***
import numpy as np
import torch
import pandas as pd
import os
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

sub_ids = [10]
def get_ds_infos():
    """ Gets inforamtion about data subjects' attributes""" 
    dss = pd.read_csv("data/data_subjects_info.csv")
    return dss

def set_data_types(data_types=["userAcceleration"]):
    """ Select the sensors and the mode to shape the final dataset. """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])
    return dt_list

def creat_time_series(dt_list, act_labels, trial_codes, mode="mag", labeled=False):
    """ It returns a time-series of sensor data. """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0, num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0, num_data_cols))
        
    ds_list = get_ds_infos()
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in sub_ids:
        for act_id, act in enumerate(act_labels): # for each activity
            for trial in trial_codes[act_id]: # for each variable
                fname = 'data/A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data))
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset, vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset

ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
TRIAL_CODES = {
    ACT_LABELS[0]:[1,2,11],
    ACT_LABELS[1]:[3,4,12],
    ACT_LABELS[2]:[7,8,15],
    ACT_LABELS[3]:[9,16],
    ACT_LABELS[4]:[6,14],
    ACT_LABELS[5]:[5,13]
}

# Here we set parameter to build labeld time-series from dataset of "(A)DeviceMotion_data"
# attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)
sdt = ["userAcceleration",'gravity','rotationRate']
print("[INFO] -- Selected sensor data types: "+str(sdt))    
act_labels = ACT_LABELS[2:4]
print("[INFO] -- Selected activites: "+str(act_labels))    
trial_codes = [TRIAL_CODES[act] for act in act_labels]
dt_list = set_data_types(sdt)
dataset = creat_time_series(dt_list, act_labels, trial_codes, mode="raw", labeled=True)
print("[INFO] -- Shape of time-Series dataset:"+str(dataset.shape))  

truncated_msense = []
features_to_keep = [1,2,3,4,6]
for sub_id in [9]:
  sub_dataset = dataset[dataset["id"].isin([float(sub_id)])]
  sub_tensor = torch.tensor(sub_dataset.values)
  sub_tensor = sub_tensor[0:18836, features_to_keep]
  sub_tensor_numpy = sub_tensor.numpy()
  truncated_msense.append(sub_tensor_numpy)
  
truncated_msense = np.stack(truncated_msense, axis=1)
print("Msense data shape:" + str(truncated_msense.shape)) # timesteps x features

seq_length_x = 12
seq_length_y = 12
y_start = 1
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 1 - train_ratio - valid_ratio
output_dir = 'data/msense_trun'
name = ''

df = truncated_msense
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