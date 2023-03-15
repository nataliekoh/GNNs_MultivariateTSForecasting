import numpy as np
import os
import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg

# data = np.load(os.path.join('data','METR-LA','test.npz'), allow_pickle=True)
# lst = data.files

# print(data['x'])
class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

def load_dataset(data_dir, batch_size, valid_batch_size= None, test_batch_size=None):
    data = {}
    for i,category in enumerate(['train', 'val', 'test']):
        cat_data = np.load(os.path.join(data_dir, category + '.npz'))
        # cat_data = dataset[i]
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def load_msense():
    file = 'data/C_Gyroscope_data'
    file2 = 'data/B_Accelerometer_data'
    d2 = '/jog_9/'
    d1 = '/wlk_7/'
    d_list = [d1,d2]
    x_data = []
    y_data = []
    for k in range(0, 3):
        sx_data = np.zeros((700, 3))
        sy_data = np.zeros((700,3))
        for f in [file, file2]:
            for y,c in enumerate(d_list):
                with open(f + c + 'sub_' + str(k+1) + '.csv') as d_file:
                    lines = d_file.readlines()
                    data = [line.split(",") for line in lines]
                    data.pop(0)
                    for d in data:
                        d[-1] = d[-1].replace('\n', '')
                        d.pop(0)
                        for i,v in enumerate(d):
                            d[i] = float(v)
                    data_y = np.array(data[700:1400])
                    data = np.array(data[:700])
                    sy_data = np.concatenate((sy_data, data_y), axis=1)           
                    sx_data = np.concatenate((sx_data, data), axis=1)
                    d_file.close()
            for z in range(0,3):
                sx_data = np.delete(sx_data, 0, 1)
                sy_data = np.delete(sy_data, 0, 1)
            print(sx_data)
            input()
            x_data.append(sx_data)
            y_data.append(sy_data)
    print(np.shape(x_data))
    return x_data, y_data


# from sklearn.model_selection import train_test_split
# x, x_test, y, y_test = train_test_split(x_data,y_data,test_size=0.1,train_size=0.9)
# x_train, x_val, y_train, y_val = train_test_split(x,y,test_size = 0.11,train_size =0.89)
# all_data = [[x_train, y_train], [x_val, y_val], [x_test, y_test]]

f = 'data/msense_trun'
data = load_dataset(f, 10, 10, 10)
print(data)
print(np.shape(data['x_train']))
print(np.shape(data['y_train']))

#x_train: (23974, 12, 207, 2)