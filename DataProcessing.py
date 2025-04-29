import copy
from pandas import read_parquet
import numpy
import torch
from torch.utils.data import Dataset
import numpy as np


# Dataset Class
class CustomTSDataset(Dataset):
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Scale numerical
def scaler(in_array: np.array, column, scale_type) -> np.array:
    if column in in_array.columns:
        if scale_type == 'minmax':
            in_array[column] = (in_array[column] - np.min(in_array[column])) / (
                    np.max(in_array[column]) - np.min(in_array[column]))
        elif scale_type == 'std':
            in_array[column] = (in_array[column] - np.mean(in_array[column])) / np.std(in_array[column])
    return in_array


# Encode Categorical
# def encoding(data_in: np.array, encode_type) -> np.array:


# Missing data numerical
def fill_empty(data: np.array, debug: bool) -> np.array:
    if debug:
        print(f"nan count before filling: {data.isna().sum()}")
    data = copy.deepcopy(data)
    data.interpolate(inplace=True)
    data.dropna(how='any', inplace=True)
    if debug:
        print(f"nan count after filling: {data.isna().sum()}")
    return data


def create_windowed_dataset(data: np.array, input_window: int, output_window: int, stride=1) -> np.array:
    X, y = [], []
    L = len(data)
    for i in range(0, L - input_window - output_window + 1, stride):
        x = data[i:(i + input_window)]
        target = data[(i + input_window):(i + input_window + output_window)]
        X.append(x)
        y.append(target)
    return np.array(X), np.array(y)


# Feature creation
def feature_creation(data: np.array, tgt_cols: list[str], roll_amt=8) -> np.array:
    # Create date time features
    data = copy.deepcopy(data)
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['month'] = data.index.month
    data['day_of_year'] = data.index.dayofyear
    data['dt'] = data.index

    # Lagged features
    lags = [1, 24]
    for col in tgt_cols:
        for lag in lags:
            temp = data[col].shift(periods=lag)
            temp = temp.to_frame()
            temp.rename(columns={f"{col}": f"{col}_{lag}"}, inplace=True)
            temp['dt'] = temp.index
            data = data.merge(temp, on='dt', how='outer')
            data['DateTime'] = data['dt'].copy()
            data.set_index('DateTime', inplace=True, drop=True, )
    data = data.dropna(axis=0, subset=['AEP'])
    # Rolling windows
    for col in tgt_cols:
        temp = data[col].rolling(roll_amt).mean()
        temp = temp.to_frame()
        temp.rename(columns={f"{col}": f"{col}_{roll_amt}_mean"}, inplace=True)
        temp['dt'] = temp.index
        data = data.merge(temp, on='dt', how='outer')
        data['DateTime'] = data['dt'].copy()
        data.set_index('DateTime', inplace=True, drop=True, )

    return data


# Missing data categorical

# Data Resampling

# Train Test split

# Np array to tensor
def get_data():
    # Create dataset
    parkdata = read_parquet('Data/archive/est_hourly.paruqet')
    parkdata.drop(['NI', 'PJM_Load'], inplace=True, axis=1)
    parkdata = parkdata.sort_index()
    filled_data = fill_empty(parkdata, debug=False)
    filled_data_feat = feature_creation(filled_data, ['AEP'], roll_amt=4)
    filled_data_feat = fill_empty(filled_data_feat, debug=False)

    return filled_data_feat
