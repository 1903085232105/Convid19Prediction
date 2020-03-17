import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters

from torch import nn, optim
import torch.nn.functional as F

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#93D30C", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

rcParams['figure.figsize'] = 12, 6
register_matplotlib_converters()

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

#load the data file
df = pd.read_csv('time_series_19-covid-Confirmed.csv')
#remove the first 4 coloum
df = df.iloc[:, 4:]

#print(df.isnull().sum())

daily_cases = df.sum(axis=0)
#print(daily_cases.head())
daily_cases.index = pd.to_datetime(daily_cases.index)
# print(daily_cases.head())


####ploting the Cumulative daily cases

# plt.plot(daily_cases)
# plt.title('Cumulative daily Cases')
# plt.show()

daily_cases = daily_cases.diff().fillna(daily_cases[0]).astype(np.int64)
# print(daily_cases.head())

# plt.plot(daily_cases)
# plt.title('daily Cases')
# plt.show()

#number of days
#print(daily_cases.shape)

##preprocessing

test_data_size = 14
train_data = daily_cases[:-test_data_size]
test_data = daily_cases[-test_data_size:]

#print(test_data.shape)

scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))
train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))

def sliding_windows(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)- seq_length -1):
        x = data[i: (i+ seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_lenght = 5
x_train,  y_train = sliding_windows(train_data, seq_lenght)
x_test,  y_test = sliding_windows(test_data, seq_lenght)

# print(x_train[:2])

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()


