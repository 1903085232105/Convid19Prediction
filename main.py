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
# print(daily_cases.head())
daily_cases.index = pd.to_datetime(daily_cases.index)
# print(daily_cases.head())

plt.plot(daily_cases)
plt.title('Cumulative daily Cases')
plt.show()



