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
### model##

class CoronaVirusPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, seq_len,num_layer=2):
        super(CoronaVirusPredictor, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim =hidden_dim
        self.seq_len = seq_len
        self.num_layer = num_layer
        #long short term memory
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layer,
            dropout=0.5
        )

        self.linear = nn.Linear(in_features=hidden_dim, out_features=1)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.num_layer, self.seq_len, self.hidden_dim),
            torch.zeros(self.num_layer, self.seq_len, self.hidden_dim)
        )
    def forward(self, input):
        lstm_out, self.hidden = self.lstm(
            input.view(len(input), self.seq_len, -1),
            self.hidden
        )
        y_pred = self.linear(
            lstm_out.view(self.seq_len, len(input), self.hidden_dim)[-1]
        )
        return y_pred

def train_model(model, traning_data,training_labels, test_data=None, test_labels=None):
    loss_fn = nn.MSELoss(reduction='sum')

    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    num_epoc = 60

    train_hist = np.zeros(num_epoc)
    test_hist = np.zeros(num_epoc)

    for t in range(num_epoc):
        model.reset_hidden_state()

        y_pred = model(x_train)

        loss = loss_fn(y_pred.float(), y_train)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(x_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(f'Epoc {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 ==0 :
            print(f'Epoc {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()
    return model.eval(), train_hist, test_hist

model = CoronaVirusPredictor(1, 512, seq_len = seq_lenght,  num_layer=2)

model, train_hist,  test_hist = train_model(model, x_train,y_train, x_test, y_test)

# plt.plot(train_hist, label='Training loss')
# plt.plot(test_hist, label='Test loss')
# plt.ylim(0,5)
# plt.legend()
# plt.show()


###predicting daily cases

with torch.no_grad():
    test_seq = x_test[:1]
    preds = []
    for _ in  range(len(x_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)

        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_lenght, 1).float()


true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases = scaler.inverse_transform(
    np.expand_dims(preds, axis=0)
).flatten()


plt.plot(daily_cases.index[:len(train_data)],
         scaler.inverse_transform(train_data).flatten(),
         label='Historical daily Cases')
plt.legend()
plt.show()