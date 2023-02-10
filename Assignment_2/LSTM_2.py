import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df = pd.read_csv("./data/power_usage_2016_to_2020.csv")

df["StartDate"] = pd.to_datetime(df.StartDate)

start_date_series = df["StartDate"]

date_df = pd.DataFrame(start_date_series)

df.set_index("StartDate", inplace=True)

dummies = pd.get_dummies(df.notes)

df = pd.concat([df[["Value (kWh)", "day_of_week"]], dummies], axis=1)

scaler = MinMaxScaler() 
data_scaled = scaler.fit_transform(df[["Value (kWh)", "day_of_week"]])
data_s = pd.DataFrame(data_scaled, columns=["Value (kWh)", "day_of_week"])

scaled = pd.concat([data_s, date_df], axis=1)

scaled.set_index("StartDate", inplace=True)

ready_df = pd.concat([scaled, df[["weekend", "weekday", "vacation", "COVID_lockdown"]]], axis=1)

energy_data_all = ready_df[["Value (kWh)", "weekend", "weekday", "vacation", "COVID_lockdown"]].to_numpy()
energy_data = ready_df[["Value (kWh)"]].to_numpy()

# Split the data into sequences for lstm input
sequence_length = 24*30 # To take a months data into consideration to predict future value 
sequence_data = []
sequence_labels = []
for i in range(len(energy_data) - sequence_length):
    sequence_data.append(energy_data[i:i+sequence_length])
    sequence_labels.append(energy_data[i+sequence_length])
sequence_data = np.array(sequence_data)
sequence_labels = np.array(sequence_labels)

training_ratio = int(.6*len(sequence_data))
print("Training length:", training_ratio)

train_data = sequence_data[:training_ratio]
train_labels = sequence_labels[:training_ratio]
test_data = sequence_data[training_ratio:]
test_labels = sequence_labels[training_ratio:]

# Split the data into training and testing sets
train_data = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
test_data = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Initialize the LSTM model
input_size = 1
hidden_size = 6
output_size = 1
learning_rate = 0.01

# 1. Creating an LSTM model
lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
fc = torch.nn.Linear(hidden_size, output_size)

