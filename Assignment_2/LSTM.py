import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pylab as pl

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

criterion = torch.nn.MSELoss()
optimizer = optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the lstm model
n_total_steps = len(train_loader)

num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    train_loss = 0

    for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()

        # Run the forward pass
        hidden = (torch.zeros(1, data.size(0), hidden_size),
                  torch.zeros(1, data.size(0), hidden_size))
        output, hidden = lstm(data.float(), hidden)
        output = fc(hidden[0][-1])

        # 2. Network Evaluation
        loss = criterion(output, label.float())

        # 3. Gradient Calculation
        loss.backward()

        # 4. Back Propagation
        optimizer.step()
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        train_loss += loss.item()

    # Calculate the average training loss
    train_loss = train_loss / len(train_loader)
    train_losses.append(train_loss)

# Plot the MSE loss for each epoch
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss over Epochs')
plt.show()

# Evaluate the lstm model
mse = 0

with torch.no_grad():
    for data, label in test_loader:
        hidden = (torch.zeros(1, data.size(0), hidden_size),
                  torch.zeros(1, data.size(0), hidden_size))
        output, hidden = lstm(data.float(), hidden)
        output = fc(hidden[0][-1])

        mse += ((output - label)**2).mean().item()

mse /= len(test_loader)
print('Test MSE: {}'.format(mse))


torch.save(lstm.state_dict(), "lstm1.pt")

lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
lstm.load_state_dict(torch.load("lstm1.pt"))

predictions = []
# Initialize the hidden state
h0 = torch.zeros(1,  hidden_size)
   

for i in range(train_loader.size(0)):
   input = train_loader[i:i+1]
   # Forward pass
   out, hn = lstm(input, h0)
   # Pass the hidden state through the output layer
   y_pred = fc(hn.squeeze(0))
   predictions.append(y_pred.data.numpy().ravel()[0])

data_time_steps = np.linspace(2, 10, train_loader.shape[0])

pl.scatter(data_time_steps[:], test_loader.data.numpy(), s = 90, label = "Actual")
pl.scatter(data_time_steps[:], predictions, label = "Predicted")
pl.legend()
pl.show()