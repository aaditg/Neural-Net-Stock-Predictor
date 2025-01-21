from NeuralNetStructure import NeuralNetwork
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

def calculate_slope(points):
    time_indices = np.arange(len(points)).reshape(-1, 1)
    model = LinearRegression().fit(time_indices, points)
    return model.coef_[0]

symbol = 'SPY'
data = yf.download(symbol, period="5d", interval="1m")

def amplify_slope(data, amplification_factor=5):
    time_indices = np.arange(len(data)).reshape(-1, 1)
    model = LinearRegression()
    model.fit(time_indices, data)
    slope = model.coef_[0]
    amplified_data = data + (data - model.predict(time_indices)) * amplification_factor
    return amplified_data

close_prices = data['Close']
data['hour'] = data.index.hour
data['minute'] = data.index.minute

first_16_points = []
slopes = []

for hour in range(24):
    hour_data = data[data['hour'] == hour]
    if len(hour_data) >= 32:
        first_16 = hour_data.iloc[:16]['Close'].values
        second_16 = hour_data.iloc[16:32]['Close'].values
        slope = calculate_slope(second_16)
        first_16_points.append(first_16)
        slopes.append(slope)

first_16_points = np.array(first_16_points)
first_16_points = np.squeeze(first_16_points)
slopes = np.array(slopes).reshape(-1, 1)

# Scale both inputs and outputs
scaler = MinMaxScaler()
first_16_points_scaled = scaler.fit_transform(first_16_points)
slopes_scaled = scaler.fit_transform(slopes)

lr = 0.1
epochs = 10000
hn = 32

nn = NeuralNetwork(input_nodes=16, hidden_nodes=hn, output_nodes=1, learning_rate=lr)

losses = nn.train(first_16_points_scaled, slopes_scaled, iterations=epochs)

predicted_slopes = []
for i in range(len(first_16_points_scaled)):
    predicted_slope = nn.forward_propagation(first_16_points_scaled[i].reshape(1, -1))
    predicted_slopes.append(predicted_slope)

predicted_slopes = np.array(predicted_slopes).reshape(-1, 1)  # Reshape to 2D array
predicted_slopes = scaler.inverse_transform(predicted_slopes)  # Inverse transform to get actual values

print(f"Predicted slopes: {predicted_slopes}")
print(f"Actual slopes: {slopes.flatten()}")
print(f"Learning rate:  {lr}")
print(f"Iterations:  {epochs}")
print(f"Hidden Nodes:  {hn}")

mse = np.mean((predicted_slopes - slopes.flatten())**2)
print(f"Mean Squared Error: {mse}")

nn.plot_loss(losses)