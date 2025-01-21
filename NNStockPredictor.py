from NeuralNetStructure import NeuralNetwork
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# TEST THE MODEL
'''inputs = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
outputs = np.array([[1], [1], [0], [0]])


nn = NeuralNetwork(input_nodes=2, hidden_nodes=8, output_nodes=1, learning_rate=0.03)

losses = nn.train(inputs, outputs, iterations=5000)

nn.plot_loss(losses)

predictions = nn.forward_propogation(inputs)
print("Predictions:")
print(predictions)
'''


def calculate_slope(points):
    time_indices = np.arange(len(points)).reshape(-1, 1)
    model = LinearRegression().fit(time_indices, points)
    return model.coef_[0]


symbol = 'SPY'
data = yf.download(symbol, period="5d", interval="1m")

def amplify_slope(data, amplification_factor=5):

    #Amplifies the slope of the data by applying a linear transformation.

    time_indices = np.arange(len(data)).reshape(-1, 1)
    
    # Fit linear regression to get the slope
    model = LinearRegression()
    model.fit(time_indices, data)
    
    # Get the current slope
    slope = model.coef_[0]
    
    # Amplify the slope by the factor
    amplified_data = data + (data - model.predict(time_indices)) * amplification_factor
    
    return amplified_data


close_prices = data['Close']
data['hour'] = data.index.hour
data['minute'] = data.index.minute

first_16_points = []
slopes = []

for hour in range(24):
    hour_data = data[data['hour'] == hour]
    
    if len(hour_data) >= 32:  # Ensure there are at least 32 data points for this hour
        first_16 = hour_data.iloc[:16]['Close'].values
        second_16 = hour_data.iloc[16:32]['Close'].values
        
        # Calculate the slope of the second 16 points
        slope = calculate_slope(second_16)
        

        first_16_points.append(first_16)
        slopes.append(slope)


first_16_points = np.array(first_16_points)
first_16_points = np.squeeze(first_16_points) #FORMAT
#first_16_points = amplify_slope(first_16_points) #AMPLIFY
slopes = np.array(slopes).reshape(-1, 1)
scaler = MinMaxScaler()
first_16_points_scaled = scaler.fit_transform(first_16_points.reshape(-1, 1)).reshape(first_16_points.shape) #FIT BETWEEN 0 AND 1

#Tuning
lr = 0.1
epochs = 10000
hn = 32

nn = NeuralNetwork(input_nodes=16, hidden_nodes=hn, output_nodes=1, learning_rate=lr)

losses = nn.train(first_16_points, slopes, iterations=epochs)

predicted_slopes = []
for i in range(len(first_16_points)):
    predicted_slope = nn.forward_propogation(first_16_points[i].reshape(1, -1))
    predicted_slopes.append(predicted_slope)


predicted_slopes = np.array(predicted_slopes)
print(f"Predicted slopes: {predicted_slopes}")
print(f"Actual slopes: {slopes.flatten()}")
print(f"Learning rate:  {lr}")
print(f"Iterations:  {epochs}")
print(f"Hidden Nodes:  {hn}")


mse = np.mean((predicted_slopes - slopes.flatten())**2)
print(f"Mean Squared Error: {mse}")

nn.plot_loss(losses)