from NeuralNetStructure import NeuralNetwork
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

class StockPredictor:

    def __init__(self, symbol, lr=0.1, epochs=10000, hn=32):
        self.symbol = symbol
        self.lr = lr
        self.epochs = epochs
        self.hn = hn
        self.input_scaler = MinMaxScaler()
        self.output_scaler = MinMaxScaler()
        self.nn = NeuralNetwork(input_nodes=16, hidden_nodes=self.hn, output_nodes=1, learning_rate=self.lr, scaler=self.input_scaler)
        self.predicted_slopes = []
        self.slopes = []
        self.losses = []

    def fetch_data(self):
        data = yf.download(self.symbol, period="5d", interval="1m")
        return data['Close']
    
    def calculate_slope(self, points):
        time_indices = np.arange(len(points)).reshape(-1, 1)
        model = LinearRegression().fit(time_indices, points)
        return model.coef_[0]
    
    def predict_next_slope(self):
        recent_data = yf.download(self.symbol, period="1d", interval="1m")['Close'].values[-16:]
        recent_data_scaled = self.input_scaler.transform(recent_data.reshape(1, -1))
        
        predicted_slope_scaled = self.nn.forward_propagation(recent_data_scaled)
        predicted_slope = self.output_scaler.inverse_transform(predicted_slope_scaled)
        print(f"Predicted slope for the next 16 points: {predicted_slope}")
        

    def train_model(self):
        data = yf.download(self.symbol, period="5d", interval="1m")
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
                slope = self.calculate_slope(second_16)
                first_16_points.append(first_16)
                slopes.append(slope)

        first_16_points = np.array(first_16_points)
        first_16_points = np.squeeze(first_16_points)
        slopes = np.array(slopes).reshape(-1, 1)

        # Scale both inputs and outputs

        first_16_points_scaled = self.input_scaler.fit_transform(first_16_points)
        slopes_scaled = self.output_scaler.fit_transform(slopes)

        losses = self.nn.train(first_16_points_scaled, slopes_scaled, iterations=self.epochs)
        self.losses = losses

        predicted_slopes = []
        for i in range(len(first_16_points_scaled)):
            predicted_slope = self.nn.forward_propagation(first_16_points_scaled[i].reshape(1, -1))
            predicted_slopes.append(predicted_slope)

        predicted_slopes = np.array(predicted_slopes).reshape(-1, 1)  # Reshape to 2D array
        predicted_slopes = self.output_scaler.inverse_transform(predicted_slopes)  # Inverse transform to get actual values

        self.predicted_slopes = predicted_slope
        self.slopes = slopes

    def print_training_statistics(self):
        print(f"Learning rate:  {self.lr}")
        print(f"Iterations:  {self.epochs}")
        print(f"Hidden Nodes:  {self.hn}")


    def mean_squared_error(self):
        mse = np.mean((self.predicted_slopes - self.slopes.flatten())**2)
        print(f"Mean Squared Error: {mse}")

    def plot_loss(self):
        self.nn.plot_loss(self.losses)


    def run(self):
        self.fetch_data()
        self.train_model()
        self.predict_next_slope()

   

        


        






