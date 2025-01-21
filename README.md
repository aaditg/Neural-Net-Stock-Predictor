Stock Slope Predictor

This project trains a neural network to predict the slope of stock price changes based on historical data. It takes 16 1-minute intervals of stock prices as input and outputs the slope of the next 16 minutes.

Data Collection: Stock price data is downloaded using Yahoo's yfinance library.

Preprocessing:
The input is the first 16 1-minute price intervals of every hour.

The target output is the slope of the next 16 minutes.


Model:
Input layer: 16 nodes (one for each price).

Hidden layers: Changeable number of layers.

Output layer: 1 node (predicted slope).

Training:
The model uses Mean Squared Error (MSE) as the loss function and gradient descent for optimization.

It will also output a loss/epochs graph.

Required Libraries - pip install numpy finance


Customization:
Change the NN parameters (layers, nodes, learning rate, iterations.

Normalize the input data if predictions are inconsistent using provided functions.

Test different stock symbols by changing the symbol.





