from NNStockPredictor import StockPredictor

symbol = 'SPY'
sp = StockPredictor(symbol, lr=0.1, epochs=10000, hn=64)
sp.run()
sp.plot_loss()
sp.print_training_statistics()
