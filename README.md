Stock prediction using LSTM RNN

- Predicts daily, weekly/monthly price of BSE SENSEX


Dataset

- The dataset (.csv) comprises of past 20 year data of BSE SENSEX (primary index for India) from 1997-2017.
- The dataset only has the closing price (which is used as the sole feature for training LSTM's)

Model

- The LSTM model has 2 hidden layers (50,200 cells respectively)
- The model uses rmsprop and adam as training algorithms for separate tasks i.e. prior for weekly / longer term predictions and the latter for daily predictions
- Linear activation (default) is used for the output layer
- The learning rates and decay rates have been optimized accordingly

Evaluation

- In order to evaluate the model's daily prediction (1 time step ahead) we used RMSE and forecast Bias. Ideally both the values should be close to 0
- In order to evaluate the model's weekly / longer term predictions (7 / 30 time steps ahead) we used directional accuracy as the performance measure.

Run


- In order the run the model, just fork / clone the repo and run the python file "stockpredictor.py" with the below command from the root folder:

```
python stockpredictor.py
```

- In order to debug or fine tune the model for testing or other purposes open the stockpredictor.py file in one of the many editors of python and play with model parameters like learning rate, dropout, cells, no. of hidden layers etc.


Measures to avoid overfitting

- Random shuffling of training windows to make sure the model doesn't fit to the sequences
- Added dropout at hidden layers to ensure regularization
- Use of early callback while training the model
