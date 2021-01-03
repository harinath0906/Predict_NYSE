# Predict NYSE Next Day closing price
In this project, I will attempt to predict next day close prices for a subset of New York Stock Exchange (NYSE) stocks. We investigated the predictive ability of linear regression, extreme gradient boosting (XGBoost), and a long short-term memory neural network (LSTM) model. The objective of our research was to determine a model that generalises across the entire NYSE as determined by the sectors to which each stock belongs. Through the processes of feature engineering and hyperparameter optimisation we found that the LSTM model exhibited the strongest performance with an average mean squared error (MSE) of 0.692 over a one year testing period in 2016.


## Dataset
The dataset used here is from [Kaggle](https://www.kaggle.com/dgawlik/nyse)

## Below are some techniques covered and demonstrated in the Jupyter Notebook
* Data preprocessing
* Linear Regression
* XGBoost
* LSTM Neural Network
* Tensorflow
* Feature Engineering - Simple moving average, Exponential moving average, Triple exponential moving average, Moving Average Convergence Divergence, Relative strength index, Bollinger bands, Williams’ %R, Log-Returns
* Visualisation using matplotlib, seaborn
* Hyperparameter Optimization

## Results
The report [here](https://github.com/harinath0906/Predict_NYSE/blob/main/Project_report.pdf) contains the extensive details of all modules used with their background, all experiment results, related work and the details of the neural network.

## Future Work
We also see a potential for future work by optimizing and expanding feature space by leveraging other python finance packages. We also see merits in evaluating other variants of LSTM such as gated recurrent units (GRU). We can also supplement our dataset with candlestick charts and then use a convolutional NN (CNN) in combination with LSTM, as this would help in identifying additional features not available in technical indicators.

## References
1.	Bischoff B. Adjusted closing price vs. closing price. Zacks Investment Research. 2019 Mar 31. Available from: https://finance.zacks.com/adjusted-closing-price-vs-closing-price-9991.html 
2.	Siew HL, Nordin MJ. Regression techniques for the prediction of stock price trend. IEEE. 2012 Dec 31
3.	Basak S, Kar S, Saha S, Khaidem L, Dey R. Predicting the direction of stock market prices using tree-based classifiers. Journal of Scientometric Research. 2019 Jan; 47:552-67
4.	Fischer T, Krauss, C. Deep learning with long short-term memory networks for financial market predictions. Journal of Economic Dynamics and Control. 2018 Oct 16; 270(2):654-669
5.	Ghosh P, Neufeld A, Sahoo JK. Forecasting directional movements of stock prices for intraday trading using LSTM and random forests. 2020
6.	Benediktsson J. Ta-lib. GitHub repository. 2020. Available from: https://github.com/mrjbq7/ta-lib 
7.	Sharma H. Technical analysis of stocks using ta-lib. Towards Data Science. 2020 Sep 5. Available from: https://towardsdatascience.com/technical-analysis-of-stocks-using-ta-lib-305614165051
8.	Sharpe M. Lognormal model for stock prices. Available from: http://www.math.ucsd.edu/~msharpe/stockgrowth.pdf 
9.	Kapse AD. Get smarter in eda+ml modelling+stock prediction, version 7. Kaggle. 2020. Available from: https://www.kaggle.com/akhileshdkapse/get-smarter-in-eda-ml-modelling-stock-prediction 
10.	Hallows S. Using xgboost with scikit-learn, version 1. Kaggle. 2018. Available from: https://www.kaggle.com/stuarthallows/using-xgboost-with-scikit-learn 
11.	Williams RJ, Hinton GE, Rumelhart, DE. Learning representations by back-propagating errors. Nature. 1986 Oct; 323(6088):533-36
12.	Castilla P. Predict stock prices with lstm, version 10. Kaggle. 2016. Available from: https://www.kaggle.com/pablocastilla/predict-stock-prices-with-lstm?scriptVersionId=1008769 
13.	Malm, R. Ny stock price prediction rnn lstm gru, version 4. Kaggle. 2017. Available from: https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru 
14.	Kandel I, Castelli M. The effect of batch size on the generalizability of the convolutional neural networks on a histopathology dataset. ICT Express. 2020 May 5
