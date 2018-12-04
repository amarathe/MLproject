import pandas as pd
import numpy as np
import glob
import functools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
import statsmodels.api as sm
from fbprophet import Prophet
from scipy import stats
from pandas.core import datetools
from keras.models import Sequential
from keras.layers import LSTM,Dense

import plotly.graph_objs as go
import plotly.offline as offl
from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import warnings

#Display percent of null values
def printNumMissing(allstockdata):
    nummissing = allstockdata.isnull().sum()
    percentmissing = (nummissing.sort_values(ascending=False)/allstockdata.count())*100
    missinganalysis = pd.concat([nummissing, percentmissing], axis=1, keys=["Total items missing", "Percent"])
    print(missinganalysis)

#Plot close data of ticker name
def plotTicker(tickers, allstockdata):
    traces = []

    for ticker in tickers.split(","):
       stockdata = allstockdata[allstockdata.Name == ticker]
       traces += [go.Scatter(x=stockdata.date, y=stockdata.close, name=ticker)]

    layout = dict(title="Closing price vs date" )
    fig = dict(data=traces, layout=layout)
    offl.plot(fig)

def preprocessdata():
   #Note: run merge.sh before this
   allstockdata = pd.read_csv("all_stocks_5yr.csv")
   
   print ("Analyzing all missing stock data. Dataframe size:", allstockdata.shape[0])
   printNumMissing(allstockdata)
   
   #drops indexes of all stock data that shows null
   for colname in allstockdata.columns.values:
       allstockdata = allstockdata.drop((allstockdata.loc[allstockdata[colname].isnull()]).index)
   
   print ("Checking that all missing data has been removed. Dataframe size:", allstockdata.shape[0])
   printNumMissing(allstockdata)
   
   return allstockdata

#Create a function to process the data into "step" day look back slices
def processData(data,lb):
    X,Y = [],[]
    for i in range(len(data)-lb-1):
        X.append(data[i:(i+lb),0])
        Y.append(data[(i+lb),0])
    return np.array(X),np.array(Y)

def runLSTM(ticker, data, epochs):

#   plotTicker(ticker, data)
   #plotTicker("GOOGL", data)
   data = pd.read_csv("./all_stocks_5yr.csv")
   stock1 = data[data['Name']== ticker].close
   scl = MinMaxScaler()
   #Scale the data
   stock1= np.array(stock1)
   stock1 = stock1.reshape(stock1.shape[0],1)
   stock1 = scl.fit_transform(stock1)

   step = 60
   X,y = processData(stock1,step)
   split = 0.8
   X_train = X[:int(X.shape[0]*split)]
   y_train = y[:int(y.shape[0]*split)]
   X_test = X[int(X.shape[0]*split):]
   y_test = y[int(y.shape[0]*split):]
   
   print(X_train.shape[0])
   print(X_test.shape[0])
   print(y_train.shape[0])
   print(y_test.shape[0])
   
   #Build the model
   model = Sequential()
   model.add(LSTM(256,input_shape=(step,1)))
   model.add(Dense(1))
   model.compile(optimizer='adam',loss='mse')
   #Reshape data for (Sample,Timestep,Features) 
   X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))
   X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],1))
   #Fit model with history to check for overfitting
   history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),shuffle=False)
   
   plt.plot(history.history['loss'])
   plt.plot(history.history['val_loss'])
   plt.xlabel('epochs')
   plt.ylabel('loss')
   plt.legend(['train','validation'])
   plt.show() 
   #We see this is pretty jumpy but we will keep it at 300 epochs. With more data, it should smooth out the loss
   #Lets look at the fit
   Xt = model.predict(X_test)
   plt.plot(scl.inverse_transform(y_test.reshape(-1,1)))
   plt.plot(scl.inverse_transform(Xt))
   plt.show() 

def runprophet(df, ticker):
   # # Time Series Forecast with Prophet
   # 
   # ## Introduction:
   # This is a simple kernel in which we will forecast stock prices using Prophet (Facebook's library for time series forecasting). However, historical prices are no indication whether a price will go up or down.  I'll rather use my own variables and use machine learning for stock price prediction rather than just using historical prices as an indication of stock price increase.
   # 
   # ## A Summary about Prophet:
   # Facebook's research team has come up with an easier implementation of forecasting through it's new library called Prophet. From what I have read, the blog state's that analyst that can produce high quality forecasting data is rarely seen. This is one of the reasons why Facebook's research team came to an easily approachable way for using advanced concepts for time series forecasting and us Python users, can easily relate to this library since it uses Scikit-Learn's api (Similar to Scikit-Learn). To gain further information, you can look at  [Prophet Blog](https://research.fb.com/prophet-forecasting-at-scale/). Prophet's team main goal is to <b>to make it easier for experts and non-experts to make high quality forecasts that keep up with demand. </b> <br><br>
   # 
   # There are several characteristics of Prophet (you can see it in the blog) that I want to share with you Kaggles that shows where Prophet works best:
   # <ul>
   # <li>hourly, daily, or weekly observations with at least a few months (preferably a year) of history </li>
   # <li>strong multiple “human-scale” seasonalities: day of week and time of year </li>
   # <li>Important holidays that occur at irregular intervals that are known in advance (e.g. the Super Bowl) </li>
   # <li>A reasonable number of missing observations or large outliers </li>
   # <li>Historical trend changes, for instance due to product launches or logging changes </li>
   # <li>Trends that are non-linear growth curves, where a trend hits a natural limit or saturates </li>
   # </ul>
   # <br><br>
   
   # Import Libraries
   # Statsmodels widely known for forecasting than Prophet
   init_notebook_mode(connected=True)
   warnings.filterwarnings("ignore")
   
   # plt.style.available
   plt.style.use("seaborn-whitegrid")

   # Brief Description of our dataset
   df.describe()
   

   # Replace the column name from name to ticks
   df = df.rename(columns={'Name': 'Ticks'})
   
   # For this simple tutorial we will analyze Amazon's stock and see what will the trend look like for the nearby future of this stock relying on past stock prices.
   amzn = df.loc[df['Ticks'] == ticker]
   # Create a copy to avoid the SettingWarning .loc issue 
   amzn_df = amzn.copy()
   # Change to datetime datatype.
   amzn_df.loc[:, 'date'] = pd.to_datetime(amzn.loc[:,'date'], format="%Y/%m/%d")
   
   # Simple plotting of Amazon Stock Price
   # First Subplot
   f, (ax0, ax1) = plt.subplots(1, 2, figsize=(14,5))
   ax0.plot(amzn_df["date"], amzn_df["close"])
   ax0.set_xlabel("Date", fontsize=12)
   ax0.set_ylabel("Stock Price")
   ax0.set_title("Amazon Close Price History")
   
   # Fourth Subplot
   ax1.plot(amzn_df["date"], amzn_df["volume"], color="orange")
   ax1.set_xlabel("Date", fontsize=12)
   ax1.set_ylabel("Stock Price")
   ax1.set_title("Amazon's Volume History")
   plt.show()
   
   # ### Prophet Introduction:
   # Prophet is Facebook's library for time series forecasting. In my opinion, Prophet works best with datasets that are higely influenced by seasonality (electricity bills, restaurant visitors etc.) However, I wanted to show the simplicity of using Prophet for simple forecasting which is the main aim of this kernel.
   # 
   # #### Steps for using Prophet:
   # <ul>
   # <li>Make sure you replace closing price for y and date for ds. </li>
   # <li>Fit that dataframe to Prophet in order to detect future patterns. </li>
   # <li>Predict the upper and lower prices of the closing price. </li>
   # </ul>
   
   m = Prophet()
   
   # Drop the columns
   ph_df = amzn_df.drop(['open', 'high', 'low','volume', 'Ticks'], axis=1)
   ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)

   numtotal = int(ph_df.shape[0])
   numtrain = int(ph_df.shape[0] * 0.99)
   ph_df_test = ph_df.head(numtrain)
   
   m.fit(ph_df)
   
   # Create Future dates
   future_prices = m.make_future_dataframe(periods=365)
   
   # Predict Prices
   forecast = m.predict(future_prices)
   print ("DEBUG: future prediction: ")
   print( forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

   from fbprophet.diagnostics import cross_validation
   print ("DEBUG: numtrain:", numtrain)
   print ("DEBUG: numtotal:", numtotal)
   #Convert from 5 day weeks to 7 day weeks
   df_cv = cross_validation(m, initial=str(int(numtrain*7/5))+ ' days', period='10 days', horizon = '10 days')
   print ("cross-validation:")
   from fbprophet.diagnostics import performance_metrics
   df_p = performance_metrics(df_cv)
   print(df_p.head())

   from fbprophet.plot import plot_cross_validation_metric
   fig = plot_cross_validation_metric(df_cv, metric='rmse')
   
   import matplotlib.dates as mdates
   
   # Dates
   starting_date = dt.datetime(2018, 4, 7)
   starting_date1 = mdates.date2num(starting_date)
   trend_date = dt.datetime(2018, 6, 7)
   trend_date1 = mdates.date2num(trend_date)
   
   pointing_arrow = dt.datetime(2018, 2, 18)
   pointing_arrow1 = mdates.date2num(pointing_arrow)
   
   # Forecast amazon
#   fig = m.plot(forecast)
   fig = m.plot(df_cv)
   ax1 = fig.add_subplot(111)
   ax1.set_title("Amazon Stock Price Forecast", fontsize=16)
   ax1.set_xlabel("Date", fontsize=12)
   ax1.set_ylabel("Close Price", fontsize=12)
   
#   # Forecast initialization arrow
#   ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow1, 1350), xytext=(starting_date1,1700),
#               arrowprops=dict(facecolor='#ff7f50', shrink=0.1),
#               )
#   
#   # Trend emphasis arrow
#   ax1.annotate('Upward Trend', xy=(trend_date1, 1225), xytext=(trend_date1,950),
#               arrowprops=dict(facecolor='#6cff6c', shrink=0.1),
#               )
   
#   ax1.axhline(y=1260, color='b', linestyle='-')
   
   plt.show()
   
   fig2 = m.plot_components(forecast)
   plt.show()
   
   # Monthly Data Predictions
   print ("DEBUG: Monthly Predictions")
   m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)
   future = m.make_future_dataframe(periods=12, freq='M')
   fcst = m.predict(future)
   fig = m.plot(fcst)
   plt.title("Monthly Prediction \n 1 year time frame")
   
   plt.show()
   
   # #### Trends:
   # <ul> 
   # <li>Amazon's stock price is showing signs of upper trend yearly. </li>
   # <li> Amazon's stock price show upper trend signs during January (December Sales tend to give a boost to Amazon's stock price)</li>
   # <li>There is no weekly trend for stock prices. </li>
   # </ul>
   
   fig = m.plot_components(fcst)
   plt.show()
   
#####END PROPHET

stockdata = preprocessdata()
#print ("DEBUG: stockdata size:", stockdata.shape[0])
#runLSTM("AMZN", stockdata, 30)
runprophet(stockdata, "AMZN")
