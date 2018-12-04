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

def stocks_data_analysis (input_data): 
    ###############################################################
    ## Stock Data Analysis (argument: stock data)
    list_ticker = ['FB', 'AAPL', 'NFLX', 'GOOGL', 'AMZN', 'GM']                   
    input_data.set_index('Name', inplace=True)
    amazon = input_data.loc['AMZN']
    amazon.reset_index(inplace=True)
    amazon.set_index("date", inplace=True)
    amazon = amazon.drop("Name", axis=1)
    
    facebook = input_data.loc['FB']
    facebook.reset_index(inplace=True)
    facebook.set_index("date", inplace=True)
    facebook = facebook.drop("Name", axis=1)
    
    apple = input_data.loc['AAPL']
    apple.reset_index(inplace=True)
    apple.set_index("date", inplace=True)
    apple = apple.drop("Name", axis=1)
    
    netflix = input_data.loc['NFLX']
    netflix.reset_index(inplace=True)
    netflix.set_index("date", inplace=True)
    netflix = netflix.drop("Name", axis=1)
    
    google = input_data.loc['GOOGL']
    google.reset_index(inplace=True)    
    google.set_index("date", inplace=True)
    google = google.drop("Name", axis=1)
    
    rand_stock = input_data.loc[list_ticker[len(list_ticker)-1]]
    rand_stock.reset_index(inplace=True)    
    rand_stock.set_index("date", inplace=True)
    rand_stock = rand_stock.drop("Name", axis=1)
                          
    ticker_stocks = pd.concat([facebook, apple,netflix, google, amazon, rand_stock], axis=1,keys=list_ticker)
    ticker_stocks.columns.names = ['Ticker','Stock Info']
    ticker_stocks.head()
    ticker_stocks.reset_index(inplace=True)
    ticker_stocks.set_index('date')
    ticker_stocks['date'] = pd.to_datetime(ticker_stocks['date'])
    ticker_stocks.head()
    
    ticker_stocks.xs(key='close',axis=1,level='Stock Info').max()
    
    returns = pd.DataFrame()
    for tick in list_ticker:
        returns[tick+' Return'] = ticker_stocks[tick]['close'].pct_change()
    returns.head()
    DateCol = ticker_stocks['date']
    returns = pd.concat([returns,DateCol], axis = 1)
    returns.head()
    returns.reset_index(inplace=True)
    returns.set_index("date", inplace=True)
    returns = returns.drop("index", axis=1)
    print(returns)
    
    import seaborn as sns
    sns.pairplot(returns[1:])
    # dates with the lowest returns for each stock
    LowReturnDates = returns.idxmin()
    LowReturnDates.head()
    
    returns.idxmax()
    returns.std()
    
    import seaborn as sns
    whitegrid = sns.set_style('whitegrid')
    plt.savefig("./whitegrid.pdf")
    plt.figure()
    heat = sns.heatmap(ticker_stocks.xs(key='close',axis=1,level='Stock Info').corr(),annot=True)
    plt.savefig('./heatmap.pdf')
    plt.figure()
    ## END OF Stock Data ANalysis


from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean() #(timeseries, window=12)
    rolstd = timeseries.rolling(window=12).std()
    
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    

def runLSTM(ticker, data, epochs):

#   plotTicker(ticker, data)
   #plotTicker("GOOGL", data)
#   data = pd.read_csv("./all_stocks_5yr.csv")
   stock1 = data[data['Name']== ticker].close
   scl = MinMaxScaler()
   #Scale the data
   stock1= np.array(stock1)
   stock1 = stock1.reshape(stock1.shape[0],1)
   stock1 = scl.fit_transform(stock1)

   step = 5
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
   plt.legend(['Actual','Model'])
   plt.xlabel('Days (shifted)')
   plt.ylabel('Stock Price')
   plt.title('AMZN: Train 80%, Test 20%')
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
   from fbprophet import Prophet
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
   f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
   ax1.plot(amzn_df["date"], amzn_df["close"])
   ax1.set_xlabel("Date", fontsize=12)
   ax1.set_ylabel("Stock Price")
   ax1.set_title("Amazon Close Price History")
   
   # Second Subplot
   ax1.plot(amzn_df["date"], amzn_df["high"], color="green")
   ax1.set_xlabel("Date", fontsize=12)
   ax1.set_ylabel("Stock Price")
   ax1.set_title("Amazon High Price History")
   
   # Third Subplot
   ax1.plot(amzn_df["date"], amzn_df["low"], color="red")
   ax1.set_xlabel("Date", fontsize=12)
   ax1.set_ylabel("Stock Price")
   ax1.set_title("Amazon Low Price History")
   
   # Fourth Subplot
   ax2.plot(amzn_df["date"], amzn_df["volume"], color="orange")
   ax2.set_xlabel("Date", fontsize=12)
   ax2.set_ylabel("Stock Price")
   ax2.set_title("Amazon's Volume History")
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
   
   m = Prophet()
   m.fit(ph_df)
   
   # Create Future dates
   future_prices = m.make_future_dataframe(periods=365)
   
   # Predict Prices
   forecast = m.predict(future_prices)
   forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
   
   import matplotlib.dates as mdates
   
   # Dates
   starting_date = dt.datetime(2018, 4, 7)
   starting_date1 = mdates.date2num(starting_date)
   trend_date = dt.datetime(2018, 6, 7)
   trend_date1 = mdates.date2num(trend_date)
   
   pointing_arrow = dt.datetime(2018, 2, 18)
   pointing_arrow1 = mdates.date2num(pointing_arrow)
   
   # Learn more Prophet tomorrow and plot the forecast for amazon.
   fig = m.plot(forecast)
   ax1 = fig.add_subplot(111)
   ax1.set_title("Amazon Stock Price Forecast", fontsize=16)
   ax1.set_xlabel("Date", fontsize=12)
   ax1.set_ylabel("Close Price", fontsize=12)
   
   # Forecast initialization arrow
   ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow1, 1350), xytext=(starting_date1,1700),
               arrowprops=dict(facecolor='#ff7f50', shrink=0.1),
               )
   
   # Trend emphasis arrow
   ax1.annotate('Upward Trend', xy=(trend_date1, 1225), xytext=(trend_date1,950),
               arrowprops=dict(facecolor='#6cff6c', shrink=0.1),
               )
   
   ax1.axhline(y=1260, color='b', linestyle='-')
   
   plt.show()
   
   m.plot_components(forecast)
   plt.show()
   
   # Monthly Data Predictions
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
   
   # ### Stocks more Susceptible to Seasonality Trends:
   # ---> Description Later:
   # American Airlines
   aal_df = df.loc[df["Ticks"] == "AAL"]
   
   aal_df.loc[:, 'date'] = pd.to_datetime(aal_df.loc[:,'date'], format="%Y/%m/%d")
   
   aal_df.info()
   
   f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,5))
   
   aal_df["close_log"] = np.log(aal_df["close"])
   aal_df["high_log"] = np.log(aal_df["high"])
   aal_df["low_log"] = np.log(aal_df["low"])
   
   ax1.plot(aal_df["date"], aal_df["close_log"])
   ax1.set_title("Normalized Close Price")
   ax2.plot(aal_df["date"], aal_df["high_log"], color="g")
   ax2.set_title("Normalized High Price")
   ax3.plot(aal_df["date"], aal_df["low_log"], color="r")
   ax3.set_title("Normalized Low Price")
   plt.show()
   
   aal_df['std_close'] = aal_df["close"].rolling(5).std()
   aal_df['mean_close'] = aal_df["close"].rolling(5).mean()
   aal_df['twenty_mean_close'] = aal_df["close"].rolling(20).mean()
   
   f, (std_ax, avg_ax) = plt.subplots(1, 2, figsize=(18,5))
   
   std_ax.plot(aal_df["date"], aal_df["std_close"], color="r", label="Standard Deviation")
   std_ax.legend(loc="upper left")
   std_ax.set_title("Standard Deviation American Airlines \n (AAL)")
   
   avg_ax.plot(aal_df["date"], aal_df["mean_close"], color="g", label="5-day MA")
   avg_ax.plot(aal_df["date"], aal_df["twenty_mean_close"], color="k", label="20-day MA")
   avg_ax.legend(loc="upper left")
   avg_ax.set_title("5 Day Average AAL \n Closing Price")
   plt.show()
   
   m = Prophet()
   
   # Drop the columns
   ph_df = aal_df.drop(['open', 'high', 'low','volume', 'Ticks', 'close_log', 'high_log', 'mean_close', 'twenty_mean_close', 'low_log', 'std_close'], axis=1)
   ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)
   
   m.fit(ph_df)
   
   future_prices = m.make_future_dataframe(periods=365)
   
   # Predict Prices
   forecast = m.predict(future_prices)
   forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
   
   # Dates
   starting_date = dt.datetime(2018, 4, 7)
   starting_date1 = mdates.date2num(starting_date)
   trend_date = dt.datetime(2018, 2, 7)
   trend_date1 = mdates.date2num(trend_date)
   
   pointing_arrow = dt.datetime(2018, 2, 18)
   pointing_arrow1 = mdates.date2num(pointing_arrow)
   
   # Dates Forecasting Resistance Line
   resistance_date = dt.datetime(2016, 4, 7)
   resistance_date1 = mdates.date2num(resistance_date)
   
   # Dates Forecasting Support Line
   support_date = dt.datetime(2013, 4, 7)
   support_date1 = mdates.date2num(support_date)
   
   # Learn more Prophet tomorrow and plot the forecast for amazon.
   fig = m.plot(forecast)
   ax1 = fig.add_subplot(111)
   ax1.set_title("American Airlines Stock Price Forecast", fontsize=16)
   ax1.set_xlabel("Date", fontsize=12)
   ax1.set_ylabel("Close Price", fontsize=12)
   
   # Forecast initialization arrow
   ax1.annotate('Forecast \n Initialization', xy=(pointing_arrow1, 55), xytext=(starting_date1,40),
               arrowprops=dict(facecolor='#ff7f50', shrink=0.1),
               )
   
   # # Trend emphasis arrow
   ax1.annotate('Last Closing Price \n Before Forecast \n ($51.40)', xy=(trend_date1, 57), xytext=(trend_date1,70),
               arrowprops=dict(facecolor='#6cff6c', shrink=0.1),
               )
   
   # Resistance Line
   ax1.annotate('Resistance Line \n of Forecast Peak ${:.2f}'.format(forecast["yhat"].max()), xy=(resistance_date1, 65), xytext=(resistance_date1,73),
               arrowprops=dict(facecolor='#FF0000', shrink=0.1),
               )
   
   # Support Line
   ax1.annotate('Support Line \n of Forecast Bottom \n $51.40', xy=(support_date1, 53), xytext=(support_date1,40),
               arrowprops=dict(facecolor='#00FF40', shrink=0.1),
               )
   
   ax1.axhline(y=65, color='r', linestyle='--')
   ax1.axhline(y=54.2, color='g', linestyle='--')
   
   plt.show()
   
   m.plot_components(forecast)
   plt.show()
#   
######END PROPHET

###############################################################
#Merging the code here  -- Tulasi 

def runARIMA(df, ticker):
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    data_3 = pd.read_csv('C:/Users/Tulasi/Documents/Tulasi/MLPROJECT/all_stocks_5yr.csv', parse_dates=['date'], index_col='date',date_parser=dateparse)
    print(data_3.head())
    data_3.head()
    data_3.dtypes
    print(data_3.index)

    ts = data_3[data_3['Name']== ticker].close

    #Testing the stationarity of the original time series
    test_stationarity(ts)

#Estimating & Eliminating Trend
#plt.plot(ts)
    ts_log = np.log(ts)
#plt.plot(ts_log)

# Moving average for 5
    moving_avg = ts_log.rolling(window=12).mean()
    plt.plot(ts_log, color='blue')
    plt.plot(moving_avg, color='red')
    plt.show()
    
    ts_log_moving_avg_diff = ts_log - moving_avg
    ts_log_moving_avg_diff.head(12)
    
    #Testing the stationarity of the error = logarithmic data -moving avg of logarithmic data
    ts_log_moving_avg_diff.dropna(inplace=True)
    test_stationarity(ts_log_moving_avg_diff)


##Testing the stationarity after exponentially weighted ma


    expwighted_avg = ts_log.ewm(halflife=12).mean()
    plt.plot(ts_log)
    plt.plot(expwighted_avg, color='red')
    plt.show()
    
    ts_log_ewma_diff = ts_log - expwighted_avg
    test_stationarity(ts_log_ewma_diff)
    


    #Eliminating Trend and Seasonality
    #The simple trend reduction techniques discussed before don’t work in all cases, particularly the ones with high seasonality. Lets discuss two ways of removing trend and seasonality:
    #
    #Differencing – taking the differece with a particular time lag
    #Decomposition – modeling both trend and seasonality and removing them from the model.
    
    ts_log_diff = ts_log - ts_log.shift()
    plt.plot(ts_log_diff)
    plt.show()
    
    ts_log_diff.dropna(inplace=True)
    test_stationarity(ts_log_diff)
    
    #Decomposition
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(ts_log,model='additive',freq=1)
    
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    
    plt.subplot(411)
    plt.plot(ts_log, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonal')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label='Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    #check stationarity of residuals:
    #The Dickey-Fuller test statistic is significantly lower than the 1% critical value. So this TS is very close to stationary. 
    ts_log_decompose = residual
    ts_log_decompose.dropna(inplace=True)
    test_stationarity(ts_log_decompose)
    
    #ACF and PACF plots:
    from statsmodels.tsa.stattools import acf, pacf
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')
    #Plot ACF: 
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    #Perform Augmented Dickey–Fuller test:
    #from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.stattools import acf, pacf
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima_model import ARIMA
    
    #Combined Model
    model = ARIMA(ts_log, order=(0,1,1))
    
    results_ARIMA = model.fit(disp=-1)  
    plt.plot(ts_log_diff)
    plt.plot(results_ARIMA.fittedvalues, color='green')
    plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
    plt.show()
    
    
    predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
    predictions_ARIMA_diff.head()
    
    predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
    predictions_ARIMA_diff_cumsum.head()
    
    predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
    predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
    predictions_ARIMA_log.head()
    
    predictions_ARIMA = np.exp(predictions_ARIMA_log)
    plt.plot(ts_log)
    plt.plot(predictions_ARIMA, color='black')
    plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-ts)**2)/len(ts)))
    plt.show()
    
    results_ARIMA.plot_predict(1,264) 
    results_ARIMA.forecast(steps=120)
    #plt.plot(moving_avg, color='black')
    plt.show()
    
    #results_ARIMA.plot_predict() 
    #x=results_ARIMA.forecast()
    #plt.plot(moving_avg, label='movavg', color='magenta')
    #plt.show()

def runLSTM_FANG(ticker_array,test_ticker,df,n_epochs):
    
    test_stock = df[df['Name']==test_ticker].close
    #Build the model
    model = Sequential()
    step = 5
    model.add(LSTM(256,input_shape=(step,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam',loss='mse')
    
    #plotTicker("AAPL", df)
    #plotTicker("GOOGL", df)

#    stock2 = df[df['Name']=='FB'].close
#    stock3 = df[df['Name']=='AAPL'].close
#    stock4 = df[df['Name']=='NFLX'].close
#    stock5 = df[df['Name']=='GOOG'].close
    
    for i in range(len(ticker_array)):
        ticker = ticker_array[i]
        print ('Train using ', ticker)
        trainstock = df[df['Name']==ticker].close             
        scl = MinMaxScaler()
        #Scale the data
        trainstock= np.array(trainstock)
        trainstock = trainstock.reshape(trainstock.shape[0],1)
        trainstock = scl.fit_transform(trainstock)
        
        test_stock= np.array(test_stock)
        test_stock = test_stock.reshape(test_stock.shape[0],1)
        test_stock = scl.fit_transform(test_stock)
        
        X,y = processData(trainstock,step)
        X_test_stock,y_test_stock = processData(test_stock,step)
        train_split = 1.00
        #X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
        X_train = X[:int(X.shape[0]*train_split)]
        y_train = y[:int(y.shape[0]*train_split)]
    
        test_split = 1.00
        X_test_stock = X_test_stock[:int(X_test_stock.shape[0]*test_split)]
        y_test_stock = y_test_stock[:int(y_test_stock.shape[0]*test_split)]
    
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],1))   
        X_test_stock = X_test_stock.reshape((X_test_stock.shape[0],X_test_stock.shape[1],1))
        history = model.fit(X_train,y_train,epochs=n_epochs,validation_data=(X_test_stock,y_test_stock),shuffle=False)
        
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train','validation'])
        plt.show() 
        #We see this is pretty jumpy but we will keep it at 300 epochs. With more data, it should smooth out the loss
        #Lets look at the fit
        Xt = model.predict(X_test_stock)
        plt.plot(scl.inverse_transform(y_test_stock.reshape(-1,1)))
        plt.plot(scl.inverse_transform(Xt))
        plt.legend(['Actual','Model'])
        plt.xlabel('Day')
        plt.ylabel('Stock Price')
        plt.title('Train with FB/AAPL/NFLX/GOOGL')
        plt.show() 


# Preprocess data to remove null values from 5year stocks data 
stockdata = preprocessdata() 
# Perform data analysis on stocks chosen:
stocks_data_analysis(stockdata) 

#print ("DEBUG: stockdata size:", stockdata.shape[0])
runLSTM("AMZN", stockdata, 50) #(ticker, data, epochs)
runprophet(stockdata, "AMZN")

##############################################################
# Analysis on Train with FB/AAPL/NFLX/GOOGL and test with AMZN
list_tickers = ['FB', 'AAPL', 'NFLX', 'GOOGL']
runLSTM_FANG(list_tickers,"AMZN",stockdata,50)
##############################################################

runARIMA(stockdata,"AMZN")

