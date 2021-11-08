# -*- coding: utf-8 -*-
#import all required libraries and api key
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import requests as rq
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
api_key = 'OWW8G2Z89A55WESC'



@st.cache(persist = True,max_entries = 3,)
def get_stock_frame(tickers):
  #get multiple tickers then append to a dictionary called companydict
  #tickers = ['IBM', 'AAPL', 'GOOGL', 'FB', 'MSFT']
  companydict = {}
  appenddict = {}
  data = None
  k = None
  timeseries = None
  if len(tickers) > 0:
    for ticker in tickers:
      payload = {
          #define the parameters for the api call
          "function": "TIME_SERIES_DAILY_ADJUSTED",
          "symbol": ticker,
          "outputsize": "full", 
          "datatype": "json",
          "apikey": api_key, 
      }
      api_url = "https://www.alphavantage.co/query"
      response = rq.get(api_url, params=payload)
      data = response.json()
      k = list(data.keys())
      timeseries = data[k[1]]
      for key in timeseries:
          appenddict.update({key : timeseries[key]['5. adjusted close']})
          #tempdict = {}
      #print(appenddict)

  
      #print(temp)
      companydict.update({ticker : appenddict})
      appenddict = {} 
      #print(companydict)
  S = pd.DataFrame.from_dict(companydict).sort_index()
  S = S.apply(pd.to_numeric)
  S.columns = [h.lstrip('12345678. ') for h in S.columns]
  return S

#this function will return a relative return dataframe for better stock comparison than absolute stock price
def relreturn(df):
  relret = df.pct_change()
  cummulative_df = (1+relret).cumprod() - 1
  cummulative_df = cummulative_df.fillna(0)
  return cummulative_df


def percent_change(df):
  daily_returns = df.pct_change()
  return daily_returns
#we can get the daily returns of the stock using the pct_change() function




@st.cache(suppress_st_warning=True)
def linear_regression_multi(df,prime_stock, columns):
    #split df into train and test data
    train_df, test_df = train_test_split(df, test_size=0.3, shuffle = False, random_state = 0)

    X_train = pd.DataFrame(train_df[columns])
    y_train = train_df[prime_stock]
    reg = LinearRegression().fit(X_train, y_train)
    st.write('slope of the line: ', reg.coef_)
    st.write('interception value: ', reg.intercept_)
    y_train_pred = reg.predict(X_train)
    MSE_train = mean_squared_error(y_true = y_train, y_pred = y_train_pred)
    st.write('Mean squared error in train data: ', MSE_train)
    st.write('R2_score in train data: ', reg.score(X_train,y_train))

    fig2 = plt.figure(2,figsize=(8,8))
    sns.scatterplot(x=y_train, y = y_train_pred,  color='black')
    sns.lineplot(x=y_train, y= y_train, color='blue', linewidth=3)
    plt.legend(labels=['Diagonal line','Train vs Predicted'])
    plt.title('Train data vs predicted data')
    st.pyplot(fig2)
    
    
    X_test = pd.DataFrame(test_df[columns])
    y_test = test_df[prime_stock]
    y_test_pred = reg.predict(X_test)
    MSE_test =  mean_squared_error(y_true=y_test, y_pred=y_test_pred)
    st.write('Mean squared error in test data: ' , MSE_test)
    st.write('R2_score in test data: ', reg.score(X_test,y_test))
    
    fig3 = plt.figure(3,figsize=(8,8))
    sns.scatterplot(x=y_test, y = y_test_pred,  color='black')
    sns.lineplot(x=y_test, y= y_test, color='blue', linewidth=3)
    plt.legend(labels=['Diagonal line','Test vs Predicted'])
    plt.title('Test data vs predicted data')
    st.pyplot(fig3)



  

def random_stock_walker(ticker,df):
    predict_df = df[ticker].iloc[-30:]
    last_price = predict_df.iloc[-1]
    num_simulations = 500
    simulation_df = pd.DataFrame()
    returns = predict_df.pct_change()
    last_price_list = []
    for x in range(num_simulations):
        count = 0
        daily_vol = returns.std()
        mu = returns.mean()
        price_series = []
    
        price = last_price * (1 + np.random.normal(mu, daily_vol))
        price_series.append(price)   
        for y in range(10):
            if count == 10:
                break
            price = price_series[count] * (1 + np.random.normal(mu, daily_vol))
            price_series.append(price)
            count += 1
            last_price_temp = price_series[-1]
            last_price_list.append(last_price_temp)
    
        simulation_df[x] = price_series
    st.line_chart(simulation_df)
    fig5 = plt.figure(5,figsize = (8,8))
    plt.hist(last_price_list,bins=100)
    plt.axvline(np.percentile(last_price_list,5), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(np.percentile(last_price_list,95), color='r', linestyle='dashed', linewidth=2)
    st.pyplot(fig5)
    st.write("Expected price: ", round(np.mean(last_price_list),2))
    st.write("Quantile (5%): ",np.percentile(last_price_list,5))
    st.write("Quantile (95%): ",np.percentile(last_price_list,95))
    
    





#with container1:
with st.form('form_1'):
    stocktickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    
    prime_stock = st.selectbox('Choose your primary stock', stocktickers, key='primestock')
    
    secondary_stocks = st.multiselect('Choose your stocks to perform analysis with', stocktickers,key='secondarystocks')
    
    submitted = st.form_submit_button("Submit")   
    tickers = []
    tickers.append(prime_stock)
    tickers.extend(secondary_stocks)
        
    #if st.button('Confirm',key=0):
    if submitted:
            tickers = []
            tickers.append(prime_stock)
            tickers.extend(secondary_stocks)
            
            df = get_stock_frame(tickers).loc['2010-01-01' :,:].dropna()
            df.index = pd.to_datetime(df.index)
            st.session_state.df = df
            
            #display daily stock price
            st.header('Daily Stock Price of {}'.format(tickers))
            st.line_chart(df)
            
            #display daily relative return
            st.header('Daily relative returns of {}'.format(tickers))
            st.line_chart(relreturn(df))
            
            #display da heat map
            fig, ax = plt.subplots()
            sns.heatmap(percent_change(df).corr(),annot=True,cmap='BuGn', ax=ax)
            #sns.heatmap(df.corr(),annot=True,cmap='BuGn', ax=ax)
            st.write(fig)
            


#with container2:
stock_for_regression = st.multiselect('Choose your stocks to perform analysis with', secondary_stocks, key=1)
if st.button('Confirm',key = 'regression_stock'):
    linear_regression_multi(st.session_state.df,prime_stock,stock_for_regression)
        
    

stock_for_predict = st.selectbox('Choose the stock you want to predict', tickers, key = 2)
if st.button('Confirm', key = 3):
    random_stock_walker(stock_for_predict,st.session_state.df)
