import os


# import tensorflow as tf
# tf.test.gpu_device_name()
# tf.keras.backend.set_image_data_format("channels_last")

from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from PIL import Image
import pickle
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
yf.pdr_override()

#import seaborn as sns
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
#import all_function

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

import random

from functools import reduce

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import picard

np.random.seed(42)
random.seed(7)

token = "6481b11d245909.63698937"

##############################################
job=pd.read_csv(r"./job (1).csv")
job.loc[len(job)] = ['2023-02', 71.68311]
job.loc[len(job)] = ['2023-03', 71.68311]
job.loc[len(job)] = ['2023-04', 71.68311]
job.loc[len(job)] = ['2023-05', 71.68311]
job.loc[len(job)] = ['2023-06', 71.68311]
Labour=pd.read_csv(r"./Labour (1).csv")
Labour.loc[len(Labour)] = ['2023-02', 42]
Labour.loc[len(Labour)] = ['2023-03', 42]
Labour.loc[len(Labour)] = ['2023-04', 42]
Labour.loc[len(Labour)] = ['2023-05', 42]
Labour.loc[len(Labour)] = ['2023-06', 42]
Manufacture_indexing=pd.read_csv(r"./Manufacture_indexing (1).csv")
Manufacture_indexing.drop("Unnamed: 0",axis=1,inplace=True)
Manufacture_indexing.loc[len(Manufacture_indexing)] = [259.904, '02-2023']
Manufacture_indexing.loc[len(Manufacture_indexing)] = [259.904, '03-2023']
Manufacture_indexing.loc[len(Manufacture_indexing)] = [259.904, '04-2023']
Manufacture_indexing.loc[len(Manufacture_indexing)] = [259.904, '05-2023']
Manufacture_indexing.loc[len(Manufacture_indexing)] = [259.904, '06-2023']
CPI=pd.read_csv(r"./CPI (1).csv")
CPI.drop("Unnamed: 0",axis=1,inplace=True)
CPI.loc[len(CPI)] = [300.536, '2023-02']
CPI.loc[len(CPI)] = [300.536, '2023-03']
CPI.loc[len(CPI)] = [300.536, '2023-04']
CPI.loc[len(CPI)] = [300.536, '2023-05']
CPI.loc[len(CPI)] = [300.536, '2023-06']
inflation=pd.read_csv(r"./inflation (2).csv")

Housing=pd.read_csv(r"./housing_new_gg (1).csv")
Housing.loc[Housing['Year-Months'] == '2023-02', 'Housing_value'] = 377
Housing.loc[Housing['Year-Months'] == '2023-03', 'Housing_value'] = 377
Housing.loc[Housing['Year-Months'] == '2023-04', 'Housing_value'] = 377
Housing.loc[Housing['Year-Months'] == '2023-05', 'Housing_value'] = 377
Housing.loc[Housing['Year-Months'] == '2023-06', 'Housing_value'] = 377
nasdaq=pd.read_csv(r"./NASDAQCOM.csv")
nasdaq['DATE'] = pd.to_datetime(nasdaq['DATE'])
start_date = nasdaq['DATE'].max()
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
missing_dates = pd.DataFrame({'DATE': date_range})
missing_dates = missing_dates[~missing_dates['DATE'].isin(nasdaq['DATE'])]
nasdaq = pd.concat([nasdaq, missing_dates], sort=False)
nasdaq.sort_values('DATE', inplace=True)
nasdaq.reset_index(drop=True, inplace=True)
nasdaq.fillna(method='ffill', inplace=True)
nasdaq['DATE'] = nasdaq['DATE'].astype(str)
########################################################################

def get_columns(stock):
    msft = yf.Ticker(stock)
    data = msft.history(start="2005-01-01")
    df=data.reset_index()

    df['DATE_new'] = df['Date'].apply(lambda x: x.strftime('%Y-%m-%d'))

    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['month_num'] = pd.DatetimeIndex(df['Date']).month
    df['month_num'] = df['month_num'].apply(lambda x: f"{x:02d}")

    df['Year_month'] = df['Year'].astype(str) + '-' + df['month_num'].astype(str)
    df['month_Year'] = df['month_num'].astype(str) + '-' + df['Year'].astype(str)

    merged_df = pd.merge(df,job, left_on='Year_month', right_on='TIME')

    merged_df = pd.merge(merged_df,Labour, left_on='Year_month', right_on='Time')

    
    merged_df = pd.merge(merged_df,Manufacture_indexing, left_on='month_Year', right_on='Month_year')

    #CPI.drop("Unnamed: 0",axis=1,inplace=True)
    merged_df = pd.merge(merged_df,CPI, left_on='TIME', right_on='Month_years')

    merged_df = pd.merge(merged_df,inflation, left_on='TIME', right_on='Year-Month')

    merged_df = pd.merge(merged_df,Housing, left_on='TIME', right_on='Year-Months')
    merged_df=pd.merge(merged_df,nasdaq, left_on='DATE_new', right_on='DATE')

#     df=merged_df.drop(['Month_year',"DATE","DATE_new", 'Month_years',"Time","Year","Year-month"],axis=1)
#     df=df.drop(["month_num","month_Year","TIME","Year-Months"], axis=1)

    df=merged_df.drop(['Month_year',"DATE","DATE_new", 'Month_years',"Year-Month", "Time","Year","Year_month","month_num","month_Year","TIME","Year-Months"], axis=1)

    df['NASDAQCOM']=df.NASDAQCOM.astype(float)

    df['inflation_Value']=df.inflation_Value.astype(float)
    
    return df
###########################################################


#####################Sidebar##################


st.write("# Forecasting Time Series Data - Stock Price")
st.markdown("""---""")
url="3kt.png"
image = Image.open(url)
st.sidebar.image(image, caption='3k Technologies – Smart Technology Solutions')
st.sidebar.markdown("Forecasting Time Series Data - Stock Price ")
st.sidebar.title("For Details on the Stock Data.")
st.sidebar.markdown(":green[['Balance Sheet','Income Statement','Cash Flow', 'Dividends', 'Volume', 'Stock Splits', 'Job_Value', 'Labour_values', 'Manufacturing_Value', 'CPI_Value', 'inflation_Value', 'Housing_value']]")

##################################################


with open('1233.pkl', 'rb') as f:
    mynewlist = pickle.load(f)

stock =st.selectbox('Enter the stock here :',mynewlist)
st.write('The stock name is', stock)

df_get_column=get_columns(stock)
st.dataframe(df_get_column.tail(5))

if st.button('Predict'):
    
    prg = st.progress(0)
    for i in range(100):
        time.sleep(0.001)
        prg.progress(i)
    
    api_key = "L82N6LJGLR5DP8L9"
    ticker_symbol = stock

    url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker_symbol}&apikey={api_key}"
    response = requests.get(url)

    data = response.json()["annualReports"]
    balance_sheet = pd.DataFrame(data).set_index("fiscalDateEnding").iloc[::-1].drop(columns="reportedCurrency")

    url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker_symbol}&apikey={api_key}"
    response = requests.get(url)

    data = response.json()["annualReports"]
    income_statement = pd.DataFrame(data).set_index("fiscalDateEnding").iloc[::-1].drop(columns="reportedCurrency")

    url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker_symbol}&apikey={api_key}"
    response = requests.get(url)

    data = response.json()["annualReports"]
    cash_flow = pd.DataFrame(data).set_index("fiscalDateEnding").iloc[::-1].drop(columns="reportedCurrency")

    previous_row = balance_sheet.iloc[-1].tolist()
    new_df = pd.DataFrame([previous_row], columns=balance_sheet.columns)
    balance_sheet = balance_sheet.reset_index()
    balance_sheet = pd.concat([balance_sheet, new_df], axis=0, ignore_index=True)
    balance_sheet.at[5, 'fiscalDateEnding'] = '2023-12-31'
    balance_sheet.set_index('fiscalDateEnding', inplace=True)

    previous_row = income_statement.iloc[-1].tolist()
    new_df = pd.DataFrame([previous_row], columns=income_statement.columns)
    income_statement = income_statement.reset_index()
    income_statement = pd.concat([income_statement, new_df], axis=0, ignore_index=True)
    income_statement.at[5, 'fiscalDateEnding'] = '2023-12-31'
    income_statement.set_index('fiscalDateEnding', inplace=True)

    previous_row = cash_flow.iloc[-1].tolist()
    new_df = pd.DataFrame([previous_row], columns=cash_flow.columns)
    cash_flow = cash_flow.reset_index()
    cash_flow = pd.concat([cash_flow, new_df], axis=0, ignore_index=True)
    cash_flow.at[5, 'fiscalDateEnding'] = '2023-12-31'
    cash_flow.set_index('fiscalDateEnding', inplace=True)

    dfs = [balance_sheet, income_statement, cash_flow]
    dfs = reduce(lambda left,right: pd.merge(left,right,on='fiscalDateEnding'), dfs)
    dfs = dfs.apply(pd.to_numeric, errors='coerce')
    nan_value_columns=dfs[dfs.columns[dfs.isna().any()]]
    dfs=dfs.drop(nan_value_columns.columns,axis=1)

    #print(sum(np.isnan(dfs)))


    ########Scalar ICA

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(dfs)

    ica = picard.Picard( random_state=42)
    components = ica.fit_transform(data_scaled)

    df_components = pd.DataFrame(components)
    seq = ['2018-12-31', '2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31']
    df_components.insert(0, 'Date', seq)



    ############Merge Data_Frame

    df2=df_components
    df1=get_columns(stock)
    df2=df2.reset_index()
    df2.drop(columns=['index'])

    df2['Date'] = df2['Date'].apply(lambda x: x[:4])
    df1['year'] = pd.to_datetime(df1['Date'], format='%Y-%m-%d').dt.year
    df2['year'] = pd.to_datetime(df2['Date']).dt.year.astype(int)

    df2.rename(columns={0:"0",1: "1",2:"2",3:"3",4:"4"}, inplace=True)
    merged_df = pd.merge(df1, df2, on='year')
    merged_df.drop(['Date_y', 'year'], axis=1, inplace=True)

    #df1=merged_df
    #dfr=merged_df

    #df1=df1.drop(['Open','High','Low','Dividends','NASDAQCOM','index'],axis=1)
    #df1['Date_x'] = pd.to_numeric(df1['Date_x'])
    
    ####################News_Data
    ###############################################################
    url = 'https://eodhistoricaldata.com/api/sentiments?s={}&from=2000-01-01&to=2023-06-30&api_token=64804202d365e9.21480491'.format(stock)
    response = requests.get(url)
    data = response.json()
    ticker = list(data.keys())[0]
    sent_df = pd.DataFrame(data[ticker])

    sent_df = sent_df[sent_df['date'] >= '2017-11-28']
    df_new=merged_df
    sent_df['date'] = pd.to_datetime(sent_df['date'])
    df_new.set_index('Date_x', inplace=True)
    sent_df.set_index('date', inplace=True)

    df_new.index = df_new.index.tz_localize(None)
    new_merged_df = df_new.merge(sent_df, how='outer', left_index=True, right_index=True)
    new_merged_df.fillna(method='ffill', inplace=True)
    new_merged_df.fillna(method='bfill', inplace=True)
    new_merged_df.dropna(inplace=True)
    new_merged_df.reset_index(inplace=True)
    new_merged_df.rename(columns={"level_0":"date"},inplace=True)
    new_merged_df = new_merged_df.iloc[1: , :]
    dfr=new_merged_df

    df1=new_merged_df
    #dfr=merged_df

    df1=df1.drop(['Open','High','Low','Dividends','NASDAQCOM','index','count'],axis=1)
    df1['date'] = pd.to_numeric(df1['date'])
    
    ###################Tweet_Sentiment
    
    url = 'https://eodhistoricaldata.com/api/tweets-sentiments?s={}&from=2000-01-01&to=2023-06-30&api_token=64804202d365e9.21480491'.format(stock)
    response = requests.get(url)
    data = response.json()
    ticker = list(data.keys())[0]
    tweet_df = pd.DataFrame(data[ticker])
    tweet_df = tweet_df.rename({'normalized': 'tweet_normalized'}, axis='columns')
    tweet_df['date'] = pd.to_datetime(tweet_df['date'])
    df_new2 = new_merged_df.merge(tweet_df, how='outer', left_index=True, right_index=True)
    df_new2.fillna(method='ffill', inplace=True)
    df_new2.fillna(method='bfill', inplace=True)
    df_new2.dropna(axis = 0, inplace=True)
    df_new2.rename(columns={"date_x":"date"},inplace=True)
    df_new2 = df_new2.iloc[1: , :]
    df_lstm=df_new2
    dfr=df_new2

    #######################################################################

    ###########Function
    
    df_lstm=df_lstm.drop(['Open','High','Low','Dividends','NASDAQCOM','index','count_x','count_y','date_y'],axis=1)
    df_lstm['date'] = pd.to_numeric(df_lstm['date'])
    df_lstm.dropna(axis = 0, inplace=True)

    def train_lstm(df_lstm):
        features=['tweet_normalized','Close','Volume','normalized','Job_Value','Labour_values',
                'Manufacturing_Value','CPI_Value','inflation_Value','Housing_value','0','1','2','3','4']
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df_lstm[features])
        n_steps = 100 # number of time steps to consider for each input
        n_features = len(features)
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(n_steps, n_features)))
        model.add(Dense(15, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')
        X_train = []
        y_train = []
        for i in range(n_steps, len(df_lstm)-1):
            X_train.append(data_scaled[i-n_steps:i])
            y_train.append(data_scaled[i+1, :]) # predict all 15 columns for next day
        X_train, y_train = np.array(X_train), np.array(y_train)
        print('X_train shape:', X_train.shape)
        print('y_train shape:', y_train.shape)
        model.fit(X_train, y_train, epochs=50, batch_size=32)
        return model, scaler, n_steps, n_features, X_train, y_train

    model, scaler, n_steps, n_features, X_train, y_train = train_lstm(df_lstm)

    def lstm_predict(model, scaler, n_steps, n_features, df_lstm):
        features = ['tweet_normalized', 'Close', 'Volume', 'normalized', 'Job_Value', 'Labour_values', 'Manufacturing_Value', 'CPI_Value', 'inflation_Value', 'Housing_value', '0','1','2','3','4']
        data_scaled = scaler.transform(df_lstm[features])
        X_test = []
        X_test.append(data_scaled[-n_steps:])
        X_test = np.array(X_test)
        y_pred_scaled = model.predict(X_test)
        y_pred = scaler.inverse_transform(y_pred_scaled)

        for i in range(30):
            X_test = np.append(X_test[:,1:,:], y_pred_scaled.reshape(1,1,n_features), axis=1)
            y_pred_scaled = model.predict(X_test[:, -n_steps:, :])
            y_pred = np.append(y_pred, scaler.inverse_transform(y_pred_scaled), axis=0)
            #print(y_pred)

        return y_pred[:,1]  # return Close column values
#######################################################################
    
    df=dfr
    df=df.drop(['Open','High','Low','Dividends','NASDAQCOM','index','index','count_x','date_y','count_y'],axis=1)

    df.columns = df.columns.astype(str)
    


    def forecast_random_pca(df):
        train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

        y_train = train_data['Close']
        y_test = test_data['Close']

        X_train = train_data.drop(['date', 'Close'], axis=1)
        X_test = test_data.drop(['date', 'Close'], axis=1)
        rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
        rf_model.fit(X_train, y_train)

        predictions = rf_model.predict(X_test)

        last_30_days = df.tail(30).drop(['date', 'Close'], axis=1)
        next_30_days_predictions = rf_model.predict(last_30_days)
        return next_30_days_predictions

    Random=forecast_random_pca(df)

    lstm=lstm_predict(model, scaler, n_steps, n_features, df_lstm)
    #b=lstm[0]
    #b=round(b,2)
    #st.write("Next Day Prediction for:")
    #st.write(b)
    #st.write(a[0])
    #average[0]=(lstm[0]+Random[0])/2
    ava=(lstm[0]+Random[0])/2
    avb=(lstm[6]+Random[6])/2
    avc=(lstm[14]+Random[14])/2
    
    
    columns =['Stock', 'Days', 'LSTM', 'Random Forest', 'Average']
    name1=[stock,1,lstm[0],Random[0],ava]
    name2=[stock,7,lstm[6],Random[6],avb]
    name3=[stock,15,lstm[14],Random[14],avc]
    
    alg_vs_score=pd.DataFrame((name1,name2,name3), columns=columns)
    
    st.dataframe(alg_vs_score)
    
    #ticker1 = yf.Ticker(stock).info
    #market_price1 = ticker1['regularMarketPrice']
    #st.write("Actual Market Price:",market_price1)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline

    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


    ticker = yf.Ticker(stock)
    news=ticker.news
    st.header("News :")
    i = 0
    while (i < len(news)-2):
        i = i + 1
        st.markdown("News : {}".format(news[i]["title"]))
        st.markdown("News Link : {}".format(news[i]["link"])) 
        results = nlp(news[i]["title"])
        results[0]["score"] = round(results[0]["score"], 2)
        st.markdown("News Sentiment: **:blue[{}]**".format(results[0]))
        #st.markdown("News Sentiment : {}".format(results))
        st.markdown("""---""")

#if __name__ == '__main__':
#    main()
