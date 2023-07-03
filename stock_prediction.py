import os
from sklearn.preprocessing import StandardScaler
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import warnings
warnings.filterwarnings("ignore")
import yfinance as yf
yf.pdr_override()

#import seaborn as sns
import datetime

from sklearn.preprocessing import StandardScaler

import random

from functools import reduce

from sklearn.preprocessing import StandardScaler

import picard

from api_calls import make_api_request_av, make_api_request_eod
from app_functions import *

np.random.seed(42)
random.seed(7)


def stock_main():

    st.write("# Forecasting Time Series Data - Stock Price")

    with open('1233.pkl', 'rb') as f:
        mynewlist = pickle.load(f)

    stock =st.selectbox('Enter the stock here :',mynewlist)
    st.write('The stock name is', stock)

    df_get_column=get_columns(stock)
    st.dataframe(df_get_column.tail(5))

    if st.button('Predict'):
        try:
            with st.spinner("Predicting..."):
                prg = st.empty()

                #url = f"https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker_symbol}&apikey={api_key}"
                flag, response = make_api_request_av(stock=stock, func="BALANCE_SHEET")
                if not flag:
                    st.error("System at capacity....... Please try again later.")
                    st.stop()
                data = response["annualReports"]
                balance_sheet = pd.DataFrame(data).set_index("fiscalDateEnding").iloc[::-1].drop(columns="reportedCurrency")

                #url = f"https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker_symbol}&apikey={api_key}"
                flag, response = make_api_request_av(stock=stock, func="INCOME_STATEMENT")
                if not flag:
                    st.error("System at capacity....... Please try again later.")
                    st.stop()
                data = response["annualReports"]
                income_statement = pd.DataFrame(data).set_index("fiscalDateEnding").iloc[::-1].drop(columns="reportedCurrency")

                #url = f"https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker_symbol}&apikey={api_key}"
                flag, response = make_api_request_av(stock=stock, func="CASH_FLOW")
                if not flag:
                    st.error("System at capacity....... Please try again later.")
                    st.stop()
                data = response["annualReports"]
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


                ########Scalar ICA

                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(dfs)

                ica = picard.Picard( random_state=42, fun='exp')
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

                
                #Sentiment data declaration
                ###############################################################
                
                flag, sentiment_data, tweets_data = make_api_request_eod(stock)#response.json()
                
                if not flag:
                    st.error("System at capacity....... Please try again later.")
                    st.stop()
                
                ticker = list(sentiment_data.keys())[0]
                sent_df = pd.DataFrame(sentiment_data[ticker])

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
                
                ticker = list(tweets_data.keys())[0]
                tweet_df = pd.DataFrame(tweets_data[ticker])
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

                df_lstm=df_lstm.drop(['Open','High','Low','Dividends','NASDAQCOM','index','count_x','count_y','date_y'],axis=1)
                df_lstm['date'] = pd.to_numeric(df_lstm['date'])
                df_lstm.dropna(axis = 0, inplace=True)

                #model training
                model, scaler, n_steps, n_features, X_train, y_train = train_lstm(df_lstm)

                #######################################################################
                
                df=dfr
                df=df.drop(['Open','High','Low','Dividends','NASDAQCOM','index','index','count_x','date_y','count_y'],axis=1)

                df.columns = df.columns.astype(str)
                

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
                
                from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
                from transformers import pipeline

                tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                model = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
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

                st.success("Prediction complete!")    
        except Exception as e:
            st.error("System Issue. Please try again later.")
            st.write(f"Error details: {e}")
            print (e)

