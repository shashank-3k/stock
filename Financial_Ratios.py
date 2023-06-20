import streamlit as st
import pandas as pd
from functools import reduce
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import yfinance as yf
import numpy as np
import pandas as pd

import yfinance as yf
yf.pdr_override()

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")

import pickle
import requests

from api_calls import make_api_request_av

with open('1233.pkl', 'rb') as f:
    mynewlist = pickle.load(f)

def x(date):
    date=str(date)
    date=date.split("-")
    return date[0]
    
def y(fiscalDateEnding):
    fiscalDateEnding=str(fiscalDateEnding)
    fiscalDateEnding=fiscalDateEnding.split("-")
    return fiscalDateEnding[0]

def financial_main():
    st.title('Predictive Investing: The Next Frontier in Financial Markets')

    stock =st.selectbox('Enter the stock here :',mynewlist)
    st.write('The stock name is', stock)

    data3 = None
    ratios2 = None
    merged_df = None 
    if st.button('Predict'):
        try:
            with st.spinner('Wait for the results...'):

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
                flag, response = make_api_request_av(stock=stock, func="INCOME_STATEMENT")
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
                dfs = reduce(lambda left,right: pd.merge(left,right,on='fiscalDateEnding', suffixes=('', '_y')), dfs)
                dfs.drop(dfs.filter(regex='_y$').columns, axis=1, inplace=True)
                dfs = dfs.apply(pd.to_numeric, errors='coerce')
                nan_value_columns=dfs[dfs.columns[dfs.isna().any()]]
                dfs=dfs.drop(nan_value_columns.columns,axis=1)
                #dfs.to_csv("dfs_fin_ratio.csv")
                ratios = pd.DataFrame()
                ratios['ROE'] = dfs['netIncome'] / dfs['totalShareholderEquity']
                ratios['Net Profit Margin'] = dfs['netIncome'] / dfs['totalRevenue']
                ratios['Total Assets to Equity'] = dfs['totalAssets'] / dfs['totalShareholderEquity']
                ratios['Asset Turnover'] = dfs['totalRevenue'] / dfs['totalAssets']
                ratios['Capital Structure Impact'] = (dfs['longTermDebt'] + dfs['totalNonCurrentLiabilities']) / dfs['totalShareholderEquity']
                ratios['Tax Ratio'] = dfs['incomeTaxExpense'] / dfs['incomeBeforeTax']
                ratios['PPE / Capital Asset Turnover'] = (dfs['propertyPlantEquipment'] - dfs['accumulatedDepreciationAmortizationPPE']) / dfs['totalRevenue']
                ratios['Working Capital Turnover'] = dfs['totalRevenue'] / (dfs['totalCurrentAssets'] - dfs['totalCurrentLiabilities'])
                ratios['Gross Margin'] = dfs['grossProfit'] / dfs['totalRevenue']
                ratios['SG&A'] = dfs['sellingGeneralAndAdministrative'] / dfs['totalRevenue']
                ratios['Depreciation & Amortization'] = dfs['depreciationAndAmortization'] / dfs['totalRevenue']
                ratios['R&D'] = dfs['researchAndDevelopment'] / dfs['totalRevenue']
                ratios['Payable Turnover'] = dfs['costOfRevenue'] / dfs['currentAccountsPayable']
                ratios['Inventory Turnover'] = dfs['costofGoodsAndServicesSold'] / dfs['inventory']
                ratios['Receivable Turnover'] = dfs['totalRevenue'] / dfs['currentNetReceivables']
                msft = yf.Ticker(stock)
                data = msft.history(start="2018-05-30")
                close = data.iloc[-1]["Close"]
                
                data= data.reset_index()
                ratios= ratios.reset_index()

                ratios["Year"]=ratios['fiscalDateEnding'].apply(x)
                data["Year"]=data['Date'].apply(x)
                merged_df = pd.merge(data, ratios, on='Year')


                data2 = data 
                data2 = data2.drop(['Dividends','Stock Splits','Year'],axis=1)
                data3 = data2.round(2)

                
                ratios = ratios.drop(['fiscalDateEnding'],axis=1)
                ratios2 = ratios.round(2)

                st.subheader('Stock Name : {}'.format(stock))
                st.subheader('Historical Stock Price')
                st.write(data3.tail())
                st.subheader('Financial Ratios')
                st.write(ratios2.tail())



                merged_df['target'] = merged_df['Close'].shift(-1)
                merged_df = merged_df.dropna(subset=['target'])

                features = ['ROE', 'Net Profit Margin', 'Total Assets to Equity', 'Asset Turnover', 'Capital Structure Impact', 
                            'Tax Ratio', 'PPE / Capital Asset Turnover', 'Working Capital Turnover', 'Gross Margin', 'SG&A', 
                            'Depreciation & Amortization', 'R&D', 'Payable Turnover', 'Inventory Turnover', 'Receivable Turnover',
                            'Open', 'High', 'Low', 'Volume']
                    
                X = merged_df[features].values
                y = merged_df['target'].values

                #data split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Linear Regression model
                lin_model = LinearRegression()
                lin_model.fit(X_train, y_train)

                last_data = merged_df[features].tail(1).values
                next_day_price = lin_model.predict(last_data)

                
                # Ridge Regression model
                alpha = 1.0  
                ridge_model = Ridge(alpha=alpha)
                ridge_model.fit(X_train, y_train)

                last_data2 = merged_df[features].tail(1).values
                next_day_price2 = ridge_model.predict(last_data2)


                # Randomforest  Regression model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                last_data3 = merged_df[features].tail(1).values
                next_day_price3 = model.predict(last_data3)


                # Decision tree 
                model = DecisionTreeRegressor(random_state=42)
                model.fit(X_train, y_train)

                last_data4 = merged_df[features].tail(1).values
                next_day_price4 = model.predict(last_data4)


                # elesticnet 
                model = ElasticNet(random_state=42)
                model.fit(X_train, y_train)

                last_data5 = merged_df[features].tail(1).values
                next_day_price5 = model.predict(last_data5)


                #Lasso 
                model = Lasso(alpha=0.1)
                model.fit(X_train, y_train)

                last_data6 = merged_df[features].tail(1).values
                next_day_price6 = model.predict(last_data6)

                #SVM
                model = SVR(kernel='rbf', C=1e3, gamma=0.1)
                model.fit(X_train, y_train)

                last_data7 = merged_df[features].tail(1).values
                next_day_price7 = model.predict(last_data7)


                # Define the results dictionary
                results_dict = {
                    'Stock': stock,
                    'Model': ['LinearRegression', 'Ridge', 'RandomForestRegressor','DecisionTree','ElasticNet','Lasso','SVM'],
                    'Closing Price' : close,
                    'Predicted price for the next day': [round(next_day_price[0],2), next_day_price2[0], next_day_price3[0], next_day_price4[0], next_day_price5[0], next_day_price6[0],next_day_price7[0]],

                }

                # Create a pandas DataFrame from the results dictionary
                results_df = pd.DataFrame(results_dict)
                results_df = results_df.round(2)

            
                st.title('Stock Prediction Results')
                st.write('Here are the results of our stock prediction model:')

                # Display the results DataFrame

                st.dataframe(results_df)


        except Exception as e:
            st.error("System Issue. Please try again later.")
            st.write(f"Error details: {e}")
