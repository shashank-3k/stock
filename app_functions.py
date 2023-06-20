import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import yfinance as yf
import random
import datetime
np.random.seed(42)
random.seed(7)


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
    model.fit(X_train, y_train, epochs=30, batch_size=32)
    return model, scaler, n_steps, n_features, X_train, y_train

###################################################################################################

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

###################################################################################################

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

###################################################################################################

