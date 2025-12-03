import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import StandardScaler

def ingestion(filename='data/market_data_ml.csv'):
    tickers = get_ticker()
    df = pd.read_csv(filename,index_col='date')
    processed = []
    for ticker, group in df.groupby('ticker'):
        processed_df = prepare_features(group.copy())
        processed.append(processed_df)
    final_df = pd.concat(processed)
    scaled = ['ticker', 'open', 'high', 'low', 'close', 'volume', 'Target_Class']
    feature_cols = [col for col in final_df.columns if col not in scaled]

    scaler = StandardScaler()
    #Fit transformation
    for col in feature_cols:
        final_df[col] = scaler.fit_transform(final_df[col].values.reshape(-1, 1))
    return final_df

def prepare_features(df:pd.DataFrame):
    #Return and log returns
    df['Daily_Return'] = df['close'].pct_change()
    df['Log_Return'] = np.log(df['close']/df['close'].shift(1))

    #Lag Features
    df['Lag1_Return'] = df['Daily_Return'].shift(1)
    df['Lag3_Return'] = df['Daily_Return'].shift(3)
    df['Lag5_Return'] = df['Daily_Return'].shift(5)

    #Technical Indicators
    df['SMA10'] = df['close'].rolling(10).mean()
    
    macd = MACD(close=df['close'], window_fast=12, window_slow=26, window_sign=9)    
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()

    # Labeling (Classification)
    df['Next_Day_Return'] = df['close'].shift(-1)
    df['Target_Class'] = (df['Next_Day_Return'] > 0).astype(int)
    df.drop(columns=['Next_Day_Return'], inplace=True)
    
    #Remove all NaN values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df
 
                   
def get_ticker():
    df = pd.read_csv('data/tickers-1.csv')
    return df['symbol'].to_list()

