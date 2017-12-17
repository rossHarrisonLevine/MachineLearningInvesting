import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

def symbol_to_path(symbol,base_dir="data"):
    #return path to cvs
    return os.path.join(base_dir,"{}.csv".format(str(symbol)))

def get_data(symbols,dates):
    df = pd.DataFrame(index=dates)
    if 'SPY' not in symbols: #add SPY for referanece
        symbols.insert(0,'SPY')
    
    for symbol in symbols:
        tmp_df = pd.read_csv(symbol_to_path(symbol),
                            usecols=['Date','Adj Close'],
                            na_values=['nan'],
                            index_col="Date",parse_dates=True)
        tmp_df = tmp_df.rename(columns={'Adj Close': symbol})
        df = df.join(tmp_df)
        if symbol == 'SPY':
            df = df.dropna(subset=['SPY'])
    return df

def plot_data(df, title="Stock Prices", xlabel="Price", ylabel="Date"):
    ax = df.plot(title=title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def plot_selected(df,columns,start_index,end_index):
    #given the start and end dates, plot the relevant info
    plot_data(df.ix[start_index:end_index,columns], title="Selected Data")

#makes all graphs start at 1
def normalize_data(df):
    return df / df.ix[0,:]

def get_rolling_mean(values,window):
    return pd.rolling_mean(values,window=window)

def get_rolling_std(values,window):
    return pd.rolling_std(values,window=window)

def get_bollinger_bands(rm,rstd):
    upper_band = 2*rstd+rm
    lower_band = -2*rstd+rm
    return upper_band, lower_band


def compute_daily_returns(df):
    daily_returns = (df / df.shift(1)) - 1
    daily_returns.ix[0,:] = 0
    return daily_returns

#fills in any missing values
def fill_missing_values(df_data):
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)