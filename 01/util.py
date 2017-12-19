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
        df_tmp = pd.read_csv(symbol_to_path(symbol),
                             index_col='Date',
                             parse_dates=True,
                             usecols=['Date','Adj Close'],
                             na_values=['nan'])
        df_tmp = df_tmp.rename(columns={'Adj Close': symbol})
        df = df.join(df_tmp)
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
    return df / df.iloc[0,:]

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

def port_val(start_val, dates, symbols, allocs):
    if "SPY" not in symbols:
        allocs.insert(0,0)
    #make a df of the adj close
    prices = get_data(symbols, dates)
    fill_missing_values(prices)
    #normalize data
    norm = normalize_data(prices)
    #find allocated ammounts
    alloced = norm * allocs
    #get the position values
    pos_vals = alloced * start_val
    #calculate portfolio values
    port_vals = pos_vals.sum(axis=1)
    return port_vals

def cum_ret(port_vals):
    #division of second to last be first
    return (port_vals[-1] / port_vals[0]) - 1

def avg_daily_ret(daily_ret):
    return daily_ret.mean()

def std_daily_ret(daily_ret):
    return daily_ret.std()

def sharpe_ratio(daily_rets, time="balls"):
    return k * (avg_daily_ret(daily_rets)/std_daily_rets(daily_rets))

