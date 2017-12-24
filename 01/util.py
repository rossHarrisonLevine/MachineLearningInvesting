import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import scipy.optimize as spo

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

def sharpe_ratio(daily_rets, time="daily"):
    if time == "daily":
        k = 15.8745
    elif time == "weekly":
        k = 7.2111
    elif time == "monthly":
        k = 3.4641
    else:
        k = 1 
    return k * (avg_daily_ret(daily_rets)/std_daily_ret(daily_rets))

def error(line, data):
    """Compute error between given line model and observed data
    
    Parameters
    -----------
    line: tuple/list/array (C0,C1) where C0 is slope and C1 is Y-intercept
    data: 2D array where each row is a point (x,y) 
    
    Returns error as a single real value
    """
    #Metric: sum of squared Y-axis differences
    err = np.sum((data[:,1] - (line[0] * data[:,0] + line[1])) ** 2)
    return err

def fit_line(data, error_func):
    """Fit a line to given data, using a supplied error function.
    
    Parameters
    -----------
    data: 2D array where each row is a point (X0,Y)
    error_func: function that computes the error between a line and observed data
    
    Returns line that minimizes the error function.
    """
    
    #Generate initial guess for the line model
    l = np.float32([0, np.mean(data[:,1])]) #slope = 0, intercept = mean(y values)
    
    #plot initial guess
    x_ends = np.float32([-5,5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth = 2.0, label = "Initial Guess")
    
    #call optimizer to minimize error function
    result = spo.minimize(error_func,l,args=(data,), method = 'SLSQP', options={'disp':True})
    return result.x

def error_poly(C,data):
    """
    Compute error betwen given polynomial and observed data.
    
    Parameters
    ----------
    C: numpy.poly1d object or equivalent array representing polynomial coefs
    data: 2D array where each row is a point (x,y)
    
    Returns error as a single real value.
    """
    
    #Metric: Sum of squared Y-axis differences
    err = np.sum((data[:,1] - np.polyval(C, data[:,0])) ** 2)
    return err

def fit_poly(data, error_func, degree = 3):
    """Fit a polynomial to given data, usinng supplied error function.
    
    Parameters
    ----------
    data: 2D array where each row is a point (x,y)
    error_func: function that computes error betwen given polynomial and observed data.
    
    Returns polynomial that minimizes the error function."""
    
    #Generate initial guess for polynomial model (all coefs = 1)
    Cguess = np.poly1d(np.ones(degree + 1, dtype = np.float32))
    
    #plot initial guess
    x = np.linspace(-5,5,21)
    plt.plot(x, np.polyval(guess,x), 'm--', linewidth = 2.0, label = "Inital Guess")
    
    #call optimizer to minimize error funciton
    result = spo.minimize(error_func, Cguess, args=(data,), method = 'SLSQP', options={'disp':True})
    return np.poly1d(result.x) #converrt optimal result into a poly1D object
