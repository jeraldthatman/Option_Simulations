import matplotlib.pyplot as plt 
from scipy.optimize import minimize
import scipy.stats as st 
import numpy as np 
import pandas as pd 
import datetime as dt 
import yfinance as yf 

print('Downloading Recent 10 year treasury yield data...')
# Start by getting up-to-date risk free rate 
TEN_YEAR = yf.Ticker('^TNX').history()[['Open','Close', 'High', 'Low']]
R = TEN_YEAR['Close'].iloc[-1] / 100

from scipy.optimize import minimize
import scipy.stats as st 

def get_volatility(market_price, price):
    def error(sigma):
        return (market_price - price)**2
    return minimize(error, 0.2).x[0]

def phi(df):
    df['d1'] = (np.log(df['stk_price']/df['strike']) + (R + df['impliedvolatility']**2/2)*df['timeValue']) / (df['impliedvolatility'] * np.sqrt(df['timeValue']))
    df['d2'] = df['d1'] - df['impliedvolatility'] * np.sqrt(df['timeValue'])
    df['nd1'] = st.norm.cdf(df['d1'])
    df['nd2'] = st.norm.cdf(df['d2'])
    return df

def call_options(df):
    df = df.copy()
    df = phi(df)
    df['theoPrice'] = df['stk_price'] * df['nd1'] - df['strike'] * np.exp(-R*df['timeValue']) * df['nd2']
    df['delta'] = df['nd1']
    df['gamma'] = st.norm.pdf(df['d1']) / (df['stk_price'] * df['impliedvolatility'] * np.sqrt(df['timeValue']))
    df['theta'] = -df['stk_price'] * st.norm.pdf(df['d1']) * df['impliedvolatility'] / (2 * np.sqrt(df['timeValue'])) - R * df['strike'] * np.exp(-R*df['timeValue']) * df['nd2']
    df['vega'] = df['stk_price'] * st.norm.pdf(df['d1']) * np.sqrt(df['timeValue'])
    df['rho'] = df['strike'] * df['timeValue'] * np.exp(-R*df['timeValue']) * df['nd2']
    return df

def put_options(df):
    df = df.copy()
    df = phi(df)
    df['theoPrice'] = df['strike'] * np.exp(-R*df['timeValue']) * df['nd2'] - df['stk_price'] + df['strike'] * np.exp(-R*df['timeValue'])
    df['delta'] = -df['nd1']
    df['gamma'] = st.norm.pdf(df['d1']) / (df['stk_price'] * df['impliedvolatility'] * np.sqrt(df['timeValue']))
    df['theta'] = -df['stk_price'] * st.norm.pdf(df['d1']) * df['impliedvolatility'] / (2 * np.sqrt(df['timeValue'])) + R * df['strike'] * np.exp(-R*df['timeValue']) * df['nd2']
    df['vega'] = df['stk_price'] * st.norm.pdf(df['d1']) * np.sqrt(df['timeValue'])
    df['rho'] = -df['strike'] * df['timeValue'] * np.exp(-R*df['timeValue']) * df['nd2']
    return df

def bs_df(df):
    df = df.copy()
    calls = df[df['type'] == 'Call'].copy()
    puts = df[df['type'] == 'Put'].copy()
    return pd.concat([call_options(calls), put_options(puts)])



def contract_overview(tmp, date_found = None):
    '''Return a plot of the contract history, stock price, and contract volatility, Delta, Open Interest and Volume given a contract symbol. Enter tmp, a dataframe of the contract history, and the date that your signal is found. '''
    plt.style.use('seaborn')


    if date_found == None:
        date_found = str(tmp.index.get_level_values(1).max())[:10]
        
    fig, axes = plt.subplots(1, 5, figsize = (30,4))
    # Plot stock price
    axes[0].plot(tmp.index.get_level_values(1), tmp.stk_price, label = 'Stock')
    axes[0].vlines(dt.datetime.strptime(date_found, '%Y-%m-%d'), tmp.stk_price.min(), tmp.stk_price.max(), color = 'grey')
    # Plot contract value 
    axes[1].plot(tmp.index.get_level_values(1), tmp.lastprice, color = 'Black', label = 'Last Price')
    axes[1].plot(tmp.index.get_level_values(1), tmp.theoPrice, color = 'red', label = 'Theoretical Price')
    axes[1].plot(tmp.index.get_level_values(1), tmp.lastprice.rolling(3).mean(), color = 'lightgreen', label = '3 SMA')
    axes[1].plot(tmp.index.get_level_values(1), tmp.lastprice.rolling(6).mean(), color = 'Green', label = '6 SMA')
    axes[1].vlines(dt.datetime.strptime(date_found, '%Y-%m-%d'), tmp.lastprice.min(), tmp.lastprice.max(), color = 'grey')
    # Plot volaitlity
    axes[2].plot(tmp.index.get_level_values(1), tmp.impliedvolatility, label = 'Implied Volatility')
    axes[2].plot(tmp.index.get_level_values(1), tmp.historicalvolatility, label = 'Historical Volatility')
    axes[2].plot(tmp.index.get_level_values(1), tmp.vega* 100 , label = 'vega')

    # Plot Volume and open interest 
    axes[3].plot(tmp.index.get_level_values(1), tmp.volume, label = 'Volume', c = 'orange')
    axes[3].plot(tmp.index.get_level_values(1), tmp.openinterest, label = 'Open Interest',c='purple')
    axes[3].vlines(dt.datetime.strptime(date_found, '%Y-%m-%d'), 0, tmp.openinterest.max(), color = 'grey')
     # Plot Delta 
    axes[4].plot(tmp.index.get_level_values(1), tmp.delta * 100, label = 'Delta', c = 'red')
    axes[4].plot(tmp.index.get_level_values(1), tmp.gamma* 100 , label = 'gamma')
    axes[4].vlines(dt.datetime.strptime(date_found, '%Y-%m-%d'), 0, tmp.delta.max(), color = 'grey')

    axes[0].set_title(f'Stock Price ${tmp.stk_price.iloc[-1]:.2f}')
    axes[1].set_title('Contract Last Price')
    axes[2].set_title('Volatility')
    axes[2].legend(loc = 'upper right')
    axes[3].set_title('Volume and Open Interest')
    axes[4].legend(loc = 'upper right')
    axes[4].set_title('Delta and Gamma')

    for x in axes:
        x.legend()
    fig.autofmt_xdate()
    #stock, contract_type, strike, expiry = describe_option(c)
    #fig.suptitle(f'Stock: {stock}, ${strike} {contract_type}, Expiration: {expiry}', fontsize = 18)
    c = tmp.index.get_level_values(0)[0]
    fig.suptitle(f'{c} Last: ${tmp.lastprice.iloc[-1]:.02f}', fontsize = 16)