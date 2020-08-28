from pycoingecko import CoinGeckoAPI
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cg = CoinGeckoAPI()

def split_dataset(ticker, denom_currency, days):
    """
    Splits the dataset returned from CoinGecko into Date and Price columns
    
    Parameters:
    ticker -  asset ticker from CoinGecko
    denom_currency - denominated currency
    days - number of days of data

    Returns:
    DataFrame of historical data sorted into 'Date' and 'Price' columns.
    """
    # Inner function calling from CoinGecko
    def select_coin(ticker, denom_currency, days):
        dataset = pd.DataFrame(cg.get_coin_market_chart_by_id(ticker, denom_currency, days))
        return dataset
    
    dataset = select_coin(ticker, denom_currency, days)

    # Converting to DataFrame, parsing 'Date' column, and setting index
    dataset = pd.DataFrame(dataset['prices'].to_list(), columns=['Date', 'Price'])
    dataset['Date'] = pd.to_datetime(dataset['Date'], unit='ms')
    dataset = dataset.set_index('Date')
    return dataset

def return_signals(dataset, short_window, long_window):
    """
    Returns a DataFrame composed of trading signal parameters.
    
    Parameters:
    dataset - Dataset composed of historical prices
    short_window - The period selected for the MA calculation. Must be lesser than long_window.
    long_window - The period selected for the MA calculations. Must be greater than short_window.

    Returns:
    DataFrame of signal parameters
    """
    # Initialising DataFrame with index from DataFrame of historical asset prices
    signals = pd.DataFrame(index=dataset.index)

    # Default set at 0.0. When 'Short Window MA' exceeds 'Long Window MA' for a period 
    # longer than length of 'Short Window MA', a value of 1.0 is returned.    
    signals['Trade Signal'] = 0.0

    # Obtaining the rolling moving averages for the short and long window
    signals['Short Window MA'] = dataset['Price'].rolling(window=short_window, min_periods=1, center=False).mean()
    signals['Long Window MA'] = dataset['Price'].rolling(window=long_window, min_periods=1, center=False).mean()

    # Return 1 when short window moving average exceeds the long window moving average for a minimum period of short window else 0
    signals['Trade Signal'][short_window:] = np.where(signals['Short Window MA'][short_window:] > signals['Long Window MA'][short_window:], 1.0, 0.0) 

    # Represents the point of entry into market. Taken as the difference between 'Trade Signal' today and yesterday.
    # +1 if Buy, -1 if Sell
    signals['Positions'] = signals['Trade Signal'].diff()

    return signals

def plot_signals(dataset, signals, title):
    """
    Returns a plot showing the trading signals superimposed over the asset price, 
    short window moving average, and long window moving average
    
    Parameters:
    dataset - Dataset composed of historical prices
    signals - Dataset composed of trading signals obtained from return_signals(dataset, short_window, long_window)
    title - Title to return to plot.

    Returns:
    Figure of trading signals, asset prices, and moving averages of the asset prices.
    """

    # Initialising the plot
    fig = plt.figure(figsize=(15,12))
    ax1 = fig.add_subplot(111, ylabel='Price in $')

    # Plot the closing price
    dataset['Price'].plot(ax=ax1, color='gold', lw=2)

    # Plot the short and long moving averages
    signals[['Short Window MA', 'Long Window MA']].plot(ax=ax1, lw=2)

    # Plot the buy signals
    ax1.plot(signals.loc[signals['Positions']==1.0].index, signals['Short Window MA'][signals['Positions']==1], '^', markersize=10, color='m')

    # Plot the sell signals
    ax1.plot(signals.loc[signals['Positions']==-1.0].index, signals['Long Window MA'][signals['Positions']==-1], 'v', markersize=10, color='k')
    
    # Plot details
    plt.grid()
    plt.title(title, fontsize=20)
    plt.legend(['Asset Price', 'Short Window MA', 'Long Window MA', 'Buy Signal', 'Sell Signal'])

def portfolio_backtest(dataset, signals, initial_capital, asset_quantity):
    """
    Backtests the trading signals over historical asset prices. 

    Parameters:
    dataset - Dataset composed of historical prices.
    signals - Dataset composed of trading signals obtained from return_signals(dataset, short_window, long_window).
    initial_capital - The starting capital allocated (Ensure sufficient to purchase for the allocated asset_quantity).
    asset_quantity - Quantity of assets to be purchased.

    Returns:
    DataFrame composed of portfolio performance over the time period.
    Figure displaying portfolio performance as a results of the trading strategy
    relative to Buy & Hold strategy
    """
    # Create a DataFrame 'positions'
    positions = pd.DataFrame(index=signals.index).fillna(0.0)

    # Position in chosen asset. If non-zero, means asset has been purchased (Long)
    positions['Position'] = asset_quantity*signals['Trade Signal']

    # Initialize the portfolio with position in asset
    portfolio = positions.mul(dataset['Price'], axis=0)

    # Marks the moment of entry into asset
    position_diff = positions.diff()

    # Add 'holdings' to portfolio
    portfolio['Holdings ($)'] = (positions.mul(dataset['Price'], axis=0)).sum(axis=1)

    # Cash on hand in portfolio
    portfolio['Cash Leftover ($)'] = initial_capital - (position_diff.mul(dataset['Price'], axis=0)).sum(axis=1).cumsum()

    # Total value of portfolio
    portfolio['Total Value ($)'] = portfolio['Cash Leftover ($)']+portfolio['Holdings ($)']

    # Percentage change in returns of holdings 
    portfolio['Returns (%)'] = portfolio['Total Value ($)'].pct_change()*100

    # Visualize the portfolio value over the period
    fig = plt.figure(figsize=(15,12))
    ax1 = fig.add_subplot(111, ylabel='Price in $')
    plt.title('Comparison of Portfolio Returns vs. Buy & Hold Strategy', fontsize=20)

    # Plot the portfolio value vs. buy and hold strategy
    portfolio['Total Value ($)'].plot(ax=ax1, lw=2)
    (dataset['Price']*asset_quantity).plot(ax=ax1)

    # Plotting the buy signals
    ax1.plot(portfolio.loc[signals['Positions']==1.0].index, portfolio['Total Value ($)'][signals['Positions']==1.0], '^', markersize=10, color='m')

    # Plotting the sell signals
    ax1.plot(portfolio.loc[signals['Positions']==-1.0].index, portfolio['Total Value ($)'][signals['Positions']==-1.0], 'v', markersize=10, color='k')

    # Plot details
    ax1.legend(['Trading Strategy', 'Buy & Hold Strategy', 'Buy Signal', 'Sell Signal'])
    plt.show()

    # Profitability of the strategy over the Buy & Hold strategy as of the present date
    print('Profit over Buy & Hold strategy as of ' + str(portfolio.index[-1]) + ': $' + str(round(portfolio['Cash Leftover ($)'].iloc[-1])))
    
    # Returns profitability over the Buy & Hold strategy in terms of percentage
    print('Percentage-wise: ' + 
         str(round(100*portfolio['Cash Leftover ($)'].iloc[-1]/(portfolio['Total Value ($)'].iloc[-1]-portfolio['Cash Leftover ($)'].iloc[-1]), 2)) 
         + '%.')
    
    # Returns the total portfolio value from the strategy itself
    print('Total portfolio value as of ' + str(portfolio.index[-1]) + ': $' +  str(round(portfolio['Total Value ($)'].iloc[-1])))

    # Returns the average number of days with 'long' signal activated
    print(
        'Average number of days with long signal: ' 
        + str((signals['Trade Signal']==1.0).sum()/len(portfolio['Total Value ($)'][signals['Positions']==1.0]))
    )

    # Returns the number of days since the current signal was activated
    if ([signals['Trade Signal']==1.0]):
        print(
            'Number of days since long signal activated: ' + str(portfolio.index.max()-strat_rets.index[signals['Trade Signal']==0.0].max())
        )
    else:
        print(
            'Number of days since short signal activated: ' + str(portfolio.index.max()-strat_rets.index[signals['Trade Signal']==1.0].max())
        )
        
    return portfolio

def annul_sharpe_ratio(portfolio_returns):
    """
    Calculates the annualised Sharpe ratio for the returns of the portfolio (risk-free rate excluded)

    Parameters:
    portfolio_returns - Returns of portfolio obtained from portfolio_backtest(dataset, signals, initial_capital, asset_quantity)

    Returns:
    """
    returns = portfolio_returns['Returns (%)']/100
    return print('Sharpe ratio: ' + str(round(np.sqrt(365)*(returns.mean()/returns.std()), 4)))

def drawdown(dataset):
    """
    Computes the maximum daily drawdown
    Measures the largest single drop from peak to bottom in the value of a portfolio
    
    Parameters:
    dataset - Historical data of portfolio total value

    Returns:
    Plot of daily and maximum daily drawdown of the portfolio value
    """

    # Define a trailing 365 trading day window
    window=365

    # Calculate the max drawdown in the past window days for each day
    rolling_max = dataset.rolling(window, min_periods=1).max()
    daily_drawdown = dataset/rolling_max - 1

    # Calculate the minimum (negative) daily drawdown 
    max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

    # Plot the results
    plt.figure(figsize=(15, 12))
    daily_drawdown.plot()
    max_daily_drawdown.plot()
    plt.legend(['Daily Drawdown', 'Max. Daily Drawdown'])
    
    return print('Maximum Daily Drawdown: ' + str(100*round(max_daily_drawdown.min(), 4)) + '%.')