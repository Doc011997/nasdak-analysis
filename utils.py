#utils.py

import pandas as pd
import io
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.dates as mdates
import plotly.graph_objs as go


def read_data(directory_path, filenames, columns_schema=None):
    dfs = {}
    for filename in filenames:
        file_path = f"{directory_path}/{filename}"
        df_name = filename.split(".")[0]
        df = pd.read_csv(file_path)
        if columns_schema:
            df = df.astype(columns_schema)
        dfs[df_name] = df
    return dfs

def display_df_info(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    return s

def explore_data(dfs):
    for name, df in dfs.items():
        print(f"Exploring {name}:\n")
        print(df.info())
        print(df.describe())
        print("\n")

def get_first_and_last_rows(dfs, num_rows=40):
    result = {}
    for name, df in dfs.items():
        result[name] = pd.concat([df.head(num_rows), df.tail(num_rows)])
    return result

def get_num_observations(dfs):
    result = {}
    for name, df in dfs.items():
        result[name] = df.shape[0]
    return result

def deduce_period(dfs, date_column='Date'):
    periods = {}
    for name, df in dfs.items():
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(by=date_column)
        diff = df[date_column].diff().dropna()
        periods[name] = diff.mode()[0]
    return periods

def descriptive_statistics(dfs):
    stats = {}
    for name, df in dfs.items():
        stats[name] = df.describe()
    return stats

def missing_values(dfs):
    missing_vals = {}
    for name, df in dfs.items():
        missing_vals[name] = df.isnull().sum()
    return missing_vals

def calculate_correlation(dfs):
    correlations = {}
    for name, df in dfs.items():
        numeric_df = df.select_dtypes(include=['number'])
        corr = numeric_df.corr()
        correlations[name] = corr

        # Plotting correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
        plt.title(f'Correlation Matrix for {name}')
        plt.show()
        
    return correlations

def calculate_daily_return(dfs, open_col='Open', close_col='Close'):
    for name, df in dfs.items():
        df['Daily_Return'] = (df[close_col] - df[open_col]) / df[open_col]
    return dfs

def calculate_moving_average(dfs, column, window):
    for name, df in dfs.items():
        df[f'{column}_Moving_Avg'] = df[column].rolling(window=window).mean()
    return dfs

def calculate_return_rate(dfs, date_column='Date', open_col='Open', close_col='Close'):
    for name, df in dfs.items():
        df[date_column] = pd.to_datetime(df[date_column])
        df['Return_Rate'] = (df[close_col] - df[open_col]) / df[open_col]
    return dfs

def best_return_rate_stock(dfs, start_date, period, date_column='Date', close_col='Close'):
    # Ensure the start_date is in datetime format
    start_date = pd.to_datetime(start_date)
    
    best_returns = {}
    for name, df in dfs.items():
        df[date_column] = pd.to_datetime(df[date_column])
        filtered_df = df[df[date_column] >= start_date]
        
        if filtered_df.empty:
            print(f"No data available for {name} after {start_date}.")
            continue
        
        if period == 'month':
            filtered_df['Period'] = filtered_df[date_column].dt.to_period('M')
        elif period == 'year':
            filtered_df['Period'] = filtered_df[date_column].dt.to_period('Y')
        else:
            raise ValueError("Unsupported period")
        
        period_return = filtered_df.groupby('Period')[close_col].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0])
        
        if period_return.empty:
            print(f"No return data available for {name} in the given periods.")
            continue
        
        best_period = period_return.idxmax()
        best_returns[name] = (best_period, period_return[best_period])
        
        print(f"{name} - Best period: {best_period}, Return: {period_return[best_period]}")
        
    return best_returns


def calculate_pairwise_correlation(dfs, stock1, stock2, value_column='Close'):
    df1 = dfs[stock1]
    df2 = dfs[stock2]
    
    # Ensure the date column is datetime
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    merged_df = pd.merge(df1[['Date', value_column]], df2[['Date', value_column]], on='Date', suffixes=(f'_{stock1}', f'_{stock2}'))
    
    # Calculate correlation only for numeric columns
    correlation = merged_df[[f'{value_column}_{stock1}', f'{value_column}_{stock2}']].corr().iloc[0, 1]
    return correlation

def average_prices(dfs, period='M', date_column='Date', price_columns=['Open', 'Close']):
    average_prices = {}
    for name, df in dfs.items():
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
        avg_price = df[price_columns].resample(period).mean()
        average_prices[name] = avg_price
    return average_prices

def stock_with_highest_daily_return(dfs, date_column='Date', close_col='Close'):
    highest_returns = {}
    for name, df in dfs.items():
        df[date_column] = pd.to_datetime(df[date_column])
        df['Daily_Return'] = df[close_col].pct_change()
        max_return_date = df.loc[df['Daily_Return'].idxmax(), date_column]
        highest_returns[name] = (max_return_date, df['Daily_Return'].max())
    return highest_returns

def average_daily_return(dfs, period='M', date_column='Date', close_col='Close'):
    average_returns = {}
    for name, df in dfs.items():
        df[date_column] = pd.to_datetime(df[date_column])
        df['Daily_Return'] = df[close_col].pct_change()
        avg_return = df.resample(period, on=date_column)['Daily_Return'].mean()
        average_returns[name] = avg_return
    return average_returns



#############

def compare_stocks(dfs, stocks, criteria='Close'):
    comparisons = {}
    seen_pairs = set()
    for stock1 in stocks:
        for stock2 in stocks:
            if stock1 != stock2 and (stock2, stock1) not in seen_pairs:
                correlation = calculate_pairwise_correlation(dfs, stock1, stock2, value_column=criteria)
                comparisons[f'{stock1} vs {stock2}'] = correlation
                seen_pairs.add((stock1, stock2))
    return comparisons


def calculate_logarithmic_return(df, value_column='Close'):
    df['Log_Return'] = np.log(df[value_column] / df[value_column].shift(1))
    return df


def plot_trends(dfs, stock, value_column='Close'):
    df = dfs[stock]
    df['Date'] = pd.to_datetime(df['Date'])
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df[value_column], label=stock)
    plt.title(f'Trend for {stock}')
    plt.xlabel('Date')
    plt.ylabel(value_column)
    plt.legend()
    st.pyplot(plt)



def plot_candlestick(dfs, stock):
    df = dfs[stock]
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'])])
    fig.update_layout(title=f'Candlestick Chart for {stock}', xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)


def calculate_rsi(df, window=14, value_column='Close'):
    delta = df[value_column].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def plot_rsi(dfs, stock):
    df = dfs[stock]
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = calculate_rsi(df)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['RSI'], label='RSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title(f'RSI for {stock}')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    
    # Format the x-axis dates for good display
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show dates every 3 months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    
    st.pyplot(plt)


def calculate_bollinger_bands(df, window=20, value_column='Close'):
    df['SMA'] = df[value_column].rolling(window=window).mean()
    df['STD'] = df[value_column].rolling(window=window).std()
    df['Upper Band'] = df['SMA'] + (df['STD'] * 2)
    df['Lower Band'] = df['SMA'] - (df['STD'] * 2)
    return df



def plot_bollinger_bands(dfs, stock):
    df = dfs[stock]
    
    # Ensure 'Date' column is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    
    df = calculate_bollinger_bands(df)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.plot(df['Date'], df['Upper Band'], label='Upper Band', linestyle='--')
    plt.plot(df['Date'], df['Lower Band'], label='Lower Band', linestyle='--')
    plt.fill_between(df['Date'], df['Upper Band'], df['Lower Band'], color='gray', alpha=0.3)
    
    plt.title(f'Bollinger Bands for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Format the x-axis dates for better display
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))  # Show dates every 3 months
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))  # Format as Year-Month
    plt.gcf().autofmt_xdate()  # Rotate date labels for better readability
    
    st.pyplot(plt)

def sharpe_ratio(returns, risk_free_rate=0.01):
    return (returns.mean() - risk_free_rate) / returns.std()

def sortino_ratio(returns, risk_free_rate=0.01):
    negative_returns = returns[returns < 0]
    return (returns.mean() - risk_free_rate) / negative_returns.std()

def beta(returns, market_returns):
    covariance = np.cov(returns, market_returns)[0, 1]
    market_variance = np.var(market_returns)
    return covariance / market_variance


