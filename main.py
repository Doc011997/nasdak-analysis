#main.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    read_data,
    explore_data,
    get_first_and_last_rows,
    get_num_observations,
    deduce_period,
    descriptive_statistics,
    missing_values,
    calculate_correlation,
    calculate_daily_return,
    calculate_moving_average,
    calculate_return_rate,
    best_return_rate_stock,
    calculate_pairwise_correlation,
    average_prices,
    stock_with_highest_daily_return,
    average_daily_return,
    calculate_logarithmic_return,
    plot_trends,
    compare_stocks,
    display_df_info,
    plot_bollinger_bands,
    plot_candlestick,
    plot_rsi,
    sharpe_ratio,
    sortino_ratio, 
    beta
)

def main():
    # Directory path and filenames
    directory_path = 'stocks_data'
    filenames = [
        'AMAZON.csv',
        'APPLE.csv',
        'FACEBOOK.csv',
        'GOOGLE.csv',
        'MICROSOFT.csv',
        'TESLA.csv',
        'ZOOM.csv'
    ]

    # Read data
    dfs = read_data(directory_path, filenames)

    # Streamlit UI
    st.title("Stock Data Analysis Application")

    st.sidebar.title("Options")
    option = st.sidebar.selectbox("Choose an analysis option", [
        "Home",
        "Explore Data",
        "First and Last Rows",
        "Number of Observations",
        "Data Periods",
        "Descriptive Statistics",
        "Missing Values",
        "Correlation Matrices",
        "Daily Returns",
        "Moving Averages",
        "Return Rates",
        "Best Return Rate Stock",
        "Pairwise Correlation",
        "Average Prices",
        "Stock with Highest Daily Return",
        "Average Daily Return",
        "Logarithmic Returns",
        "Plot Trends",
        "Compare Stocks",
        "RSI",
        "Candlestick Chart",
        "Bollinger Bands", 
        "Portfolio Management"
    ])

    if option == "Home":
    # Existing home page code
        st.title("Empirical Financial Analysis Project")
        st.write("Name: Pierre BENJAMIN")
        st.write("Student Number: 20211369")
        
        st.write("""
        ## Introduction
        Welcome to the home page of my ITF project. In this project, I applied concepts from pandas and numpy to analyze a set of Nasdaq tech stocks. The primary objective was to develop a Python application capable of providing valuable insights for investors or portfolio analysts.

        ### Achievements
        Through this project, I accomplished the following steps:

        1. **Data Exploration**: Reading CSV files, creating DataFrames, and verifying column types.
        2. **Pre-processing**: Cleaning and preparing the data for analysis.
        3. **Data Analysis**: Calculating descriptive statistics, analyzing correlations, and extracting key information.
        4. **Visualizations**: Creating interactive charts to illustrate trends and insights from the data.
        5. **Advanced Functions**: Developing functions to calculate moving averages, return rates, and identifying the stocks with the best returns.

        ### Project Objectives
        - Apply pandas and numpy concepts in a real use case.
        - Build a generic application that can be used to analyze other stocks and periods without major modifications.
        - Provide useful insights for investors and portfolio analysts.

        ### Innovations and Insights
        - Automatic extraction of useful information such as daily, monthly, and annual returns.
        - Visualization of Bollinger Bands and RSI indices to aid in decision-making.
        - Analysis of correlation between different stocks to understand their relationships.

        I hope this application provides you with valuable information and helps you make informed investment decisions.
        """)

    elif option == "Explore Data":
        st.header("Explore Data")
        for name, df in dfs.items():
            st.subheader(f"{name}")
            st.text(display_df_info(df))
            st.write(df.describe())

    elif option == "First and Last Rows":
        st.header("First and Last Rows")
        first_last_rows = get_first_and_last_rows(dfs)
        for name, rows in first_last_rows.items():
            st.subheader(f"First and Last rows of {name}")
            st.write(rows)

    elif option == "Number of Observations":
        st.header("Number of Observations")
        num_observations = get_num_observations(dfs)
        st.write(num_observations)

    elif option == "Data Periods":
        st.header("Data Periods")
        periods = deduce_period(dfs)
        st.write(periods)

    elif option == "Descriptive Statistics":
        st.header("Descriptive Statistics")
        stats = descriptive_statistics(dfs)
        for name, stat in stats.items():
            st.subheader(f"{name}")
            st.write(stat)

    elif option == "Missing Values":
        st.header("Missing Values")
        missing_vals = missing_values(dfs)
        st.write(missing_vals)

    elif option == "Correlation Matrices":
        st.header("Correlation Matrices")
        correlation_matrices = calculate_correlation(dfs)
        for name, matrix in correlation_matrices.items():
            st.subheader(f"Correlation Matrix for {name}")
            st.write(matrix)
            plt.figure(figsize=(10, 8))
            sns.heatmap(matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True)
            st.pyplot(plt)

    elif option == "Daily Returns":
        st.header("Daily Returns")
        dfs = calculate_daily_return(dfs)
        for name, df in dfs.items():
            st.subheader(f"Daily returns of {name}")
            st.write(df.head())

    elif option == "Moving Averages":
        st.header("Moving Averages")
        window = st.sidebar.slider("Select moving average window", 1, 50, 20)
        dfs = calculate_moving_average(dfs, 'Open', window)
        for name, df in dfs.items():
            st.subheader(f"Moving average of {name}")
            st.write(df.head(50))

    elif option == "Return Rates":
        st.header("Return Rates")
        dfs = calculate_return_rate(dfs)
        for name, df in dfs.items():
            st.subheader(f"Return rate of {name}")
            st.write(df.head())

    elif option == "Best Return Rate Stock":
        st.header("Best Return Rate Stock")
        start_date = st.sidebar.date_input("Start date", pd.to_datetime("2023-01-01"))
        period = st.sidebar.selectbox("Period", ["month", "year"])
        best_stock = best_return_rate_stock(dfs, start_date, period)
        st.write(best_stock)

    elif option == "Pairwise Correlation":
        st.header("Pairwise Correlation")
        stock1 = st.sidebar.selectbox("Stock 1", filenames)
        stock2 = st.sidebar.selectbox("Stock 2", filenames)
        correlation = calculate_pairwise_correlation(dfs, stock1.split('.')[0], stock2.split('.')[0])
        st.write(f"Correlation between {stock1.split('.')[0]} and {stock2.split('.')[0]}: {correlation}")

    elif option == "Average Prices":
        st.header("Average Prices")
        avg_prices = average_prices(dfs)
        for name, avg_price in avg_prices.items():
            st.subheader(f"Average prices for {name}")
            st.write(avg_price)

    elif option == "Stock with Highest Daily Return":
        st.header("Stock with Highest Daily Return")
        highest_daily_return = stock_with_highest_daily_return(dfs)
        st.write(highest_daily_return)

    elif option == "Average Daily Return":
        st.header("Average Daily Return")
        avg_daily_return = average_daily_return(dfs)
        for name, avg_return in avg_daily_return.items():
            st.subheader(f"Average daily return for {name}")
            st.write(avg_return)

    elif option == "Logarithmic Returns":
        st.header("Logarithmic Returns")
        for name, df in dfs.items():
            df = calculate_logarithmic_return(df)
            st.subheader(f"Logarithmic returns of {name}")
            st.write(df.head())
    
    elif option == "Plot Trends":
        st.header("Plot Trends")
        stock = st.sidebar.selectbox("Select stock", filenames)
        plot_trends(dfs, stock.split('.')[0])



    elif option == "Compare Stocks":
        st.header("Compare Stocks")
        stocks = st.sidebar.multiselect("Select stocks to compare", filenames)
        criteria = st.sidebar.selectbox("Criteria", ['Close', 'Open', 'High', 'Low'])
        if len(stocks) > 1:
            comparisons = compare_stocks(dfs, [s.split('.')[0] for s in stocks], criteria)
            st.write(comparisons)

    elif option == "Candlestick Chart":
        st.header("Candlestick Chart")
        stock = st.sidebar.selectbox("Select stock", [s.split('.')[0] for s in filenames])
        plot_candlestick(dfs, stock)

    
    elif option == "RSI":
        st.header("RSI (Relative Strength Index)")
        stock = st.sidebar.selectbox("Select stock", [s.split('.')[0] for s in filenames])
        plot_rsi(dfs, stock)


    elif option == "Bollinger Bands":
        st.header("Bollinger Bands")
        stock = st.sidebar.selectbox("Select stock", [s.split('.')[0] for s in filenames])
        plot_bollinger_bands(dfs, stock)

    
    elif option == "Portfolio Management":
        st.title("Portfolio Management")

        # Dropdown to select the dataset
        selected_dataset = st.selectbox("Select a dataset", list(dfs.keys()))

        if selected_dataset:
            df = dfs[selected_dataset]

            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df['Returns'] = df['Adj Close'].pct_change()

            st.write(f"## {selected_dataset} Performance")
            st.line_chart(df['Adj Close'])

            st.write("### Indicators")

            st.write(f"**Sharpe Ratio**: {sharpe_ratio(df['Returns']):.2f}")
            st.write(f"**Sortino Ratio**: {sortino_ratio(df['Returns']):.2f}")
      

            # RSI Calculation
            delta = df['Adj Close'].diff()
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            st.write(f"**Relative Strength Index (RSI)**: {rsi.iloc[-1]:.2f}")




    

if __name__ == "__main__":
    main()
