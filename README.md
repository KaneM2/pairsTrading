# Equities Pair Trading Tool

## Overview
This tool allows users to test and evaluate various equity pair-trading strategies using historical data for components of S&P 500, Russell 2000, and Nasdaq 100. A video of a demo of the dashboard can be found in the repository at `docs/dashboard_demo.mp4`

---

## Goals
- **Identify highly correlated pairs of stocks.**
- **Backtest pair-trading strategies based on user-defined conditions.**
- **Provide relevant risk and return metrics** (e.g., Sharpe, Sortino, Max Drawdown).
- **Offer optimized parameter suggestions for maximizing performance.**

---

## Features
- **Data Source**: Pulls end-of-day data from Yahoo Finance or a database of your choice.
- **Correlation Metrics**: Provides relevant metrics to identify the strength of correlation, mean reversion speeds, etc.
- **Flexible Time Periods**: Allows the user to select time periods for correlation and backtesting.
- **Methods for Pairs Identification**: Offers multiple ways (Cointegration/Correlation/Distance) to identify stock pairs.
- **Backtesting**: Allows backtesting with various entry/exit conditions, trade durations, and other parameters.
- **Parameter Tuning**: Sliders for real-time sensitivity evaluation.
- **Optimized Parameters**: Suggests optimized parameters based on maximum Sharpe ratio.
- **Performance Metrics**: Shows risk and return metrics in a table.
- **Graphs**: Includes graphs to illustrate correlation strength and strategy performance.

---

## Requirements
- **Python 3.8**



## Usage
1. Clone the repository:
   ``` git clone https://github.com/KaneM2/pairsTrading.git ```
2. Navigate to Folder:
   ``` cd pairsTrading ```
3. Install requirements :
 ``` pip install -r requirements.txt ```
4. Create an empty folder db in data : data/db
5. Initialise mongodb in this folder : ```mongod --dbpath <path-to-project>\pairsTrading\data\db```
6. Run main.py with arguments to use local db and to initialise it by scraping data (Only S&P 500 data is included for speed purposes but Nasdaq 100 and Russell 2000 can be added by adding to the command line argument) :
```
python main.py --db local --db_initialise True --indices "S&P 500"
```


7. Once the db initialisation is run and the data has been scraped , any subsequent runs can be done with :
```
python main.py --db local --db_initialise False 
```

8. A link to open the dashboard should appear in the command line once main.py is run ( and data collection is complete )
   
