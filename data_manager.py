import duckdb
import polars as pl
import yfinance as yf
import pandas as pd
import os
from datetime import date, timedelta

DB_PATH = "stocks.duckdb"

def get_connection():
    return duckdb.connect(DB_PATH)

def init_db():
    conn = get_connection()
    # Create a table to store stock data if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            Date DATE,
            Open DOUBLE,
            High DOUBLE,
            Low DOUBLE,
            Close DOUBLE,
            Volume BIGINT,
            Ticker VARCHAR
        )
    """)
    
    # Create user portfolio table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_portfolio (
            Ticker VARCHAR PRIMARY KEY,
            DateAdded DATE
        )
    """)
    conn.close()

def fetch_and_store_stock(ticker):
    """
    Fetches stock data from yfinance and stores it in DuckDB.
    Returns the Polars DataFrame.
    """
    print(f"Fetching data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # Fetch max history
        hist = stock.history(period="max")
        
        if hist.empty:
            return None
        
        # Reset index to get Date as a column
        hist = hist.reset_index()
        
        # Ensure Date is just date (no timezone if possible, or handle it)
        hist['Date'] = pd.to_datetime(hist['Date']).dt.date
        
        # Add Ticker column
        hist['Ticker'] = ticker.upper()
        
        # Select only relevant columns
        hist = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Ticker']]
        
        # Convert to Polars for easy handling
        pl_df = pl.from_pandas(hist)
        
        conn = get_connection()
        
        # Delete existing data for this ticker to avoid duplicates
        conn.execute("DELETE FROM stock_prices WHERE Ticker = ?", [ticker.upper()])
        
        # Insert data
        conn.register('temp_df', hist)
        conn.execute("INSERT INTO stock_prices SELECT * FROM temp_df")
        conn.close()
        
        return pl_df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def get_stock_data(ticker):
    """
    Retrieves stock data from DuckDB. If not present, fetches it first.
    Returns a Polars DataFrame.
    """
    ticker = ticker.upper()
    conn = get_connection()
    
    # Check if ticker exists
    try:
        result = conn.execute("SELECT count(*) FROM stock_prices WHERE Ticker = ?", [ticker]).fetchone()
        count = result[0]
    except duckdb.CatalogException:
        init_db()
        count = 0
        
    conn.close()
    
    if count == 0:
        return fetch_and_store_stock(ticker)
    
    conn = get_connection()
    query = f"SELECT * FROM stock_prices WHERE Ticker = '{ticker}' ORDER BY Date"
    df = conn.execute(query).pl()
    conn.close()
    
    return df

def get_unique_tickers():
    conn = get_connection()
    try:
        df = conn.execute("SELECT DISTINCT Ticker FROM stock_prices").pl()
        return df['Ticker'].to_list()
    except:
        return []
    finally:
        conn.close()

def get_market_summary_data(tickers, avg_years=2, change_days=1):
    """
    Fetches data and computes:
    - Current Price (latest Close)
    - {avg_years}-Year Average (mean of Close)
    - Change % over {change_days} days
    """
    change_col_name = "Change %"
    if change_days > 1:
        # Determine label (e.g. 365 days -> 1 Year)
        if change_days % 365 == 0:
            change_col_name = f"{change_days // 365}-Year Change %"
        elif change_days % 30 == 0:
             change_col_name = f"{change_days // 30}-Month Change %"
        else:
             change_col_name = f"{change_days}-Day Change %"
    
    if not tickers:
        col_name = f"{avg_years}-Year Avg"
        return pd.DataFrame(columns=["Ticker", "Current Price", col_name, change_col_name])
    
    # Calculate how far back we need to go
    # Max of avg_years or change_days/365
    needed_years = max(avg_years, change_days / 365.0)
    # Add a small buffer (e.g. 10 days) to ensure we have the start date
    buffer_days = 10
    
    print(f"Bulk fetching data for {len(tickers)} tickers (approx {needed_years:.1f} years)...")
    try:
        start_date = date.today() - timedelta(days=int(needed_years*365) + buffer_days)
        # group_by='ticker' ensures we have a MultiIndex with Ticker as top level
        data = yf.download(tickers, start=start_date, group_by='ticker', progress=False, threads=True)
        
        results = []
        
        def process_ticker_df(ticker, df):
            if not df.empty and 'Close' in df.columns:
                valid_close = df['Close'].dropna()
                if not valid_close.empty:
                    current_price = valid_close.iloc[-1]
                    
                    # Calculate Average over last X years
                    # Filter for average period
                    avg_start_date = pd.Timestamp(date.today() - timedelta(days=int(avg_years*365)))
                    avg_data = valid_close[valid_close.index >= avg_start_date]
                    if not avg_data.empty:
                        avg_price = avg_data.mean()
                    else:
                        avg_price = valid_close.mean() # Fallback

                    # Calculate Change %
                    pct_change = 0.0
                    if change_days == 1:
                        # Daily change
                        if len(valid_close) > 1:
                            prev_price = valid_close.iloc[-2]
                            if prev_price != 0:
                                pct_change = ((current_price - prev_price) / prev_price) * 100
                    else:
                        # Change over N days
                        # Find price N days ago
                        latest_date = valid_close.index[-1]
                        target_date = latest_date - timedelta(days=change_days)
                        
                        # Use asof to find nearest price on or before target_date
                        # valid_close index is sorted
                        # asof works on Series if index is sorted
                        try:
                            # valid_close.asof(target_date) might return NaN if target_date is before start
                            # If we use searchsorted, we can get the nearest index
                            idx = valid_close.index.searchsorted(target_date)
                            # idx is where target_date would be inserted
                            # If target_date is exact match, idx points to it
                            # If target_date is missing (weekend), idx points to next available day
                            # We want price AROUND that time.
                            # If idx < len, valid_close.index[idx] >= target_date
                            # Let's just grab the price at idx if valid, or idx-1
                            
                            start_price = None
                            if idx < len(valid_close):
                                # Check if close enough?
                                # Let's just take the price at that index (which is >= target_date)
                                # Or maybe we want the price *before*?
                                # Usually "1 Year Return" is (Price_Now - Price_1Y_Ago) / Price_1Y_Ago
                                # If 1Y Ago is Saturday, take Friday (before).
                                # searchsorted returns "next" day (Monday).
                                # So maybe idx-1 is better if exact match not found?
                                # Let's use asof which returns last valid value (Friday)
                                start_price = valid_close.asof(target_date)
                            else:
                                start_price = valid_close.iloc[0] # Too far back?
                                
                            if pd.isna(start_price):
                                # Try first available if target is before data start
                                start_price = valid_close.iloc[0]

                            if start_price and start_price != 0:
                                pct_change = ((current_price - start_price) / start_price) * 100
                        except Exception:
                            pass

                    return {
                        "Ticker": ticker,
                        "Current Price": current_price,
                        f"{avg_years}-Year Avg": avg_price,
                        change_col_name: pct_change
                    }
            return None

        # If only one ticker is passed and found
        if len(tickers) == 1:
            ticker = tickers[0]
            df = data
            if isinstance(df.columns, pd.MultiIndex):
                try:
                    df = df[ticker]
                except KeyError:
                    pass 
            res = process_ticker_df(ticker, df)
            if res:
                results.append(res)
        else:
            for ticker in tickers:
                try:
                    if ticker in data.columns:
                        df = data[ticker]
                        res = process_ticker_df(ticker, df)
                        if res:
                            results.append(res)
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    continue
                    
        return pd.DataFrame(results)
    except Exception as e:
        print(f"Error bulk fetching data: {e}")
        col_name = f"{avg_years}-Year Avg"
        return pd.DataFrame(columns=["Ticker", "Current Price", col_name, change_col_name])

# --- Portfolio Functions ---

def add_to_portfolio(ticker):
    ticker = ticker.upper()
    conn = get_connection()
    try:
        today = date.today()
        # Use INSERT OR IGNORE or similar logic. DuckDB supports ON CONFLICT
        conn.execute("INSERT OR REPLACE INTO user_portfolio (Ticker, DateAdded) VALUES (?, ?)", [ticker, today])
    finally:
        conn.close()

def remove_from_portfolio(ticker):
    ticker = ticker.upper()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM user_portfolio WHERE Ticker = ?", [ticker])
    finally:
        conn.close()

def get_portfolio_tickers():
    conn = get_connection()
    try:
        # Check if table exists first (in case init_db wasn't run recently)
        # But init_db is usually run at startup. 
        # We can just try-except
        df = conn.execute("SELECT Ticker FROM user_portfolio ORDER BY Ticker").pl()
        return df['Ticker'].to_list()
    except duckdb.CatalogException:
        return []
    finally:
        conn.close()

def is_in_portfolio(ticker):
    ticker = ticker.upper()
    conn = get_connection()
    try:
        result = conn.execute("SELECT count(*) FROM user_portfolio WHERE Ticker = ?", [ticker]).fetchone()
        return result[0] > 0
    except duckdb.CatalogException:
        return False
    finally:
        conn.close()
