import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import data_manager as dm
import market_data as md
import polars as pl
import numpy as np
import pandas as pd
from datetime import datetime, date

# Initialize DB
dm.init_db()

st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("🚀 Fast Stock Dashboard")
st.markdown("Built with **DuckDB**, **Polars**, and **Streamlit**.")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Dashboard", "🌍 Market Explorer", "💼 My Stocks"])

# --- TAB 1: DASHBOARD ---
with tab1:
    # Sidebar for controls (only for Tab 1 mostly, but sidebar is global)
    st.sidebar.header("Configuration")

    # Selection Mode
    selection_mode = st.sidebar.radio("Select Stock Source:", ["Search", "My Portfolio"], horizontal=True)

    ticker_input = None
    
    if selection_mode == "My Portfolio":
        portfolio_tickers = dm.get_portfolio_tickers()
        if portfolio_tickers:
            ticker_input = st.sidebar.selectbox("Select from My Stocks:", portfolio_tickers)
        else:
            st.sidebar.warning("Your portfolio is empty.")
            st.sidebar.info("Switch to 'Search' to find and save stocks.")
    else:
        # Stock Search
        default_ticker = "AAPL"
        ticker_input = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", value=default_ticker)

    if ticker_input:
        ticker = ticker_input.upper().strip()
        
        with st.spinner(f"Retrieving data for {ticker}..."):
            # Fetch data
            df = dm.get_stock_data(ticker)
        
        if df is not None and not df.is_empty():
            # Data found
            
            # --- SAVE TO PORTFOLIO BUTTON ---
            col_header1, col_header2 = st.columns([0.8, 0.2])
            with col_header1:
                st.subheader(f"Analysis for {ticker}")
            with col_header2:
                is_saved = dm.is_in_portfolio(ticker)
                if is_saved:
                    if st.button("★ Remove from My Stocks", key="remove_main"):
                        dm.remove_from_portfolio(ticker)
                        st.rerun()
                else:
                    if st.button("☆ Save to My Stocks", key="save_main"):
                        dm.add_to_portfolio(ticker)
                        st.rerun()

            # Convert Date column to consistent format if needed (Polars handles this well usually)
            min_date = df['Date'].min()
            max_date = df['Date'].max()
            
            # Date Slider
            st.sidebar.subheader("Time Line")
            
            # Streamlit slider returns a tuple (start, end)
            date_range = st.sidebar.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )
            
            start_date, end_date = date_range
            
            # Filter data based on selection for the view
            filtered_df = df.filter(
                (pl.col("Date") >= start_date) & 
                (pl.col("Date") <= end_date)
            )
            
            if filtered_df.height > 1: # Need at least 2 points for a line
                # Calculate Average for the LIFE of the stock (Global Average)
                life_avg = df['Close'].mean()
                
                # Calculate Average for the SELECTED range
                selected_avg = filtered_df['Close'].mean()
                
                # Display Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    current_price = df['Close'].tail(1).item()
                    st.metric("Latest Close Price", f"${current_price:.2f}")
                with col2:
                    st.metric("Lifetime Average", f"${life_avg:.2f}")
                with col3:
                    st.metric("Selected Range Average", f"${selected_avg:.2f}")
                    
                # Create Chart
                plot_data = filtered_df.to_pandas()
                
                # Base Line Chart
                fig = px.line(
                    plot_data, 
                    x="Date", 
                    y="Close", 
                    title=f"{ticker} Price Timeline",
                    labels={"Close": "Price (USD)"}
                )
                
                # --- Best Fit Line Calculation ---
                # Convert dates to ordinal for regression
                # We need numeric x values
                x_numeric = pd.to_datetime(plot_data['Date']).map(datetime.toordinal)
                y_values = plot_data['Close']
                
                # Linear Regression: y = mx + c
                slope, intercept = np.polyfit(x_numeric, y_values, 1)
                
                # Calculate best fit line values
                best_fit_y = slope * x_numeric + intercept
                
                # Add Best Fit Trace
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['Date'],
                        y=best_fit_y,
                        mode='lines',
                        name='Trend Line (Best Fit)',
                        line=dict(color='orange', width=2, dash='dashdot')
                    )
                )
                
                # Add Average Line (Lifetime)
                fig.add_hline(
                    y=life_avg, 
                    line_dash="dash", 
                    line_color="red", 
                    annotation_text=f"Lifetime Avg: ${life_avg:.2f}",
                    annotation_position="bottom right"
                )

                # Add Average Line (Selected Range)
                fig.add_hline(
                    y=selected_avg, 
                    line_dash="dot", 
                    line_color="blue", 
                    annotation_text=f"Selected Avg: ${selected_avg:.2f}",
                    annotation_position="top right"
                )
                
                # Update layout
                fig.update_layout(hovermode="x unified")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats section
                with st.expander("See Summary Statistics"):
                    st.write(filtered_df.describe().to_pandas())
            else:
                st.warning("Not enough data points in the selected range to display a chart.")
                
        else:
            st.error(f"Could not find data for ticker: {ticker}. Please check the symbol and try again.")
            st.info("Note: This app uses yfinance. Some niche tickers might not be available or require a suffix (e.g. .L for London).")

    else:
        st.info("Please enter a stock ticker to begin.")

# --- TAB 2: MARKET EXPLORER ---
@st.cache_data(ttl=3600, show_spinner=False)
def get_market_data_cached_v3(tickers, avg_years=2, change_days=1):
    return dm.get_market_summary_data(tickers, avg_years=avg_years, change_days=change_days)

with tab2:
    st.header("🌍 Global Markets & Top Stocks")
    st.markdown("Explore major markets, sectors, and their leading companies.")
    
    # Search or Select Mode
    col_mode, col_m_dummy = st.columns([2, 2])
    with col_mode:
        explore_mode = st.radio("Explore By:", ["Market Index", "Sector Search"], horizontal=True)

    col_m1, col_m2 = st.columns([3, 1])
    
    selected_stocks = []
    selection_title = ""
    
    if explore_mode == "Market Index":
        with col_m1:
            markets = md.get_market_names()
            selected_market = st.selectbox("Select a Market", markets)
            if selected_market:
                selected_stocks = md.get_stocks_for_market(selected_market)
                selection_title = f"Top Stocks in {selected_market}"
    else:
        with col_m1:
            sector_query = st.text_input("Search Sector (e.g., 'car', 'tech', 'bank')", placeholder="Type a sector or industry...")
            if sector_query:
                selected_stocks = md.search_sector(sector_query)
                if selected_stocks:
                    selection_title = f"Results for '{sector_query}'"
                else:
                    st.warning("No matching sector found. Try 'tech', 'auto', 'finance', 'energy', etc.")
            else:
                st.info("Enter a keyword above to find top stocks in that sector.")

    with col_m2:
        market_years = st.number_input("Avg Years", min_value=1, max_value=20, value=2, key="market_years")
        
        # Performance Period Selector
        perf_options = {
            "1 Day": 1,
            "1 Week": 7,
            "1 Month": 30,
            "3 Months": 90,
            "6 Months": 180,
            "1 Year": 365,
            "2 Years": 730,
            "3 Years": 1095,
            "5 Years": 1825
        }
        selected_perf_label = st.selectbox("Perf. Period", list(perf_options.keys()), index=0)
        selected_perf_days = perf_options[selected_perf_label]
        
        filter_option = st.radio("Show:", ["All", "Top Gainers", "Top Losers"], index=0, key="market_filter")
    
    if selected_stocks:
        st.subheader(selection_title)
        
        with st.spinner(f"Fetching data (Avg: {market_years}y, Change: {selected_perf_label})..."):
            # Get tickers
            ticker_list = [s['ticker'] for s in selected_stocks]
            
            # Fetch summary data
            summary_df = get_market_data_cached_v3(ticker_list, avg_years=market_years, change_days=selected_perf_days)
            
            # Create base DataFrame
            base_df = pd.DataFrame(selected_stocks)
            
            # Merge
            if not summary_df.empty:
                # Merge on ticker
                # Note: summary_df has "Ticker" column, base_df has "ticker"
                base_df['Ticker_Upper'] = base_df['ticker'].str.upper()
                # Ensure summary_df Ticker is upper case (it should be from yfinance/our code)
                summary_df['Ticker'] = summary_df['Ticker'].astype(str).str.upper()
                
                merged_df = pd.merge(base_df, summary_df, left_on='Ticker_Upper', right_on='Ticker', how='left')
                
                # Dynamic column name for avg
                avg_col = f"{market_years}-Year Avg"
                
                # Determine Change Column Name
                # It should match what data_manager returns
                # We can dynamically find it or reconstruct it
                # data_manager logic:
                change_col = "Change %"
                if selected_perf_days > 1:
                    if selected_perf_days % 365 == 0:
                        change_col = f"{selected_perf_days // 365}-Year Change %"
                    elif selected_perf_days % 30 == 0:
                         change_col = f"{selected_perf_days // 30}-Month Change %"
                    else:
                         change_col = f"{selected_perf_days}-Day Change %"

                # Check if "Change %" exists (it should with updated code)
                cols_to_keep = ['ticker', 'name', 'Current Price', avg_col]
                final_cols = ['Ticker', 'Name', 'Current Price', avg_col]
                
                if change_col in merged_df.columns:
                    cols_to_keep.append(change_col)
                    final_cols.append(change_col)
                
                # Clean up and Format
                final_df = merged_df[cols_to_keep]
                final_df.columns = final_cols
                
                # Apply Filters/Sorting
                if filter_option == "Top Gainers" and change_col in final_df.columns:
                    final_df = final_df.sort_values(by=change_col, ascending=False).head(10)
                elif filter_option == "Top Losers" and change_col in final_df.columns:
                    final_df = final_df.sort_values(by=change_col, ascending=True).head(10)
                
                # Format formatting for display
                # We can use st.dataframe column config for better visuals
                col_config = {
                    "Current Price": st.column_config.NumberColumn(
                        "Current Price",
                        format="$%.2f"
                    ),
                    avg_col: st.column_config.NumberColumn(
                        avg_col,
                        format="$%.2f"
                    )
                }
                
                if change_col in final_df.columns:
                    col_config[change_col] = st.column_config.NumberColumn(
                        change_col,
                        format="%.2f%%"
                    )

                st.dataframe(
                    final_df, 
                    use_container_width=True,
                    column_config=col_config
                )
            else:
                st.dataframe(base_df, use_container_width=True)
        
        st.markdown("### Quick Analysis")
        # Let user select a stock from this list to view quickly
        selected_ticker = st.selectbox("Select a stock to analyze:", [s['ticker'] for s in selected_stocks])
        
        if st.button(f"Analyze {selected_ticker}"):
            # We can just update the session state or similar, 
            # but simplest way is to fetch and show a mini chart here
            with st.spinner(f"Fetching {selected_ticker}..."):
                mini_df = dm.get_stock_data(selected_ticker)
                
            if mini_df is not None:
                 mini_fig = px.line(
                    mini_df.to_pandas(), 
                    x="Date", 
                    y="Close", 
                    title=f"{selected_ticker} - {next((s['name'] for s in selected_stocks if s['ticker'] == selected_ticker), '')}",
                    height=400
                )
                 st.plotly_chart(mini_fig, use_container_width=True)

# --- TAB 3: MY STOCKS ---
with tab3:
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        st.header("💼 My Stocks Portfolio")
    with col_p2:
        portfolio_years = st.number_input("Avg Years", min_value=1, max_value=20, value=2, key="portfolio_years")
    
    portfolio_tickers = dm.get_portfolio_tickers()
    
    if not portfolio_tickers:
        st.info("You haven't saved any stocks yet. Go to the **Dashboard** tab, search for a stock, and click 'Save to My Stocks'.")
    else:
        st.markdown(f"Tracking **{len(portfolio_tickers)}** stocks.")
        
        # Create a summary table
        portfolio_data = []
        
        # Progress bar if many stocks
        progress_text = "Updating portfolio data..."
        my_bar = st.progress(0, text=progress_text)
        
        start_date_limit = date.today() - pd.Timedelta(days=portfolio_years*365)
        
        for i, ticker in enumerate(portfolio_tickers):
            # Fetch latest data (should be fast from DB, or fetch if new)
            df = dm.get_stock_data(ticker)
            
            if df is not None and not df.is_empty():
                latest = df.tail(1)
                prev = df.tail(2).head(1)
                
                latest_close = latest['Close'].item()
                prev_close = prev['Close'].item() if df.height > 1 else latest_close
                
                change = latest_close - prev_close
                pct_change = (change / prev_close) * 100 if prev_close != 0 else 0
                
                # Calculate Custom Range Avg
                # Filter data for the last X years
                # Ensure Date column is date type
                # df['Date'] is likely date objects. 
                # Polars filtering:
                recent_df = df.filter(pl.col("Date") >= start_date_limit)
                
                if not recent_df.is_empty():
                    period_avg = recent_df['Close'].mean()
                else:
                    period_avg = 0.0
                
                avg_col_name = f"{portfolio_years}-Year Avg"
                
                portfolio_data.append({
                    "Ticker": ticker,
                    "Latest Price": latest_close, # Keep numeric for formatting
                    "Change ($)": change,
                    "Change (%)": pct_change,
                    avg_col_name: period_avg,
                    "Last Updated": latest['Date'].item()
                })
            
            my_bar.progress((i + 1) / len(portfolio_tickers), text=progress_text)
            
        my_bar.empty()
        
        if portfolio_data:
            # Display Summary Table with formatting
            p_df = pd.DataFrame(portfolio_data)
            
            # Dynamic column name
            avg_col_name = f"{portfolio_years}-Year Avg"
            
            st.dataframe(
                p_df,
                use_container_width=True,
                column_config={
                    "Latest Price": st.column_config.NumberColumn(
                        "Latest Price",
                        format="$%.2f"
                    ),
                    "Change ($)": st.column_config.NumberColumn(
                        "Change ($)",
                        format="$%.2f"
                    ),
                    "Change (%)": st.column_config.NumberColumn(
                        "Change (%)",
                        format="%.2f%%"
                    ),
                    avg_col_name: st.column_config.NumberColumn(
                        avg_col_name,
                        format="$%.2f"
                    ),
                    "Last Updated": st.column_config.DateColumn(
                        "Last Updated",
                        format="YYYY-MM-DD"
                    )
                }
            )
            
            st.divider()
            
            # Individual Management
            st.subheader("Manage Portfolio")
            
            col1, col2 = st.columns(2)
            
            with col1:
                stock_to_remove = st.selectbox("Select stock to remove:", portfolio_tickers)
                if st.button("Remove Selected Stock"):
                    dm.remove_from_portfolio(stock_to_remove)
                    st.success(f"Removed {stock_to_remove}")
                    st.rerun()
            
            with col2:
                # Comparison Chart
                st.subheader(f"Price Comparison (Last {portfolio_years} Years)")
                
                # Collect data for chart
                combined_data = []
                for ticker in portfolio_tickers:
                     df = dm.get_stock_data(ticker)
                     if df is not None:
                         # Filter by date range
                         recent = df.filter(pl.col("Date") >= start_date_limit).to_pandas()
                         if not recent.empty:
                             recent['Ticker'] = ticker
                             combined_data.append(recent)
                
                if combined_data:
                    all_stocks = pd.concat(combined_data)
                    fig_comp = px.line(
                        all_stocks, 
                        x="Date", 
                        y="Close", 
                        color="Ticker", 
                        title=f"Performance Comparison",
                        markers=False # Too many markers for multi-year
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
