import streamlit as st
import data_manager as dm
import market_data as md
import polars as pl
import numpy as np
import pandas as pd
import altair as alt
import yfinance as yf
from datetime import datetime, date, timedelta

# Initialize DB
dm.init_db()

alt.data_transformers.disable_max_rows()


def _format_market_cap(value):
    if value is None:
        return "-"
    try:
        value = float(value)
    except Exception:
        return "-"
    if value >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value:,.0f}"


def _format_percent(value, digits=2):
    if value is None:
        return "-"
    try:
        v = float(value)
        if v < 0:
            return "-"
        if v > 1 and v <= 100:
            v = v / 100.0
        if v > 1:
            return "-"
        return f"{v * 100:.{digits}f}%"
    except Exception:
        return "-"


@st.cache_data(ttl=3600, show_spinner=False)
def get_ticker_overview_cached(ticker):
    ticker = str(ticker).upper().strip()
    try:
        t = yf.Ticker(ticker)
    except Exception:
        return {}

    info = {}
    try:
        info = t.info or {}
    except Exception:
        info = {}

    dividends = None
    try:
        dividends = t.dividends
    except Exception:
        dividends = None

    ttm_div = None
    ttm_div_count = None
    pays_dividends = False
    last_dividend_date = None
    last_dividend_amount = None
    if dividends is not None and hasattr(dividends, "empty") and not dividends.empty:
        try:
            last_dividend_amount = float(dividends.tail(1).iloc[0])
            last_dividend_date = pd.to_datetime(dividends.index[-1]).date()

            one_year_ago = pd.Timestamp.today() - pd.Timedelta(days=365)
            dividends_index = pd.to_datetime(dividends.index)
            div_last_year = dividends[dividends_index >= one_year_ago]
            if not div_last_year.empty:
                ttm_div = float(div_last_year.sum())
                ttm_div_count = int(div_last_year.shape[0])
                pays_dividends = ttm_div > 0
            else:
                last_div = float(dividends.tail(1).iloc[0])
                pays_dividends = last_div > 0
        except Exception:
            pass
    else:
        div_rate = info.get("dividendRate")
        if div_rate is not None:
            try:
                div_rate = float(div_rate)
                if div_rate > 0:
                    ttm_div = div_rate
                    pays_dividends = True
            except Exception:
                pass

    if not pays_dividends:
        div_yield = info.get("dividendYield")
        if div_yield is not None:
            try:
                div_yield = float(div_yield)
                if div_yield > 0:
                    pays_dividends = True
            except Exception:
                pass

    ex_div_date = None
    raw_ex_div = info.get("exDividendDate")
    if raw_ex_div:
        try:
            ex_div_date = pd.to_datetime(raw_ex_div, unit="s").date()
        except Exception:
            try:
                ex_div_date = pd.to_datetime(raw_ex_div).date()
            except Exception:
                ex_div_date = None

    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "currency": info.get("currency") or "USD",
        "market_cap": info.get("marketCap"),
        "beta": info.get("beta"),
        "pe_ttm": info.get("trailingPE"),
        "pe_fwd": info.get("forwardPE"),
        "dividend_yield": info.get("dividendYield"),
        "dividend_rate": info.get("dividendRate"),
        "payout_ratio": info.get("payoutRatio"),
        "ex_dividend_date": ex_div_date,
        "pays_dividends": pays_dividends,
        "dividend_ttm": ttm_div,
        "dividend_payments_ttm": ttm_div_count,
        "last_dividend_date": last_dividend_date,
        "last_dividend_amount": last_dividend_amount,
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def get_dividends_history_cached(ticker):
    ticker = str(ticker).upper().strip()
    try:
        t = yf.Ticker(ticker)
    except Exception:
        return pd.DataFrame(columns=["Date", "Dividend"])

    try:
        dividends = t.dividends
    except Exception:
        dividends = None

    if dividends is None or (hasattr(dividends, "empty") and dividends.empty):
        return pd.DataFrame(columns=["Date", "Dividend"])

    div_df = dividends.reset_index()
    if div_df.shape[1] >= 2:
        div_df.columns = ["Date", "Dividend"]
    else:
        return pd.DataFrame(columns=["Date", "Dividend"])

    div_df["Date"] = pd.to_datetime(div_df["Date"], utc=True).dt.tz_convert(None)
    div_df["Dividend"] = pd.to_numeric(div_df["Dividend"], errors="coerce")
    div_df = div_df.dropna(subset=["Date", "Dividend"])
    return div_df

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
            overview = get_ticker_overview_cached(ticker)

            if df is not None and not df.is_empty():
                try:
                    max_date_check = df["Date"].max()
                    if max_date_check is not None and (date.today() - max_date_check).days >= 2:
                        refreshed = dm.fetch_and_store_stock(ticker)
                        if refreshed is not None and not refreshed.is_empty():
                            df = refreshed
                except Exception:
                    pass
        
        if df is not None and not df.is_empty():
            # Data found
            
            # --- SAVE TO PORTFOLIO BUTTON ---
            col_header1, col_header2 = st.columns([0.8, 0.2])
            with col_header1:
                company_name = (overview or {}).get("name")
                if company_name:
                    st.subheader(f"{ticker} — {company_name}")
                else:
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
            
            slider_slot = st.sidebar.empty()
            quick_slot = st.sidebar.empty()

            quick_options = ["1 Month", "6 Months", "1 Year", "2 Years", "5 Years", "10 Years", "Lifetime"]
            if st.session_state.get("timeline_ticker") != ticker:
                st.session_state["timeline_ticker"] = ticker
                st.session_state["timeline_quick"] = "1 Year"
                st.session_state["timeline_quick_applied"] = None
                st.session_state["date_range_slider"] = (max(min_date, max_date - timedelta(days=365)), max_date)

            selected_quick = quick_slot.selectbox("Quick Range", quick_options, key="timeline_quick")

            quick_days = {
                "1 Month": 30,
                "6 Months": 182,
                "1 Year": 365,
                "2 Years": 730,
                "5 Years": 1825,
                "10 Years": 3650,
                "Lifetime": None,
            }

            if "date_range_slider" not in st.session_state:
                days = quick_days.get(selected_quick, 365)
                if days is None:
                    start_default = min_date
                else:
                    start_default = max(min_date, max_date - timedelta(days=days))
                st.session_state["date_range_slider"] = (start_default, max_date)

            if st.session_state.get("timeline_quick_applied") != selected_quick:
                days = quick_days.get(selected_quick)
                if days is None:
                    start_default = min_date
                else:
                    start_default = max(min_date, max_date - timedelta(days=days))
                st.session_state["date_range_slider"] = (start_default, max_date)
                st.session_state["timeline_quick_applied"] = selected_quick

            date_range = slider_slot.slider(
                "Select Date Range",
                min_value=min_date,
                max_value=max_date,
                key="date_range_slider",
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

                if overview:
                    currency = overview.get("currency") or "USD"
                    col4, col5, col6, col7 = st.columns(4)
                    with col4:
                        st.metric("Dividends", "Yes" if overview.get("pays_dividends") else "No")
                    with col5:
                        st.metric("Dividend Yield", _format_percent(overview.get("dividend_yield")))
                    with col6:
                        div_ttm = overview.get("dividend_ttm")
                        div_ttm_display = f"{div_ttm:.2f} {currency}" if isinstance(div_ttm, (int, float)) else "-"
                        st.metric("Dividend (TTM)", div_ttm_display)
                    with col7:
                        last_div_amt = overview.get("last_dividend_amount")
                        last_div_display = f"{last_div_amt:.2f} {currency}" if isinstance(last_div_amt, (int, float)) else "-"
                        st.metric("Last Dividend", last_div_display)

                    last_div_date = overview.get("last_dividend_date")
                    st.metric("Last Dividend Date", last_div_date.isoformat() if last_div_date else "-")

                    col8, col9, col10, col11 = st.columns(4)
                    with col8:
                        st.metric("Market Cap", _format_market_cap(overview.get("market_cap")))
                    with col9:
                        pe_ttm = overview.get("pe_ttm")
                        st.metric("P/E (TTM)", f"{pe_ttm:.2f}" if isinstance(pe_ttm, (int, float)) else "-")
                    with col10:
                        low_52w = overview.get("fifty_two_week_low")
                        high_52w = overview.get("fifty_two_week_high")
                        if isinstance(low_52w, (int, float)) and isinstance(high_52w, (int, float)):
                            st.metric("52W Range", f"${low_52w:.2f} – ${high_52w:.2f}")
                        else:
                            st.metric("52W Range", "-")
                    with col11:
                        beta = overview.get("beta")
                        st.metric("Beta", f"{beta:.2f}" if isinstance(beta, (int, float)) else "-")
                    
                # Create Chart
                plot_data = filtered_df.to_pandas()
                plot_data["Date"] = pd.to_datetime(plot_data["Date"])
                
                x_numeric = plot_data["Date"].map(datetime.toordinal)
                y_values = plot_data["Close"]
                slope, intercept = np.polyfit(x_numeric, y_values, 1)
                trend_df = plot_data[["Date"]].copy()
                trend_df["Trend"] = slope * x_numeric + intercept

                base = alt.Chart(plot_data).encode(x=alt.X("Date:T", title=None))
                price_line = base.mark_line().encode(
                    y=alt.Y("Close:Q", title="Price"),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date"),
                        alt.Tooltip("Close:Q", title="Close", format="$.2f"),
                        alt.Tooltip("Volume:Q", title="Volume", format=",")
                    ],
                )

                trend_line = alt.Chart(trend_df).mark_line(color="orange", strokeDash=[6, 3]).encode(
                    x=alt.X("Date:T"),
                    y=alt.Y("Trend:Q"),
                )

                life_rule = alt.Chart(pd.DataFrame({"y": [life_avg]})).mark_rule(color="red", strokeDash=[6, 6]).encode(
                    y=alt.Y("y:Q"),
                )

                selected_rule = alt.Chart(pd.DataFrame({"y": [selected_avg]})).mark_rule(color="blue", strokeDash=[2, 2]).encode(
                    y=alt.Y("y:Q"),
                )

                dividends_df = get_dividends_history_cached(ticker)
                div_in_range = None
                if not dividends_df.empty:
                    try:
                        dividends_df = dividends_df.copy()
                        dividends_df["Date"] = pd.to_datetime(dividends_df["Date"], utc=True).dt.tz_convert(None)
                    except Exception:
                        pass
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    div_filtered = dividends_df[(dividends_df["Date"] >= start_dt) & (dividends_df["Date"] <= end_dt)]
                    if not div_filtered.empty:
                        div_in_range = div_filtered

                nearest = alt.selection_point(nearest=True, on="mouseover", fields=["Date"], empty=False)
                hover_base = base.encode(y=alt.Y("Close:Q"))
                selectors = hover_base.mark_point(opacity=0).add_params(nearest)
                points = hover_base.mark_point(size=55).encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
                rule = base.mark_rule(color="gray").encode(opacity=alt.condition(nearest, alt.value(0.3), alt.value(0)))
                text = hover_base.mark_text(align="left", dx=6, dy=-6).encode(
                    text=alt.condition(nearest, alt.Text("Close:Q", format="$.2f"), alt.value("")),
                )

                layers = [
                    price_line,
                    trend_line,
                    life_rule,
                    selected_rule,
                ]
                layers.extend([selectors, points, rule, text])

                price_chart = alt.layer(*layers).properties(height=420, title=f"{ticker} Price Timeline")

                if div_in_range is not None and not div_in_range.empty:
                    dividend_chart = (
                        alt.Chart(div_in_range)
                        .mark_bar(color="green")
                        .encode(
                            x=alt.X("Date:T", title=None),
                            y=alt.Y("Dividend:Q", title="Dividend"),
                            tooltip=[
                                alt.Tooltip("Date:T", title="Dividend Date"),
                                alt.Tooltip("Dividend:Q", title="Dividend", format="$.4f"),
                            ],
                        )
                        .properties(height=140)
                    )
                    chart = alt.vconcat(price_chart, dividend_chart).resolve_scale(x="shared")
                else:
                    chart = price_chart

                st.altair_chart(chart, use_container_width=True)
                
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
                 mini_pd = mini_df.to_pandas()
                 mini_pd["Date"] = pd.to_datetime(mini_pd["Date"])
                 mini_title = f"{selected_ticker} - {next((s['name'] for s in selected_stocks if s['ticker'] == selected_ticker), '')}"
                 mini_chart = (
                    alt.Chart(mini_pd)
                    .mark_line()
                    .encode(
                        x=alt.X("Date:T", title=None),
                        y=alt.Y("Close:Q", title="Price"),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Close:Q", title="Close", format="$.2f"),
                        ],
                    )
                    .properties(height=400, title=mini_title)
                 )
                 st.altair_chart(mini_chart, use_container_width=True)

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
                    all_stocks["Date"] = pd.to_datetime(all_stocks["Date"])
                    fig_comp = (
                        alt.Chart(all_stocks)
                        .mark_line()
                        .encode(
                            x=alt.X("Date:T", title=None),
                            y=alt.Y("Close:Q", title="Price"),
                            color=alt.Color("Ticker:N"),
                            tooltip=[
                                alt.Tooltip("Ticker:N", title="Ticker"),
                                alt.Tooltip("Date:T", title="Date"),
                                alt.Tooltip("Close:Q", title="Close", format="$.2f"),
                            ],
                        )
                        .properties(title="Performance Comparison")
                    )
                    st.altair_chart(fig_comp, use_container_width=True)
