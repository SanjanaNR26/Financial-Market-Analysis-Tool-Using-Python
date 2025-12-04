import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Helper Functions
# -------------------------

def fetch_data(tickers, start, end):
    """Fetch historical price data for multiple tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    
    # Handle MultiIndex columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        # Extract 'Adj Close' level
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
        elif 'Close' in data.columns.get_level_values(0):
            data = data['Close']
    else:
        # Single ticker returns Series or flat columns
        if isinstance(data, pd.DataFrame) and 'Adj Close' in data.columns:
            data = data['Adj Close']
        elif isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            data = data['Close']
    
    # Ensure DataFrame format
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # Clean up column names - remove MultiIndex level if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)  # Get ticker names
    
    return data.dropna(how='all')

def compute_returns(prices):
    """Daily percentage returns."""
    return prices.pct_change().dropna()

def compute_volatility(returns, window=30):
    """Rolling volatility (stdev) and annualized volatility."""
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    annual_vol = returns.std() * np.sqrt(252)
    return rolling_vol, annual_vol

# -------------------------
# Streamlit UI
# -------------------------

st.set_page_config(page_title="Financial Market Analysis Tool", layout="wide")

st.title("📈 Financial Market Analysis Tool")
st.markdown("A professional Python-based FICC & trading analysis system.")

# -------------------------
# User Input Section
# -------------------------

tickers = st.text_input(
    "Enter asset tickers (comma-separated):",
    value="AAPL, MSFT, SPY, EURUSD=X"
)

start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("today"))

if st.button("Run Analysis"):
    tickers_list = [t.strip() for t in tickers.split(",")]

    # Fetch data
    st.subheader("📥 Fetching Data...")
    prices = fetch_data(tickers_list, start=start_date, end=end_date)

    st.write("### Raw Price Data")
    st.dataframe(prices)

    # Download CSV
    st.download_button(
        "Download Price Data (CSV)",
        data=prices.to_csv().encode('utf-8'),
        file_name="market_data.csv",
    )

    # -------------------------
    # Time-Series Plot
    # -------------------------
    st.subheader("📊 Price Time-Series")
    fig, ax = plt.subplots(figsize=(12, 5))
    prices.plot(ax=ax)
    ax.set_title("Asset Prices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # -------------------------
    # Returns Analysis
    # -------------------------
    returns = compute_returns(prices)
    st.subheader("📉 Daily Returns")
    st.dataframe(returns)

    # Histogram of returns
    st.subheader("📊 Returns Distribution")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    returns.hist(ax=ax2, bins=40)
    ax2.set_title("Return Distribution")
    st.pyplot(fig2)

    # -------------------------
    # Volatility Analysis
    # -------------------------
    st.subheader("⚡ Volatility Analysis (30-day rolling)")

    rolling_vol, annual_vol = compute_volatility(returns)

    st.write("### Annualized Volatility")
    st.write(annual_vol)

    fig3, ax3 = plt.subplots(figsize=(12, 5))
    rolling_vol.plot(ax=ax3)
    ax3.set_title("Rolling Annualized Volatility")
    st.pyplot(fig3)

    # -------------------------
    # Correlation Matrix
    # -------------------------
    st.subheader("🔗 Correlation Matrix")

    corr = returns.corr()
    st.dataframe(corr)

    fig4, ax4 = plt.subplots(figsize=(8, 6))
    cax = ax4.matshow(corr, cmap='coolwarm')
    fig4.colorbar(cax)
    ax4.set_xticks(range(len(corr.columns)))
    ax4.set_yticks(range(len(corr.columns)))
    ax4.set_xticklabels(corr.columns, rotation=45)
    ax4.set_yticklabels(corr.columns)
    st.pyplot(fig4)

st.markdown("---")
st.markdown("Built with ❤️ Using Python, Pandas, Matplotlib & Streamlit")
