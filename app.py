import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import numpy as np
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="QuantScan | NSE Scanner",
    page_icon="⚡",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #464B5C;
        margin-bottom: 10px;
    }
    .success-text { color: #00FF99; font-weight: bold; }
    .danger-text { color: #FF4B4B; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_fno_stocks():
    """Fetches official FnO list from NSE Archives"""
    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        # Clean columns
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Find symbol column
        symbol_col = next((c for c in df.columns if 'SYM' in c), 'SYMBOL')
        tickers = df[symbol_col].dropna().astype(str).str.strip().unique().tolist()
        
        # Exclude indices
        exclude = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SYMBOL']
        tickers = [f"{t}.NS" for t in tickers if t not in exclude]
        
        return tickers
    except Exception as e:
        return ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS']

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_scanner(tickers):
    # Progress Bar UI
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Initializing Market Data...")
    
    # Batch download (max 3 months data)
    try:
        data = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
    except Exception:
        return [], []
    
    bullish_signals = []
    bearish_signals = []
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        # Update UI every 5 stocks
        if i % 5 == 0:
            progress = min(i / total, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Scanning {ticker}...")

        try:
            # Handle Data Frame layers
            if len(tickers) > 1:
                if ticker not in data.columns.levels[0]: continue
                df = data[ticker].copy()
            else:
                df = data.copy()

            # Clean data
            df.dropna(subset=['Close', 'High', 'Low'], inplace=True)
            if len(df) < 30: continue

            # Calc RSI
            df['RSI'] = calculate_rsi(df['Close'])
            if len(df) < 15: continue

            # Get Data Points
            curr_close = df['Close'].iloc[-1]
            curr_rsi = df['RSI'].iloc[-1]
            past_high = df['High'].iloc[-15] # 14 days ago
            past_low = df['Low'].iloc[-15]   # 14 days ago
            past_rsi = df['RSI'].iloc[-15]

            if np.isnan(curr_rsi) or np.isnan(past_rsi): continue

            clean_name = ticker.replace('.NS', '')

            # BEARISH LOGIC
            if (curr_close >= past_high) and (curr_rsi < past_rsi):
                bearish_signals.append({
                    'Stock': clean_name,
                    'Price': round(curr_close, 2),
                    'RSI': round(curr_rsi, 2),
                    'Divergence': 'Bearish'
                })
            
            # BULLISH LOGIC
            elif (curr_close < past_low) and (curr_rsi > past_rsi):
                bullish_signals.append({
                    'Stock': clean_name,
                    'Price': round(curr_close, 2),
                    'RSI': round(curr_rsi, 2),
                    'Divergence': 'Bullish'
                })

        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()
    return bullish_signals, bearish_signals

# --- MAIN LAYOUT ---

st.title("⚡ QuantScan")
st.markdown("### Automated NSE F&O Divergence Scanner")
st.markdown("This tool scans all liquid NSE stocks for **RSI Divergence** (Price vs RSI) over a **14-day lookback**.")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🚀 START SCAN", type="primary", use_container_width=True):
        with st.spinner("Fetching Stock List..."):
            stock_list = get_fno_stocks()
        
        st.toast(f"Found {len(stock_list)} stocks. Processing...", icon="⏳")
        bull, bear = run_scanner(stock_list)
        
        st.success("Scan Complete!")
        
        # Display Results
        st.markdown("---")
        
        # Bullish Column
        st.subheader(f"🟢 Bullish Candidates ({len(bull)})")
        if bull:
            st.dataframe(pd.DataFrame(bull), use_container_width=True)
        else:
            st.info("No Bullish Divergence found today.")

        # Bearish Column
        st.subheader(f"🔴 Bearish Candidates ({len(bear)})")
        if bear:
            st.dataframe(pd.DataFrame(bear), use_container_width=True)
        else:
            st.info("No Bearish Divergence found today.")

with col2:
    st.markdown("") # Spacer

with col3:
    st.markdown("") # Spacer

# Footer
st.markdown("---")
st.caption("Data provided by Yahoo Finance. This is for educational purposes only.")