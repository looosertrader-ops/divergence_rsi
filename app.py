import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import io
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="QuantScan Pro | Fibonacci Levels",
    page_icon="⚡",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    /* Make the Target column stand out */
    div[data-testid="stTable"] td:nth-child(5) {
        color: #4facfe;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---

@st.cache_data(ttl=3600)
def get_fno_data():
    """
    Fetches the official F&O list from NSE to get Tickers AND Market Lot Sizes.
    """
    url = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Decode CSV
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        df.columns = [c.strip().upper() for c in df.columns]
        
        # Identify Columns
        symbol_col = next((c for c in df.columns if 'SYM' in c), 'SYMBOL')
        lot_col = next((c for c in df.columns if 'LOT' in c), None)
        
        # Filter Indices
        exclude = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY', 'SYMBOL']
        
        stock_data = {}
        for _, row in df.iterrows():
            sym = str(row[symbol_col]).strip()
            if sym not in exclude:
                try:
                    lot = int(str(row[lot_col]).strip()) if lot_col else 0
                except:
                    lot = 0
                stock_data[f"{sym}.NS"] = {'symbol': sym, 'lot': lot}
                
        return stock_data

    except Exception as e:
        # Fallback Data if NSE is down
        return {
            'RELIANCE.NS': {'symbol': 'RELIANCE', 'lot': 250},
            'TCS.NS': {'symbol': 'TCS', 'lot': 175}, 
            'INFY.NS': {'symbol': 'INFY', 'lot': 400},
            'SBIN.NS': {'symbol': 'SBIN', 'lot': 1500},
            'HDFCBANK.NS': {'symbol': 'HDFCBANK', 'lot': 550}
        }

def calculate_rsi(series, period=14):
    """Calculates RSI (Relative Strength Index)"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def run_scanner(stock_data):
    tickers = list(stock_data.keys())
    
    # UI Progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Fetching Market Data (3 Months)...")
    
    # Download Data
    try:
        data = yf.download(tickers, period="3mo", group_by='ticker', threads=True, progress=False)
    except Exception:
        return [], []
    
    bullish_signals = []
    bearish_signals = []
    total = len(tickers)
    
    for i, ticker in enumerate(tickers):
        # Update Progress
        if i % 10 == 0:
            progress = min(i / total, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Analyzing {ticker}...")

        try:
            # Extract Data
            if len(tickers) > 1:
                if ticker not in data.columns.levels[0]: continue
                df = data[ticker].copy()
            else:
                df = data.copy()

            # Data Validation
            df.dropna(subset=['Close', 'High', 'Low'], inplace=True)
            if len(df) < 30: continue

            df['RSI'] = calculate_rsi(df['Close'])
            if len(df) < 15: continue

            # --- KEY DATA POINTS ---
            curr_close = df['Close'].iloc[-1]
            curr_rsi = df['RSI'].iloc[-1]
            
            # 14 Days Ago
            past_high = df['High'].iloc[-15]
            past_low = df['Low'].iloc[-15]
            past_rsi = df['RSI'].iloc[-15]
            
            # 14-Day Swing Range (for Fib Calc)
            swing_high = df['High'].tail(14).max()
            swing_low = df['Low'].tail(14).min()
            swing_diff = swing_high - swing_low

            if np.isnan(curr_rsi) or np.isnan(past_rsi): continue

            # Meta Data
            sym_info = stock_data.get(ticker, {'symbol': ticker, 'lot': 0})
            lot_size = sym_info['lot']
            clean_name = sym_info['symbol']

            # --- BEARISH DIVERGENCE (Sell) ---
            # Logic: Price made a higher high (or matched), but RSI made a lower high
            if (curr_close >= past_high) and (curr_rsi < past_rsi):
                
                # Stop Loss: Recent Swing High + 0.5%
                sl = swing_high * 1.005
                sl_pct = ((sl - curr_close) / curr_close) * 100
                
                # Targets: Fib Retracement DOWN (38.2%, 50%, 61.8%)
                t1 = swing_high - (swing_diff * 0.382)
                t2 = swing_high - (swing_diff * 0.500)
                t3 = swing_high - (swing_diff * 0.618)
                
                targets_fmt = f"{int(t1)} - {int(t2)} - {int(t3)}"
                risk_val = (sl - curr_close) * lot_size

                bearish_signals.append({
                    'Stock': clean_name,
                    'Price': round(curr_close, 1),
                    'SL': round(sl, 1),
                    'SL %': f"{round(sl_pct, 2)}%",
                    'Targets (Fib)': targets_fmt,
                    'Lot Size': lot_size,
                    'Risk/Lot': f"₹{int(risk_val)}"
                })
            
            # --- BULLISH DIVERGENCE (Buy) ---
            # Logic: Price made a lower low (or matched), but RSI made a higher low
            elif (curr_close < past_low) and (curr_rsi > past_rsi):
                
                # Stop Loss: Recent Swing Low - 0.5%
                sl = swing_low * 0.995
                sl_pct = ((curr_close - sl) / curr_close) * 100
                
                # Targets: Fib Retracement UP (38.2%, 50%, 61.8%)
                t1 = swing_low + (swing_diff * 0.382)
                t2 = swing_low + (swing_diff * 0.500)
                t3 = swing_low + (swing_diff * 0.618)

                targets_fmt = f"{int(t1)} - {int(t2)} - {int(t3)}"
                risk_val = (curr_close - sl) * lot_size

                bullish_signals.append({
                    'Stock': clean_name,
                    'Price': round(curr_close, 1),
                    'SL': round(sl, 1),
                    'SL %': f"{round(sl_pct, 2)}%",
                    'Targets (Fib)': targets_fmt,
                    'Lot Size': lot_size,
                    'Risk/Lot': f"₹{int(risk_val)}"
                })

        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()
    return bullish_signals, bearish_signals

# --- MAIN UI ---

st.title("⚡ QuantScan Pro")
st.markdown("### NSE Scanner with Fibonacci Targets")
st.info("Targets calculated using **38.2%, 50%, and 61.8% Fibonacci Retracement** levels of the 14-day swing.")

if st.button("RUN LIVE SCAN", type="primary", use_container_width=True):
    with st.spinner("Fetching Tickers & Lot Sizes..."):
        stock_data = get_fno_data()
    
    st.toast(f"Scanning {len(stock_data)} Stocks...", icon="🚀")
    bull, bear = run_scanner(stock_data)
    
    # --- BULLISH RESULTS ---
    st.subheader(f"🟢 Bullish Setups ({len(bull)})")
    if bull:
        df_bull = pd.DataFrame(bull)
        st.dataframe(
            df_bull,
            column_config={
                "Stock": st.column_config.TextColumn("Stock"),
                "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                "SL": st.column_config.NumberColumn("SL", format="%.2f"),
                "Targets (Fib)": st.column_config.TextColumn("Targets (Fib)", help="Fibonacci Levels: 38.2% - 50% - 61.8%"),
                "Risk/Lot": st.column_config.TextColumn("Risk/Lot", help="Total Risk per Lot (Entry - SL)"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No Bullish Divergence found right now.")

    st.markdown("---")

    # --- BEARISH RESULTS ---
    st.subheader(f"🔴 Bearish Setups ({len(bear)})")
    if bear:
        df_bear = pd.DataFrame(bear)
        st.dataframe(
            df_bear,
            column_config={
                "Stock": st.column_config.TextColumn("Stock"),
                "Price": st.column_config.NumberColumn("Price", format="%.2f"),
                "SL": st.column_config.NumberColumn("SL", format="%.2f"),
                "Targets (Fib)": st.column_config.TextColumn("Targets (Fib)", help="Fibonacci Levels: 38.2% - 50% - 61.8%"),
                "Risk/Lot": st.column_config.TextColumn("Risk/Lot", help="Total Risk per Lot (SL - Entry)"),
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No Bearish Divergence found right now.")
