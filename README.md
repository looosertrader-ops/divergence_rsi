⚡ QuantScan: NSE RSI Divergence Scanner

This is a Python-based web application that scans the NSE (National Stock Exchange of India) F&O list for trading setups based on RSI Divergence.

How it Works

Fetches Data: Automatically pulls the latest F&O stock list from NSE Archives.

Downloads Price History: Uses Yahoo Finance to get OHLCV data.

Applies Logic: - Bullish: Price is Lower than 14 days ago, but RSI is Higher.

Bearish: Price is Higher than 14 days ago, but RSI is Lower.

How to Run

This app is built with Streamlit.

Install requirements: pip install -r requirements.txt

Run app: streamlit run app.py