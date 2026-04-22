import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf # NEW: For real data
from datetime import datetime, timedelta

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StockConnect AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .big-number {
        font-size: 3em;
        font-weight: bold;
        color: #00CC96;
    }
    .label {
        font-size: 1.2em;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# --- TITLE ---
col1, col2 = st.columns([1, 4])
with col1:
    st.image("https://img.icons8.com/color/96/000000/bullish.png", width=80) 
with col2:
    st.title("StockConnect Pro")
    st.caption("Advanced LSTM Neural Network Prediction System (Real-World Data Mode)")

st.divider()

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_url = st.text_input("API URL", "http://127.0.0.1:8000")
    
    st.subheader("Data Source")
    ticker = st.selectbox("Stock Ticker Symbol", ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"])
    
    if st.button("🔄 Fetch Live Data", type="primary"):
        try:
            with st.spinner(f"Downloading {ticker} data..."):
                # We need 60 days of history. 
                # To be safe (weekends/holidays), we request 100 days back and take the last 60.
                end_date = datetime.now()
                start_date = end_date - timedelta(days=120) 
                
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if len(df) < 60:
                    st.error(f"Not enough data found for {ticker}. Need at least 60 trading days.")
                else:
                    # Take last 60 'Close' prices
                    # Handle MultiIndex if necessary, squeeze helps flatten it
                    last_60 = df['Close'].tail(60).squeeze()
                    
                    st.session_state['prices'] = last_60.tolist()
                    st.session_state['dates'] = last_60.index.strftime('%Y-%m-%d').tolist()
                    st.session_state['current_ticker'] = ticker
                    st.success(f"Loaded {len(last_60)} days for {ticker}!")
                    
        except Exception as e:
            st.error(f"Failed to download data: {e}")

# --- MAIN CONTENT ---
if 'prices' not in st.session_state:
    # Initial Fallback
    st.info("👈 Please enter a Ticker (like AAPL, NVDA, TSLA) and click 'Fetch Live Data'")
    st.stop() # Stop rendering until data is loaded

prices = st.session_state['prices']
dates = st.session_state.get('dates', None)
current_ticker = st.session_state.get('current_ticker', "Unknown")

# 1. Visualization
st.subheader(f"📊 Market Analysis: {current_ticker} (Last 60 Days)")

# Create beautiful Plotly chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=dates if dates else list(range(len(prices))),
    y=prices, 
    mode='lines+markers',
    name='History',
    line=dict(color='#636EFA', width=3),
    fill='tozeroy',
    hovertemplate='<b>Date</b>: %{x}<br><b>Price</b>: $%{y:.2f}<extra></extra>'
))
fig.update_layout(
    height=400,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis_title="Date" if dates else "Trading Days",
    yaxis_title="Price ($)"
)
st.plotly_chart(fig, use_container_width=True)

# 2. Action Area
col_btn, col_res = st.columns([1, 2])

with col_btn:
    st.write("### 🤖 AI Prediction")
    st.write(f"Ask the Neural Network to predict the price for **tomorrow** based on this {current_ticker} trend.")
    
    predict_clicked = st.button("🚀 Predict Next Day Price", use_container_width=True)

with col_res:
    if predict_clicked:
        try:
            with st.spinner("Consulting the Neural Network..."):
                payload = {"features": prices, "ticker": current_ticker}
                response = requests.post(f"{api_url}/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    pred_price = result["prediction_price_usd"]
                    confidence = result.get("prediction_raw_scaled", 0) 
                    
                    last_price = prices[-1]
                    delta = pred_price - last_price
                    delta_percent = (delta / last_price) * 100
                    
                    # Display Result Card
                    color = "#00CC96" if delta > 0 else "#EF553B"
                    arrow = "▲" if delta > 0 else "▼"
                    
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Predicted Price ({current_ticker})</div>
                        <div class="big-number" style="color: {color}">
                            ${pred_price:.2f}
                        </div>
                        <div style="color: {color}; margin-top: 10px;">
                            {arrow} {delta:+.2f} ({delta_percent:+.2f}%)
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.error(f"Server Error: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            st.error("❌ Could not connect to API. Is 'run_api.bat' running?")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- FOOTER ---
st.write("---")
with st.expander("Show API Response Debug Info"):
    if predict_clicked and 'result' in locals():
        st.json(result)
