import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Configurazione della pagina
st.set_page_config(page_title="Revolut Auto-Scanner", page_icon="🚀", layout="wide")

st.title("🚀 Revolut Market Auto-Scanner")
st.markdown("Scansione automatica di azioni, ETF e criptovalute disponibili su Revolut con nomi estesi. Classifica in tempo reale dei **Top 20 Buy** e **Top 20 Sell**.")

# Dizionario dei ticker con i rispettivi nomi completi per una lettura immediata
ASSET_NAMES = {
    # Big Tech & US Stocks
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com, Inc.",
    "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.",
    "NFLX": "Netflix, Inc.",
    "AMD": "Advanced Micro Devices",
    "INTC": "Intel Corporation",
    "CRM": "Salesforce, Inc.",
    "ADBE": "Adobe Inc.",
    "PYPL": "PayPal Holdings, Inc.",
    "SQ": "Block, Inc. (Square)",
    "SHOP": "Shopify Inc.",
    "UBER": "Uber Technologies",
    "ABNB": "Airbnb, Inc.",
    "PLTR": "Palantir Technologies",
    "COIN": "Coinbase Global, Inc.",
    "HOOD": "Robinhood Markets",
    "DIS": "The Walt Disney Company",
    "NKE": "NIKE, Inc.",
    "SBUX": "Starbucks Corporation",
    "MCD": "McDonald's Corporation",
    "KO": "The Coca-Cola Company",
    "PEP": "PepsiCo, Inc.",
    "WMT": "Walmart Inc.",
    "COST": "Costco Wholesale Corp.",
    "JPM": "JPMorgan Chase & Co.",
    "BAC": "Bank of America Corp.",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "GS": "The Goldman Sachs Group",
    "JNJ": "Johnson & Johnson",
    "PFE": "Pfizer Inc.",
    "MRNA": "Moderna, Inc.",
    "XOM": "Exxon Mobil Corporation",
    "CVX": "Chevron Corporation",
    "BA": "The Boeing Company",
    "F": "Ford Motor Company",
    "GM": "General Motors Company",
    "RIVN": "Rivian Automotive",
    "LCID": "Lucid Group, Inc.",
    "NIO": "NIO Inc.",
    "BABA": "Alibaba Group Holding",
    "TSM": "Taiwan Semiconductor",
    "ASML": "ASML Holding N.V.",
    "SPOT": "Spotify Technology S.A.",
    "SNAP": "Snap Inc.",
    "ROKU": "Roku, Inc.",
    # Criptovalute principali su Revolut
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
    "SOL-USD": "Solana",
    "XRP-USD": "XRP",
    "ADA-USD": "Cardano",
    "DOGE-USD": "Dogecoin",
    "AVAX-USD": "Avalanche",
    "DOT-USD": "Polkadot",
    "LINK-USD": "Chainlink",
    "MATIC-USD": "Polygon",
    "LTC-USD": "Litecoin",
    "BCH-USD": "Bitcoin Cash",
    "UNI7083-USD": "Uniswap",
    "ATOM-USD": "Cosmos",
    "XLM-USD": "Stellar",
    "ALGO-USD": "Algorand",
    "NEAR-USD": "NEAR Protocol",
    "ICP-USD": "Internet Computer",
    # ETF Popolari
    "SPY": "SPDR S&P 500 ETF Trust",
    "QQQ": "Invesco QQQ Trust (Nasdaq 100)",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "ARKK": "ARK Innovation ETF",
    "GLD": "SPDR Gold Shares",
    "SLV": "iShares Silver Trust",
    "TLT": "iShares 20+ Year Treasury Bond"
}

REVOLUT_WATCHLIST = list(ASSET_NAMES.keys())

# Funzione con cache (aggiorna i dati ogni 15 minuti)
@st.cache_data(ttl=900)
def scan_market(tickers):
    data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    results = []

    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if len(df) < 50:
                continue

            # Calcolo RSI (14)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Calcolo Medie Mobili (EMA 20 e EMA 50)
            df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

            # Calcolo Bande di Bollinger (20, 2)
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['STD_20'] = df['Close'].rolling(window=20).std()
            df['Upper_BB'] = df['SMA_20'] + (df['STD_20'] * 2)
            df['Lower_BB'] = df['SMA_20'] - (df['STD_20'] * 2)

            last = df.iloc[-1]
            close_price = float(last['Close'])
            rsi = float(last['RSI'])
            ema_20 = float(last['EMA_20'])
            ema_50 = float(last['EMA_50'])
            lower_bb = float(last['Lower_BB'])
            upper_bb = float(last['Upper_BB'])

            if np.isnan(rsi) or np.isnan(ema_50):
                continue

            # Calcolo Score Probabilistico
            score = 50
            signals = []

            if rsi < 30:
                score += 25
                signals.append("RSI Ipervenduto")
            elif rsi < 40:
                score += 10
                signals.append("RSI Basso")
            elif rsi > 70:
                score -= 25
                signals.append("RSI Ipercomprato")
            elif rsi > 60:
                score -= 10
                signals.append("RSI Alto")

            if close_price > ema_20 > ema_50:
                score += 15
                signals.append("Trend Rialzista")
            elif close_price < ema_20 < ema_50:
                score -= 15
                signals.append("Trend Ribassista")

            if close_price <= lower_bb:
                score += 15
                signals.append("Sotto Banda Inf. BB")
            elif close_price >= upper_bb:
                score -= 15
                signals.append("Sopra Banda Sup. BB")

            score = max(5, min(95, score))
            
            buy_prob = score
            sell_prob = 100 - score

            results.append({
                "Asset": ASSET_NAMES.get(ticker, ticker),
                "Ticker": ticker,
                "Prezzo ($)": round(close_price, 2),
                "RSI": round(rsi, 1),
                "Prob. Buy (%)": buy_prob,
                "Prob. Sell (%)": sell_prob,
                "Segnali Principali": ", ".join(signals) if signals else "Neutrale"
            })
        except Exception:
            continue

    return pd.DataFrame(results)

if st.button("🔄 Aggiorna Scansione Mercato Ora"):
    st.cache_data.clear()

with st.spinner("⏳ Analisi di azioni, ETF e criptovalute su Revolut in corso..."):
    df_results = scan_market(REVOLUT_WATCHLIST)

if df_results.empty:
    st.error("Errore nel recupero dei dati di mercato. Riprova tra qualche istante.")
else:
    top_20_buy = df_results.sort_values(by="Prob. Buy (%)", ascending=False).head(20)[
        ["Asset", "Ticker", "Prezzo ($)", "RSI", "Prob. Buy (%)", "Segnali Principali"]
    ].reset_index(drop=True)

    top_20_sell = df_results.sort_values(by="Prob. Sell (%)", ascending=False).head(20)[
        ["Asset", "Ticker", "Prezzo ($)", "RSI", "Prob. Sell (%)", "Segnali Principali"]
    ].reset_index(drop=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🟢 TOP 20 DA COMPRARE (BUY)")
        st.dataframe(
            top_20_buy,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prob. Buy (%)": st.column_config.ProgressColumn(
                    "Successo Buy",
                    help="Probabilità stimata di rialzo",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )

    with col2:
        st.subheader("🔴 TOP 20 DA VENDERE (SELL)")
        st.dataframe(
            top_20_sell,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Prob. Sell (%)": st.column_config.ProgressColumn(
                    "Successo Sell",
                    help="Probabilità stimata di ribasso/presa di profitto",
                    format="%d%%",
                    min_value=0,
                    max_value=100,
                )
            }
        )
