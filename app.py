import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Configurazione della pagina
st.set_page_config(page_title="Revolut Auto-Scanner", page_icon="🚀", layout="wide")

st.title("🚀 Revolut Market Auto-Scanner")
st.markdown("Scansione automatica di azioni, ETF e criptovalute disponibili su Revolut. Classifica in tempo reale dei **Top 20 Buy** e **Top 20 Sell** basata su indicatori tecnici combinati.")

# Paniere ampio di asset popolari disponibili su Revolut (Azioni US, Tech, Cripto, ETF)
REVOLUT_WATCHLIST = [
    # Big Tech & US Stocks
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "AMD", "INTC",
    "CRM", "ADBE", "PYPL", "SQ", "SHOP", "UBER", "ABNB", "PLTR", "COIN", "HOOD",
    "DIS", "NKE", "SBUX", "MCD", "KO", "PEP", "WMT", "COST", "JPM", "BAC",
    "V", "MA", "GS", "JNJ", "PFE", "MRNA", "XOM", "CVX", "BA", "F", "GM",
    "RIVN", "LCID", "NIO", "BABA", "TSM", "ASML", "SPOT", "SNAP", "ROKU",
    # Criptovalute principali su Revolut
    "BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "DOGE-USD", 
    "AVAX-USD", "DOT-USD", "LINK-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
    "UNI7083-USD", "ATOM-USD", "XLM-USD", "ALGO-USD", "NEAR-USD", "ICP-USD",
    # ETF Popolari
    "SPY", "QQQ", "VOO", "VTI", "ARKK", "GLD", "SLV", "TLT"
]

# Funzione con cache (aggiorna i dati ogni 15 minuti per velocizzare l'app)
@st.cache_data(ttl=900)
def scan_market(tickers):
    # Scarica i dati storici degli ultimi 6 mesi per tutti i ticker insieme
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

            # Calcolo Score Probabilistico (da 0 a 100: >50 = Buy, <50 = Sell)
            score = 50
            signals = []

            # 1. Peso RSI
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

            # 2. Peso Trend EMA
            if close_price > ema_20 > ema_50:
                score += 15
                signals.append("Trend Rialzista")
            elif close_price < ema_20 < ema_50:
                score -= 15
                signals.append("Trend Ribassista")

            # 3. Peso Bande di Bollinger
            if close_price <= lower_bb:
                score += 15
                signals.append("Sotto Banda Inf. BB")
            elif close_price >= upper_bb:
                score -= 15
                signals.append("Sopra Banda Sup. BB")

            score = max(5, min(95, score))
            
            # Calcolo probabilità direzionale
            buy_prob = score
            sell_prob = 100 - score

            results.append({
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

# Pulsante per forzare l'aggiornamento manuale
if st.button("🔄 Aggiorna Scansione Mercato Ora"):
    st.cache_data.clear()

with st.spinner("⏳ Scansione di oltre 75 asset e criptovalute su Revolut in corso..."):
    df_results = scan_market(REVOLUT_WATCHLIST)

if df_results.empty:
    st.error("Errore nel recupero dei dati di mercato. Riprova tra qualche istante.")
else:
    # Ordina e filtra i Top 20 Buy e Top 20 Sell
    top_20_buy = df_results.sort_values(by="Prob. Buy (%)", ascending=False).head(20)[
        ["Ticker", "Prezzo ($)", "RSI", "Prob. Buy (%)", "Segnali Principali"]
    ].reset_index(drop=True)

    top_20_sell = df_results.sort_values(by="Prob. Sell (%)", ascending=False).head(20)[
        ["Ticker", "Prezzo ($)", "RSI", "Prob. Sell (%)", "Segnali Principali"]
    ].reset_index(drop=True)

    # Visualizzazione a due colonne affiancate
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
