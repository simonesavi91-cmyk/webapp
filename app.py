import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# Configurazione della pagina
st.set_page_config(page_title="Revolut Pro Market Intelligence", page_icon="🌐", layout="wide")

st.title("🌐 Revolut Market Intelligence & News Scanner")
st.markdown("Scanner automatico con analisi tecnica, sentiment delle notizie in tempo reale e ricerca avanzata (incluso il segmento **Mid-Cap**).")

# Watchlist ampliata con Big Tech, Cripto, ETF e Mid-Cap popolari su Revolut
ASSET_NAMES = {
    # Big Tech & Large Cap US
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Google)", "AMZN": "Amazon.com, Inc.", "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.", "NFLX": "Netflix, Inc.", "AMD": "Advanced Micro Devices",
    "INTC": "Intel Corporation", "CRM": "Salesforce, Inc.", "ADBE": "Adobe Inc.",
    "PYPL": "PayPal Holdings, Inc.", "SQ": "Block, Inc.", "SHOP": "Shopify Inc.",
    "UBER": "Uber Technologies", "ABNB": "Airbnb, Inc.", "PLTR": "Palantir Technologies",
    "COIN": "Coinbase Global, Inc.", "HOOD": "Robinhood Markets", "DIS": "The Walt Disney Company",
    # Mid-Cap & Growth Stocks
    "Roku, Inc.": "ROKU", "ROKU": "Roku, Inc.", "SNAP": "Snap Inc.", "PINS": "Pinterest, Inc.",
    "ROKU": "Roku, Inc.", "DKNG": "DraftKings Inc.", "RBLX": "Roblox Corporation",
    "ETSY": "Etsy, Inc.", "DOCU": "DocuSign, Inc.", "ZM": "Zoom Video Communications",
    "PTON": "Peloton Interactive", "UPST": "Upstart Holdings", "SOFI": "SoFi Technologies",
    "AFRM": "Affirm Holdings", "NIO": "NIO Inc.", "RIVN": "Rivian Automotive",
    "LCID": "Lucid Group, Inc.", "IONQ": "IonQ, Inc.", "RGTI": "Rigetti Computing",
    # Criptovalute principali su Revolut
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "XRP",
    "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche",
    "DOT-USD": "Polkadot", "LINK-USD": "Chainlink", "MATIC-USD": "Polygon",
    "LTC-USD": "Litecoin", "NEAR-USD": "NEAR Protocol", "ICP-USD": "Internet Computer",
    # ETF Popolari
    "SPY": "SPDR S&P 500 ETF Trust", "QQQ": "Invesco QQQ Trust", "ARKK": "ARK Innovation ETF"
}

REVOLUT_WATCHLIST = list(set(ASSET_NAMES.keys()))

# Funzione per analizzare le notizie e calcolare il sentiment
def get_news_sentiment(ticker_symbol):
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        news = ticker_obj.news
        if not news:
            return 50, "Nessuna notizia recente rilevante."
        
        score_news = 50
        summary_points = []
        
        # Analisi euristica delle parole chiave nelle notizie recenti
        positive_keywords = ["surge", "jump", "beat", "upgrade", "growth", "bull", "rally", "record", "profit", "partnership"]
        negative_keywords = ["drop", "fall", "miss", "downgrade", "loss", "bear", "crash", "investigation", "lawsuit", "risk"]
        
        count = 0
        for item in news[:5]: # Analizza le ultime 5 notizie
            title = item.get('title', '').lower()
            publisher = item.get('publisher', 'News')
            
            # Controllo parole chiave
            pos_matches = sum(1 for kw in positive_keywords if kw in title)
            neg_matches = sum(1 for kw in negative_keywords if kw in title)
            
            if pos_matches > neg_matches:
                score_news += 8
                summary_points.append(f"🟢 [{publisher}] {item.get('title')}")
            elif neg_matches > pos_matches:
                score_news -= 8
                summary_points.append(f"🔴 [{publisher}] {item.get('title')}")
            else:
                summary_points.append(f"⚪ [{publisher}] {item.get('title')}")
            count += 1
            
        final_news_score = max(20, min(80, score_news))
        return final_news_score, summary_points
    except Exception:
        return 50, ["Impossibile recuperare le notizie in tempo reale."]

# Funzione di scansione unificata (Tecnica + News)
@st.cache_data(ttl=900)
def scan_market(tickers):
    results = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")
            if df.empty or len(df) < 50:
                continue

            # Calcolo Indicatori Tecnici
            close_price = float(df['Close'].iloc[-1])
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = float(100 - (100 / (1 + (gain / loss).iloc[-1])))

            ema_20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
            ema_50 = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[-1])

            # Punteggio Tecnico
            tech_score = 50
            signals = []
            if rsi < 30:
                tech_score += 25
                signals.append("RSI Ipervenduto")
            elif rsi > 70:
                tech_score -= 25
                signals.append("RSI Ipercomprato")

            if close_price > ema_20 > ema_50:
                tech_score += 15
                signals.append("Trend Rialzista")
            elif close_price < ema_20 < ema_50:
                tech_score -= 15
                signals.append("Trend Ribassista")

            # Sentiment Notizie
            news_score, news_headlines = get_news_sentiment(ticker)

            # Triangolazione Finale (70% Tecnica + 30% Notizie)
            final_score = int((tech_score * 0.7) + (news_score * 0.3))
            final_score = max(5, min(95, final_score))

            results.append({
                "Asset": ASSET_NAMES.get(ticker, ticker),
                "Ticker": ticker,
                "Prezzo ($)": round(close_price, 2),
                "RSI": round(rsi, 1),
                "Prob. Buy (%)": final_score,
                "Prob. Sell (%)": 100 - final_score,
                "Segnali": ", ".join(signals) if signals else "Neutrale",
                "Notizie": news_headlines
            })
        except Exception:
            continue

    return pd.DataFrame(results)

# --- INTERFACCIA A SCHEDE (TABS) ---
tab1, tab2 = st.tabs(["🚀 Scanner Automatico Mercato (Top 20)", "🔍 Ricerca Dettagliata per Singolo Asset"])

with tab1:
    if st.button("🔄 Aggiorna Scansione e Notizie Ora"):
        st.cache_data.clear()

    with st.spinner("⏳ Analisi tecnica e scraping delle notizie finanziarie in corso (inclusi Mid-Cap)..."):
        df_results = scan_market(REVOLUT_WATCHLIST)

    if df_results.empty:
        st.error("Errore nel recupero dei dati. Riprova tra qualche istante.")
    else:
        top_20_buy = df_results.sort_values(by="Prob. Buy (%)", ascending=False).head(20)[
            ["Asset", "Ticker", "Prezzo ($)", "RSI", "Prob. Buy (%)", "Segnali"]
        ].reset_index(drop=True)

        top_20_sell = df_results.sort_values(by="Prob. Sell (%)", ascending=False).head(20)[
            ["Asset", "Ticker", "Prezzo ($)", "RSI", "Prob. Sell (%)", "Segnali"]
        ].reset_index(drop=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟢 TOP 20 OPPORTUNITÀ DI ACQUISTO (BUY)")
            st.dataframe(top_20_buy, use_container_width=True, hide_index=True,
                         column_config={"Prob. Buy (%)": st.column_config.ProgressColumn("Score Triangolato", format="%d%%", min_value=0, max_value=100)})

        with col2:
            st.subheader("🔴 TOP 20 OPPORTUNITÀ DI VENDITA (SELL)")
            st.dataframe(top_20_sell, use_container_width=True, hide_index=True,
                         column_config={"Prob. Sell (%)": st.column_config.ProgressColumn("Score Vendita", format="%d%%", min_value=0, max_value=100)})

with tab2:
    st.subheader("🔍 Cerca e Analizza un Asset Specifico (Azioni, Mid-Cap o Cripto)")
    search_query = st.text_input("Inserisci il Ticker o il nome (es. AAPL, PLTR, ROKU, BTC-USD):", "PLTR").upper().strip()

    if st.button("🔎 Analizza Asset in Profondità"):
        with st.spinner(f"Analisi di mercato e raccolta notizie per {search_query}..."):
            try:
                stock = yf.Ticker(search_query)
                df = stock.history(period="6mo")
