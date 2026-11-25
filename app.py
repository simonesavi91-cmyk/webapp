import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from google import genai

# Configurazione della pagina
st.set_page_config(page_title="Revolut AI Market Intelligence", page_icon="🧠", layout="wide")

st.title("🧠 Revolut AI Market Intelligence")
st.markdown("Scanner quantitativo automatico + **Ragionamento AI di Gemini** su dati tecnici, notizie e titoli Mid-Cap.")

# Inizializza il client Gemini (legge dai Secrets o da una chiave inserita a mano)
API_KEY = st.secrets.get("GEMINI_API_KEY", "")

if API_KEY:
    client = genai.Client(api_key=API_KEY)
    ai_ready = True
else:
    ai_ready = False
    st.info("💡 **Nota:** Inserisci la tua `GEMINI_API_KEY` nei Secrets di Streamlit (oppure incollala nel campo qui sotto) per attivare il cervello AI.")
    manual_key = st.text_input("Incolla qui la tua chiave API di Google AI Studio (opzionale se già nei Secrets):", type="password")
    if manual_key:
        client = genai.Client(api_key=manual_key)
        ai_ready = True

# Watchlist ampliata (Big Tech, Mid-Cap, Cripto, ETF su Revolut)
ASSET_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "GOOGL": "Alphabet Inc. (Google)", "AMZN": "Amazon.com, Inc.", "META": "Meta Platforms, Inc.",
    "TSLA": "Tesla, Inc.", "NFLX": "Netflix, Inc.", "AMD": "Advanced Micro Devices",
    "INTC": "Intel Corporation", "CRM": "Salesforce, Inc.", "PYPL": "PayPal Holdings",
    "SQ": "Block, Inc.", "SHOP": "Shopify Inc.", "UBER": "Uber Technologies",
    "PLTR": "Palantir Technologies", "COIN": "Coinbase Global", "HOOD": "Robinhood Markets",
    "ROKU": "Roku, Inc.", "SNAP": "Snap Inc.", "DKNG": "DraftKings Inc.",
    "RBLX": "Roblox Corporation", "SOFI": "SoFi Technologies", "AFRM": "Affirm Holdings",
    "UPST": "Upstart Holdings", "NIO": "NIO Inc.", "RIVN": "Rivian Automotive",
    "IONQ": "IonQ, Inc.", "RGTI": "Rigetti Computing", "SMCI": "Super Micro Computer",
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana", "XRP-USD": "XRP",
    "ADA-USD": "Cardano", "DOGE-USD": "Dogecoin", "AVAX-USD": "Avalanche",
    "LINK-USD": "Chainlink", "NEAR-USD": "NEAR Protocol", "RENDER-USD": "Render",
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust", "ARKK": "ARK Innovation ETF"
}

REVOLUT_WATCHLIST = list(ASSET_NAMES.keys())

# Funzione che passa dati tecnici e news a Gemini con il modello aggiornato
def ask_gemini_analysis(ticker, price, rsi, ema20, ema50, upper_bb, lower_bb, news_list):
    if not ai_ready:
        return "⚠️ Chiave API mancante. Inserisci la chiave di Google AI Studio in alto o nei Secrets per avviare l'analisi."

    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "Nessuna notizia recente disponibile."

    prompt = f"""
    Sei un analista finanziario quantitativo esperto. Analizza l'asset {ticker} (quotato su Revolut) triangolando i dati tecnici e le ultime notizie.

    DATI TECNICI ATTUALI:
    - Prezzo attuale: ${price:.2f}
    - RSI (14 giorni): {rsi:.1f}
    - EMA 20: ${ema20:.2f} - EMA 50:${ema50:.2f}
    - Banda di Bollinger Superiore: ${upper_bb:.2f} - Banda di Bollinger Inferiore:${lower_bb:.2f}

    ULTIME NOTIZIE DI MERCATO:
    {news_text}

    Fornisci una risposta strutturata, chiara e diretta in italiano con:
    1. **Verdetto Operativo:** (COMPRA / VENDI / ATTENDI) accompagnato da una **Percentuale di Successo stimata (0-100%)**.
    2. **Triangolazione Tecnica + News:** Spiega come i grafici e le notizie si confermano o si contraddicono e se ci sono segnali di un movimento imminente.
    3. **Fattori di Rischio:** Cosa potrebbe invalidare questo scenario nel breve termine.
    """

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"❌ Errore durante la chiamata a Gemini: {e}"

# Funzione di screening veloce per le classifiche Top 20
@st.cache_data(ttl=900)
def scan_market(tickers):
    data = yf.download(tickers, period="6mo", group_by='ticker', threads=True, progress=False)
    results = []

    for ticker in tickers:
        try:
            df = data[ticker].dropna()
            if len(df) < 50:
                continue

            close_price = float(df['Close'].iloc[-1])
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = float(100 - (100 / (1 + (gain / loss).iloc[-1])))

            ema_20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
            ema_50 = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
            sma_20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
            std_20 = float(df['Close'].rolling(window=20).std().iloc[-1])
            lower_bb = sma_20 - (std_20 * 2)
            upper_bb = sma_20 + (std_20 * 2)

            if np.isnan(rsi) or np.isnan(ema_50):
                continue

            score = 50
            signals = []

            if rsi < 30:
                score += 25
            
