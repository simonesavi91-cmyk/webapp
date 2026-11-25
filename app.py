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

# Funzione che passa dati tecnici e news a Gemini
def ask_gemini_analysis(ticker, price, rsi, ema20, ema50, upper_bb, lower_bb, news_list):
    if not ai_ready:
        return "⚠️ Chiave API mancante. Inserisci la chiave di Google AI Studio in alto o nei Secrets per avviare l'analisi."

    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "Nessuna notizia recente disponibile."

    prompt = f"""
    Sei un analista finanziario quantitativo esperto. Analizza l'asset {ticker} (quotato su Revolut) triangolando i dati tecnici e le ultime notizie.

    DATI TECNICI ATTUALI:
    - Prezzo attuale: ${price:.2f}
    - RSI (14 giorni): {rsi:.1f}
    - EMA 20: ${ema20:.2f}
    - EMA 50: ${ema50:.2f}
    - Banda di Bollinger Superiore: ${upper_bb:.2f}
    - Banda di Bollinger Inferiore: ${lower_bb:.2f}

    ULTIME NOTIZIE DI MERCATO:
    {news_text}

    Fornisci una risposta strutturata, chiara e diretta in italiano con:
    1. **Verdetto Operativo:** (COMPRA / VENDI / ATTENDI) accompagnato da una **Percentuale di Successo stimata (0-100%)**.
    2. **Triangolazione Tecnica + News:** Spiega come i grafici e le notizie si confermano o si contraddicono e se ci sono segnali di un movimento imminente.
    3. **Fattori di Rischio:** Cosa potrebbe invalidare questo scenario nel breve termine.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
                signals.append("Sotto Banda Inf.")
            elif close_price >= upper_bb:
                score -= 15
                signals.append("Sopra Banda Sup.")

            score = max(5, min(95, score))

            results.append({
                "Asset": ASSET_NAMES.get(ticker, ticker),
                "Ticker": ticker,
                "Prezzo ($)": round(close_price, 2),
                "RSI": round(rsi, 1),
                "Prob. Buy (%)": score,
                "Prob. Sell (%)": 100 - score,
                "Segnali": ", ".join(signals) if signals else "Neutrale"
            })
        except Exception:
            continue

    return pd.DataFrame(results)

# --- INTERFACCIA A SCHEDE ---
tab1, tab2 = st.tabs(["🧠 Analisi AI con Gemini (Triangolazione)", "🚀 Scanner Automatico (Top 20 Buy/Sell)"])

with tab1:
    st.subheader("🧠 Chiedi a Gemini di analizzare un Asset (Dati Tecnici + Notizie)")
    st.markdown("Scegli un titolo dal menu a tendina oppure digita qualsiasi ticker (es. `NVDA`, `PLTR`, `SOFI`, `BTC-USD`, `HTLM`).")

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        selected_preset = st.selectbox(
            "Scegli dalla lista rapida:",
            options=["-- Scrivi un ticker a mano --"] + [f"{k} - {v}" for k, v in ASSET_NAMES.items()]
        )
    with col_sel2:
        manual_ticker = st.text_input("Oppure scrivi il Ticker qui:", "PLTR").upper().strip()

    if selected_preset != "-- Scrivi un ticker a mano --":
        target_ticker = selected_preset.split(" - ")[0].strip()
    else:
        target_ticker = manual_ticker

    if st.button(f"✨ AVVIA ANALISI AI SU {target_ticker}", type="primary", use_container_width=True):
        if not target_ticker:
            st.warning("Inserisci o seleziona un ticker valido.")
        else:
            with st.spinner(f"⏳ Raccolta dati, notizie e ragionamento di Gemini su {target_ticker}..."):
                try:
                    stock = yf.Ticker(target_ticker)
                    df = stock.history(period="6mo")

                    if df.empty or len(df) < 30:
                        st.error("❌ Dati insufficienti o ticker non trovato su Yahoo Finance.")
                    else:
                        close_price = float(df['Close'].iloc[-1])
                        delta = df['Close'].diff()
                        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                        rsi = float(100 - (100 / (1 + (gain / loss).iloc[-1])))

                        ema20 = float(df['Close'].ewm(span=20, adjust=False).mean().iloc[-1])
                        ema50 = float(df['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
                        sma20 = float(df['Close'].rolling(window=20).mean().iloc[-1])
                        std20 = float(df['Close'].rolling(window=20).std().iloc[-1])
                        upper_bb = sma20 + (std20 * 2)
                        lower_bb = sma20 - (std20 * 2)

                        raw_news = stock.news
                        headlines = []
                        if raw_news:
                            for item in raw_news[:7]:
                                title = item.get('title') or item.get('content', {}).get('title', '')
                                if title:
                                    headlines.append(title)

                        ai_report = ask_gemini_analysis(
                            target_ticker, close_price, rsi, ema20, ema50, upper_bb, lower_bb, headlines
                        )

                        st.divider()
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Prezzo", f"${close_price:.2f}")
                        c2.metric("RSI (14)", f"{rsi:.1f}")
                        c3.metric("EMA 20", f"${ema20:.2f}")
                        c4.metric("EMA 50", f"${ema50:.2f}")

                        st.markdown("### 🧠 Riflessione e Verdetto di Gemini")
                        st.markdown(ai_report)

                        with st.expander("📰 Vedi le notizie analizzate dall'AI"):
                            if headlines:
                                for h in headlines:
                                    st.markdown(f"- {h}")
                            else:
                                st.write("Nessuna notizia recente restituita dal feed.")

                        st.markdown("### 📊 Grafico Andamento (6 Mesi)")
                        st.line_chart(df['Close'])

                except Exception as e:
                    st.error(f"Errore durante l'elaborazione: {e}")

with tab2:
    if st.button("🔄 Aggiorna Scansione Mercato"):
        st.cache_data.clear()

    with st.spinner("⏳ Scansione di azioni, mid-cap e criptovalute in corso..."):
        df_results = scan_market(REVOLUT_WATCHLIST)

    if df_results.empty:
        st.error("Errore nel recupero dei dati di mercato.")
    else:
        top_20_buy = df_results.sort_values(by="Prob. Buy (%)", ascending=False).head(20)[
            ["Asset", "Ticker", "Prezzo ($)", "RSI", "Prob. Buy (%)", "Segnali"]
        ].reset_index(drop=True)

        top_20_sell = df_results.sort_values(by="Prob. Sell (%)", ascending=False).head(20)[
            ["Asset", "Ticker", "Prezzo ($)", "RSI", "Prob. Sell (%)", "Segnali"]
        ].reset_index(drop=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🟢 TOP 20 DA COMPRARE")
            st.dataframe(
                top_20_buy, use_container_width=True, hide_index=True,
                column_config={"Prob. Buy (%)": st.column_config.ProgressColumn("Score Tecnico", format="%d%%", min_value=0, max_value=100)}
            )

        with col2:
            st.subheader("🔴 TOP 20 DA VENDERE")
            st.dataframe(
                top_20_sell, use_container_width=True, hide_index=True,
                column_config={"Prob. Sell (%)": st.column_config.ProgressColumn("Score Vendita", format="%d%%", min_value=0, max_value=100)}
            )
