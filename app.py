import streamlit as st
import yfinance as yf
import pandas as pd

# Configurazione della pagina
st.set_page_config(page_title="Revolut Market Analyzer", page_icon="📈", layout="centered")

st.title("📈 Revolut Market Analyzer")
st.markdown("Monitora i titoli quotati su Revolut, calcola gli indicatori tecnici e scopri la **percentuale di successo** in tempo reale.")

# Input per il ticker
ticker_input = st.text_input("Inserisci il Ticker (es. AAPL, TSLA, BTC-USD, NVDA):", "AAPL").upper()

if st.button("🔍 Analizza Titolo"):
    if not ticker_input:
        st.warning("Per favore, inserisci un ticker valido.")
    else:
        with st.spinner(f"Analisi in corso per {ticker_input}..."):
            try:
                # Scarica i dati storici
                stock = yf.Ticker(ticker_input)
                df = stock.history(period="6mo")
                
                if df.empty or len(df) < 50:
                    st.error("❌ Ticker non trovato o dati insufficienti su Yahoo Finance.")
                else:
                    # Calcolo RSI (14 periodi)
                    delta = df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['RSI'] = 100 - (100 / (1 + rs))

                    # Medie Mobili ed EMA
                    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
                    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()

                    last_row = df.iloc[-1]
                    close_price = last_row['Close']
                    rsi = last_row['RSI']
                    ema_20 = last_row['EMA_20']
                    ema_50 = last_row['EMA_50']

                    # Logica di Scoring e Percentuale di Successo
                    score = 50  # Base neutrale
                    reasons = []

                    if rsi < 30:
                        score += 30
                        reasons.append("🟢 **RSI in ipervenduto (< 30):** Forte segnale di rimbalzo/acquisto.")
                    elif rsi > 70:
                        score -= 30
                        reasons.append("🔴 **RSI in ipercomprato (> 70):** Rischio correzione o vendita.")
                    else:
                        reasons.append(f"⚪ RSI neutrale ({rsi:.1f}).")

                    if close_price > ema_20 and ema_20 > ema_50:
                        score += 20
                        reasons.append("🟢 **Trend Rialzista:** Prezzo sopra EMA20 e EMA50.")
                    elif close_price < ema_20 and ema_20 < ema_50:
                        score -= 20
                        reasons.append("🔴 **Trend Ribassista:** Prezzo sotto EMA20 e EMA50.")

                    score = max(5, min(95, score))

                    # Determinazione Azione
                    if score >= 70:
                        action = "🟢 COMPRA (STRONG BUY)"
                        bar_color = "green"
                    elif score <= 30:
                        action = "🔴 VENDI (STRONG SELL)"
                        bar_color = "red"
                    else:
                        action = "🟡 MANTIENI / ATTENDI (HOLD)"
                        bar_color = "orange"

                    # --- VISUALIZZAZIONE GRAFICA NELLA WEB APP ---
                    st.divider()
                    
                    col1, col2 = st.columns(2)
                    col1.metric(label="Prezzo Attuale", value=f"${close_price:.2f}")
                    col2.metric(label="RSI (14)", value=f"{rsi:.1f}")

                    st.subheader(f"Consiglio: {action}")
                    
                    # Barra di progresso per la percentuale di successo
                    st.markdown(f"**Percentuale di Successo Stimata:** {score}%")
                    st.progress(score / 100)

                    st.markdown("### 🔍 Dettagli Tecnici:")
                    for r in reasons:
                        st.markdown(f"- {r}")

                    # Grafico storico dei prezzi integrato
                    st.markdown("### 📊 Andamento Ultimi 6 Mesi")
                    st.line_chart(df['Close'])

            except Exception as e:
                st.error(f si è verificato un errore durante l'elaborazione: {e})