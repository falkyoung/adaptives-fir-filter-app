"""
Streamlit-Oberfläche: bis zu 3 Aufnahmen nebeneinander vergleichen.

  Aufnahme(n) -> main.c (LMS) -> Plots, je eine Spalte.

Start:   streamlit run app.py
"""
import io

import numpy as np
from scipy.io import wavfile
import streamlit as st

import lpc_runner as L
import lpc_plots as P

st.set_page_config(page_title="Adaptives FIR-Filter", layout="wide")
st.title("Adaptives FIR-Filter zur Spektralschätzung (LPC)")
st.write(
    "Nimm **bis zu 3 kurze Aufnahmen** auf (je ~3-5 s, ein gehaltener Vokal) und "
    "vergleiche sie direkt nebeneinander. Z.B. **i / a / o** (verschiedene Vokale) "
    "oder **3x a in verschiedenen Tonhöhen**."
)
with st.expander("Was ist LPC?"):
    st.markdown(
        "**LPC = Linear Predictive Coding** (lineare Vorhersage). Das aktuelle "
        "Sample wird aus den vergangenen Samples geschätzt - per FIR-Filter mit "
        "16 Koeffizienten:\n\n"
        "- **Phase 1 (Schätzung):**  x_dach(t) = Summe  a_k * x(t-1-k)\n"
        "- **Fehler:**  e(t) = x(t) - x_dach(t)  ,   Kosten  E(t) = 0.5 * e(t)^2\n"
        "- **Phase 2 (Anpassung):**  Koeffizienten a_k per Gradientenabstieg "
        "langsam anpassen, bis E(t) klein wird (ideal 0).\n\n"
        "Die eingependelten Koeffizienten beschreiben den **Vokaltrakt**. Ihr "
        "Spektrum H(z) = 1/A(z) ist die **Spektralschätzung** - die Spitzen sind "
        "die **Formanten**. Vokale unterscheiden sich in den Formanten, die "
        "Tonhöhe (Grundfrequenz) kaum."
    )


@st.cache_data(show_spinner=False)
def process(wav_bytes):
    """Audio-Bytes -> main.c -> Arrays. Gecacht, damit unveränderte
    Aufnahmen bei jedem Rerun nicht neu gerechnet werden."""
    samples = L.read_wav(io.BytesIO(wav_bytes))
    t, x, xhat, err, coeffs = L.run_lpc(samples)
    return samples, t, x, xhat, err, coeffs


SMALL = (5, 2.4)   # kompakte Plotgroesse fuer schmale Spalten

st.caption(
    "⚠️ Hinweis: Geräuschunterdrückung von Browser, System und Mikrofon muss "
    "deaktiviert sein, sonst wird die Aufnahme nach 2-3 s unterdrückt. "
    "Alternativ extern aufnehmen und je Spalte als WAV hochladen."
)

cols = st.columns(3)
results = []   # (label, coeffs) fuer das Overlay unten

for i, col in enumerate(cols):
    with col:
        st.markdown(f"### Aufnahme {i + 1}")
        audio = st.audio_input("aufnehmen", key=f"audio_{i}",
                               label_visibility="collapsed")
        st.caption("oder: WAV-Datei hochladen")
        upcol, playcol = st.columns([2, 1], vertical_alignment="center")
        upload = upcol.file_uploader("WAV-Datei", type=["wav"], key=f"up_{i}",
                                     label_visibility="collapsed")
        if upload is not None:        # hochgeladene WAV mittig daneben anhoeren
            playcol.audio(upload)
        source = upload if upload is not None else audio   # Upload hat Vorrang
        if source is None:
            continue

        with st.spinner("main.c ..."):
            samples, t, x, xhat, err, coeffs = process(source.getvalue())
        results.append((f"#{i + 1}", coeffs))

        # Diagnose: bis wann ist ueberhaupt Signal da? (beantwortet die Frage
        # "nimmt die Aufnahme nach ein paar Sekunden nichts mehr auf?")
        dur = len(samples) / L.FS
        w_env = max(1, int(0.03 * L.FS))
        env = np.sqrt(np.convolve(x ** 2, np.ones(w_env) / w_env, mode="same"))
        thr = 0.05 * env.max() if env.max() > 0 else 0.0
        active = t[env > thr]
        last = float(active[-1]) if len(active) else 0.0
        st.caption(f"{dur:.1f} s @ {L.FS} Hz · Signal aktiv bis {last:.1f} s")

        # kurzer Roh-Ueberblick: womit arbeiten wir?
        st.pyplot(P.figure_overview(t, x, L.FS, figsize=(5, 3.4)))

        st.pyplot(P.figure_spectrum(coeffs, L.FS, figsize=SMALL))
        st.pyplot(P.figure_coeffs(t, coeffs, figsize=SMALL, legend=False))
        st.pyplot(P.figure_error(t, err, L.FS, figsize=SMALL))
        st.pyplot(P.figure_signal(t, x, xhat, figsize=SMALL))

        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        wbuf = io.BytesIO()
        wavfile.write(wbuf, L.FS, pcm16)
        with st.expander("Download"):
            st.download_button("WAV", wbuf.getvalue(),
                               f"aufnahme{i + 1}_16k.wav", "audio/wav", key=f"w{i}")
            st.download_button("RAW int16", pcm16.astype("<i2").tobytes(),
                               f"aufnahme{i + 1}_16k.raw",
                               "application/octet-stream", key=f"r{i}")

# Direkter Vergleich: Spektren uebereinander
if len(results) >= 2:
    st.subheader("Direkter Vergleich: LPC-Spektren übereinander")
    st.pyplot(P.figure_spectra_overlay(results, L.FS))
    st.caption(
        "Gleicher Vokal, andere Tonhöhe -> Kurven fast deckungsgleich.   "
        "Anderer Vokal -> die Formant-Spitzen verschieben sich deutlich."
    )

# --- Footer ---
st.markdown("---")
st.caption("Made with ❤️ for AdaSys (by Falk and Claude 4.8 Opus)")