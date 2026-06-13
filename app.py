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
st.title("Adaptives FIR-Filter (LPC)")
st.write(
    "Nimm **bis zu 3 kurze Aufnahmen** auf (je ~3-5 s, ein gehaltener Vokal) und "
    "vergleiche sie direkt nebeneinander. Z.B. **i / a / o** (verschiedene Vokale) "
    "oder **3x a in verschiedenen Tonhöhen**."
)
with st.expander("Was ist LPC?"):
    st.markdown(
        "**LPC = Linear Predictive Coding** (lineare Vorhersage). Das aktuelle "
        "Sample wird aus ausgewählten vergangenen Samples geschätzt - per "
        "FIR-Filter. Welche Vergangenheitswerte x(t-d) benutzt werden, wählst "
        "du je Spalte selbst (1,2,3,… = Standard; 4,8,12,… = jeder 4.; "
        "2,3,5,7,… = Primzahlen):\n\n"
        "- **Phase 1 (Schätzung):**  x_dach(t) = Summe  a_k * x(t-d_k)\n"
        "- **Fehler:**  e(t) = x(t) - x_dach(t)  ,   Kosten  E(t) = 0.5 * e(t)^2\n"
        "- **Phase 2 (Anpassung):**  Koeffizienten a_k per Gradientenabstieg "
        "langsam anpassen, bis E(t) klein wird (ideal 0).\n\n"
        "Die eingependelten Koeffizienten beschreiben den **Vokaltrakt**. Ihr "
        "Spektrum H(z) = 1/A(z) ist die **Spektralschätzung** - die Spitzen sind "
        "die **Formanten**. Vokale unterscheiden sich in den Formanten, die "
        "Tonhöhe (Grundfrequenz) kaum."
    )


MAX_DELAY = 100 # max Taps


def choose_taps(i):
    """Kompaktes Dropdown (Popover) mit Häkchen-Raster zur Tap-Auswahl.
    Gibt die sortierte Liste der gewählten Verzögerungen d zurück.
    Buttons 'Alle'/'Keine' für schnelles (Ab-)Wählen in der Präsentation."""
    def key(d):
        return f"tap_{i}_{d}"

    for d in range(1, MAX_DELAY + 1):          # Default: 1..16 angehakt
        st.session_state.setdefault(key(d), d <= 16)

    chosen = [d for d in range(1, MAX_DELAY + 1) if st.session_state[key(d)]]
    label = f"Taps wählen — {len(chosen)} aktiv" if chosen else "Taps wählen"

    with st.popover(label, use_container_width=True):
        b1, b2 = st.columns(2)
        if b1.button("Alle", key=f"all_{i}", use_container_width=True):
            for d in range(1, MAX_DELAY + 1):
                st.session_state[key(d)] = True
        if b2.button("Keine", key=f"none_{i}", use_container_width=True):
            for d in range(1, MAX_DELAY + 1):
                st.session_state[key(d)] = False

        grid = st.columns(8)                   # 8 Häkchen pro Reihe
        for d in range(1, MAX_DELAY + 1):
            grid[(d - 1) % 8].checkbox(str(d), key=key(d))

    return [d for d in range(1, MAX_DELAY + 1) if st.session_state[key(d)]]


@st.cache_data(show_spinner=False)
def process(wav_bytes, delays):
    """Audio-Bytes -> main.c -> Arrays. Gecacht pro (Aufnahme, Tap-Auswahl),
    damit unveränderte Aufnahmen bei gleicher Auswahl nicht neu gerechnet
    werden. delays = Tupel der gewaehlten Vergangenheitswerte."""
    samples = L.read_wav(io.BytesIO(wav_bytes))
    t, x, xhat, err, coeffs = L.run_lpc(samples, list(delays))
    return samples, t, x, xhat, err, coeffs


SMALL = (5, 2.4)   # kompakte Plotgroesse fuer schmale Spalten

st.caption(
    "⚠️ Hinweis: Geräuschunterdrückung von Browser, System und Mikrofon muss "
    "deaktiviert sein, sonst wird die Aufnahme nach 2-3 s unterdrückt. "
    "Alternativ extern aufnehmen und je Spalte als WAV hochladen."
)

data = [None, None, None]   # je Spalte: (samples, t, x, xhat, err, coeffs)
results = []                # (label, coeffs) fuer das Overlay

cols = st.columns(3)
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

        delays = tuple(choose_taps(i))
        if source is None:
            continue
        if not delays:
            st.warning("Mindestens einen Vergangenheitswert auswählen.")
            continue

        with st.spinner("main.c ..."):
            samples, t, x, xhat, err, coeffs = process(source.getvalue(), delays)
        data[i] = (samples, t, x, xhat, err, coeffs, delays)
        results.append((f"#{i + 1}", coeffs, list(delays)))

        dur = len(samples) / L.FS
        w_env = max(1, int(0.03 * L.FS))
        env = np.sqrt(np.convolve(x ** 2, np.ones(w_env) / w_env, mode="same"))
        thr = 0.05 * env.max() if env.max() > 0 else 0.0
        active = t[env > thr]
        last = float(active[-1]) if len(active) else 0.0
        st.caption(f"{dur:.1f} s @ {L.FS} Hz · Signal aktiv bis {last:.1f} s")

present = [d for d in data if d is not None]
ylim_amp = ylim_err = ylim_coeff = ylim_spec = ylim_db = None
if present:
    amp = max(float(np.max(np.abs(d[2]))) for d in present)        # x
    emax = max(float(np.max(np.abs(d[4]))) for d in present)       # err
    cmax = max(float(np.max(np.abs(d[5]))) for d in present)       # coeffs
    ylim_amp = (-1.05 * amp, 1.05 * amp)
    ylim_err = (-1.05 * emax, 1.05 * emax)
    ylim_coeff = (-1.05 * cmax, 1.05 * cmax)
    ylim_spec = P.spectrum_ylim([(d[5], d[6]) for d in present], L.FS)
    ylim_db = P.welch_ylim([d[2] for d in present], L.FS)

# --- Reihe 1: Überblick + LPC-Spektrum (gleiche Skalen) ---
for i, col in enumerate(st.columns(3)):
    with col:
        if data[i] is None:
            continue
        samples, t, x, xhat, err, coeffs, delays = data[i]
        st.pyplot(P.figure_overview(t, x, L.FS, figsize=(5, 3.4),
                                    ylim=ylim_amp, ylim_db=ylim_db))
        st.pyplot(P.figure_spectrum(coeffs, list(delays), L.FS, figsize=SMALL,
                                    ylim=ylim_spec))

if len(results) >= 2:
    st.pyplot(P.figure_spectra_overlay(results, L.FS, figsize=(15, 2.4)))

# --- Reihe 2: Koeffizienten, Fehler, Zeitbereich (gleiche Skalen) ---
for i, col in enumerate(st.columns(3)):
    with col:
        if data[i] is None:
            continue
        samples, t, x, xhat, err, coeffs, delays = data[i]
        st.pyplot(P.figure_coeffs(t, coeffs, list(delays), figsize=(5, 3.6),
                                  legend=True, ylim=ylim_coeff))
        st.pyplot(P.figure_error(t, err, L.FS, figsize=SMALL, ylim=ylim_err))
        st.pyplot(P.figure_signal(t, x, xhat, figsize=SMALL, ylim=ylim_amp))

        pcm16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        wbuf = io.BytesIO()
        wavfile.write(wbuf, L.FS, pcm16)
        with st.expander("Download"):
            st.download_button("WAV", wbuf.getvalue(),
                               f"aufnahme{i + 1}_16k.wav", "audio/wav", key=f"w{i}")
            st.download_button("RAW int16", pcm16.astype("<i2").tobytes(),
                               f"aufnahme{i + 1}_16k.raw",
                               "application/octet-stream", key=f"r{i}")

st.markdown("---")
st.caption("Made with ❤️ for AdaSys (by Falk and Claude 4.8 Opus)")