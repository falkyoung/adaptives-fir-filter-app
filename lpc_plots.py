"""Matplotlib-Plots fuer die LPC-Analyse. Jede Funktion gibt eine Figure zurueck.

Konventionen:
- Legenden liegen ausserhalb der Kurven (rechts neben dem Plot).
- layout="constrained" reserviert dafuer automatisch Platz.
- figsize ist einstellbar (schmale Streamlit-Spalten).
"""
import numpy as np
import matplotlib.pyplot as plt


def _stride(n, target=3000):
    """Schrittweite zum Ausduennen langer Signale (nur fuers Zeichnen)."""
    return max(1, n // target)


def _legend_right(ax, ncol=1):
    """Legende rechts neben den Plot legen, nie ueber die Kurven."""
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8,
              ncol=ncol, borderaxespad=0.0)


def figure_overview(t, x, fs=16000, figsize=(10, 3.6)):
    """Kurzer Roh-Ueberblick ueber das Eingangssignal:
    oben Amplitude (Zeitbereich), unten Frequenzspektrum (Welch)."""
    from scipy.signal import welch
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, layout="constrained")

    s = _stride(len(t))
    ax1.plot(t[::s], x[::s], lw=0.6, color="C0")
    ax1.set_xlabel("Zeit [s]")
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Amplitude (gesamtes Signal)")
    ax1.grid(alpha=0.3)

    f, pxx = welch(x, fs=fs, nperseg=min(2048, len(x)))
    ax2.plot(f, 10 * np.log10(pxx + 1e-12), lw=0.8, color="C4")
    ax2.set_xlim(0, fs / 2)
    ax2.set_xlabel("Frequenz [Hz]")
    ax2.set_ylabel("Leistung [dB]")
    ax2.set_title("Frequenzspektrum des Signals")
    ax2.grid(alpha=0.3)
    return fig


def figure_signal(t, x, xhat, fs=16000, win_ms=35.0, figsize=(10, 3)):
    """Zeitbereich als ZOOM auf ein eingeschwungenes, energiereiches Fenster.
    Zeigt zusaetzlich die (kleine) Differenz e(t) = x - x_dach verstaerkt,
    damit sichtbar wird, dass x und x_dach eben NICHT exakt gleich sind."""
    n = len(x)
    w = max(2, int(win_ms / 1000.0 * fs))
    we = max(1, int(0.02 * fs))
    err = x - xhat
    sig = np.convolve(x ** 2, np.ones(we) / we, mode="same")    # Signal-Energie
    ers = np.convolve(err ** 2, np.ones(we) / we, mode="same")  # Fehler-Energie
    # Stelle mit Signal UND kleinstem relativen Fehler (Filter eingeschwungen)
    good = sig > 0.2 * sig.max() if sig.max() > 0 else np.ones(n, bool)
    ratio = np.where(good, ers / (sig + 1e-12), np.inf)
    c = int(np.argmin(ratio)) if np.isfinite(ratio).any() else int(np.argmax(sig))
    a = min(max(0, c - w // 2), max(0, n - w))
    b = min(n, a + w)

    tw = (t[a:b] - t[a]) * 1000.0
    ew = err[a:b]
    fac = max(1, int(round(0.4 * (np.max(np.abs(x[a:b])) + 1e-9)
                           / (np.max(np.abs(ew)) + 1e-9))))

    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.axhline(0, color="0.7", lw=0.5)
    ax.plot(tw, x[a:b], lw=1.1, label="x(t)")
    ax.plot(tw, xhat[a:b], lw=1.1, alpha=0.85, label="x_dach(t)")
    ax.plot(tw, ew * fac, lw=0.9, color="C3", label=f"e(t) x{fac}")
    ax.set_xlabel("Zeit [ms] (Ausschnitt)")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Zeitbereich-Ausschnitt {win_ms:.0f} ms bei t={t[a]:.2f} s")
    _legend_right(ax)
    ax.grid(alpha=0.3)
    return fig


def figure_error(t, err, fs=16000, figsize=(10, 3)):
    s = _stride(len(t))
    win = max(1, int(0.02 * fs))                      # 20-ms-Fenster
    rms = np.sqrt(np.convolve(err ** 2, np.ones(win) / win, mode="same"))
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(t[::s], err[::s], lw=0.5, color="0.6", label="Fehler e(t)")
    ax.plot(t[::s], rms[::s], lw=1.6, color="C3", label="RMS (20 ms)")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("e(t) = x - x_dach")
    ax.set_title("Fehler e(t) wird kleiner  (Kosten E(t) = 0.5 e^2)")
    _legend_right(ax)
    ax.grid(alpha=0.3)
    return fig


def figure_coeffs(t, coeffs, figsize=(10, 4), legend=True):
    s = _stride(coeffs.shape[1])
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for k in range(coeffs.shape[0]):
        ax.plot(t[::s], coeffs[k, ::s], lw=1.0, label=f"a{k}")
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Koeffizientenwert a_k")
    ax.set_title("Koeffizienten ueber die Zeit (Start bei 0)")
    if legend:
        _legend_right(ax, ncol=2)
    ax.grid(alpha=0.3)
    return fig


def _lpc_response(coeffs_col, fs, n=1024):
    from scipy.signal import freqz
    w, h = freqz([1.0], np.concatenate(([1.0], -coeffs_col)), worN=n, fs=fs)
    return w, 20 * np.log10(np.abs(h) + 1e-9)


def figure_spectrum(coeffs, fs=16000, figsize=(10, 3)):
    """LPC-Spektrum aus den End-Koeffizienten (Vokaltrakt/Formanten)."""
    w, mag = _lpc_response(coeffs[:, -1], fs)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.plot(w, mag, color="C2")
    ax.set_xlabel("Frequenz [Hz]")
    ax.set_ylabel("Betrag [dB]")
    ax.set_title("LPC-Spektrum (Formanten)")
    ax.grid(alpha=0.3)
    return fig


def figure_spectra_overlay(items, fs=16000, figsize=(10, 4)):
    """items: Liste von (label, coeffs). Spektren uebereinander zum Vergleich."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for label, coeffs in items:
        w, mag = _lpc_response(coeffs[:, -1], fs)
        ax.plot(w, mag, lw=2, label=label)
    ax.set_xlabel("Frequenz [Hz]")
    ax.set_ylabel("Betrag [dB]")
    ax.set_title("Vergleich der LPC-Spektren (Formanten)")
    _legend_right(ax)
    ax.grid(alpha=0.3)
    return fig


def figure_lpcgram(t, coeffs, fs=16000, n_freq=256, n_frames=240):
    """LPC-Spektrum H(z)=1/A(z) ueber die Zeit als Heatmap."""
    N = coeffs.shape[1]
    idx = np.linspace(0, N - 1, min(n_frames, N)).astype(int)
    mags = np.empty((n_freq, len(idx)))
    for j, i in enumerate(idx):
        _, mags[:, j] = _lpc_response(coeffs[:, i], fs, n=n_freq)
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    im = ax.imshow(mags, origin="lower", aspect="auto", cmap="magma",
                   extent=[t[0], t[-1], 0, fs / 2])
    ax.set_xlabel("Zeit [s]")
    ax.set_ylabel("Frequenz [Hz]")
    ax.set_title("LPC-Spektrum ueber die Zeit  —  Vokal aendert die Formanten, "
                 "Tonhoehe nicht")
    fig.colorbar(im, ax=ax, label="Betrag [dB]")
    return fig
