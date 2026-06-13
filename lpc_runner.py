import shutil
import subprocess
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

FS = 16000
MU = 0.02          # Lernrate des LMS-Filters

HERE = Path(__file__).resolve().parent
MAIN_C = HERE / "main.c"
BIN = HERE / "lpc"
SAMPLES = HERE / "samples.txt"
OUTPUT = HERE / "output.csv"


def _find_cc():
    for c in ("cc", "gcc", "clang"):
        if shutil.which(c):
            return c
    raise RuntimeError("Kein C-Compiler gefunden (cc/gcc/clang).")


def ensure_binary():
    """Kompiliere main.c bei Bedarf (wenn Binary fehlt oder aelter ist)."""
    if (not BIN.exists()) or MAIN_C.stat().st_mtime > BIN.stat().st_mtime:
        res = subprocess.run(
            [_find_cc(), "-O2", str(MAIN_C), "-o", str(BIN), "-lm"],
            capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError("Kompilieren von main.c fehlgeschlagen:\n" + res.stderr)
    return BIN


def to_mono_16k(data, sr):
    """-> float64, mono, normiert auf [-1, 1], 16 kHz."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim > 1:                      # Stereo -> Mono
        data = data.mean(axis=1)
    peak = float(np.max(np.abs(data)))
    if peak > 0:
        data = data / peak
    if int(sr) != FS:                      # auf 16 kHz resamplen
        g = gcd(int(sr), FS)
        data = resample_poly(data, FS // g, int(sr) // g)
    return data


def read_wav(path_or_buf):
    """WAV (Pfad oder Datei-aehnliches Objekt, z.B. von st.audio_input) -> Samples."""
    from scipy.io import wavfile
    sr, data = wavfile.read(path_or_buf)
    if np.issubdtype(data.dtype, np.integer):
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    return to_mono_16k(data, sr)


def run_lpc(samples, delays, mu=MU):
    """Schreibe Samples, rufe main.c, lies Ergebnis zurueck.

    delays: Liste der Tap-Verzoegerungen (z.B. aus make_delays).
    Returns: t [s], x, x_hat, error, coeffs (shape [len(delays), N]).
    """
    ensure_binary()
    np.savetxt(SAMPLES, np.asarray(samples, dtype=np.float64), fmt="%.6f")
    args = [str(BIN), str(SAMPLES), str(OUTPUT), str(mu)] + [str(d) for d in delays]
    res = subprocess.run(args, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError("main.c-Lauf fehlgeschlagen:\n" + res.stderr)
    raw = np.loadtxt(OUTPUT, delimiter=",", skiprows=1)
    t = raw[:, 0] / FS
    x, xhat, err = raw[:, 1], raw[:, 2], raw[:, 3]
    p = len(delays)
    coeffs = raw[:, 4:4 + p].T
    return t, x, xhat, err, coeffs
