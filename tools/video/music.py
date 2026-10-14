"""Original ambient background music for the explainer video, synthesised with NumPy.

Soft pads, a sine bass and a sparse bell-like arpeggio at 72 BPM over a slow I-vi-IV-V
progression, with a convolution reverb. Everything is generated here (no samples), so the track
carries no third-party licence.

    python tools/video/music.py OUT.wav SECONDS
"""
from __future__ import annotations

import sys
import wave

import numpy as np
from scipy.signal import oaconvolve

SR = 48000
BPM = 72.0
BEAT = 60.0 / BPM
CHORD_BEATS = 8

# two progressions in D major, alternating
PROG_A = [[50, 57, 61, 64, 66], [47, 54, 57, 61, 62], [43, 50, 54, 57, 62], [45, 52, 57, 59, 64]]   # Dmaj9 Bm7 Gmaj7 Asus2
PROG_B = [[52, 59, 62, 66, 67], [43, 50, 55, 59, 66], [42, 54, 57, 61, 62], [45, 52, 57, 62, 64]]   # Em7 Gmaj7 D/F# Asus4


def hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def env_adsr(n, a, r):
    e = np.ones(n, np.float32)
    na, nr = int(a * SR), int(r * SR)
    e[:na] = np.linspace(0, 1, na) ** 2
    e[-nr:] *= np.linspace(1, 0, nr) ** 1.5
    return e


def pad_note(m, dur, rng):
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for det in (-0.07, 0.0, 0.06):                       # three detuned voices (cents/100 of a semitone)
        f = hz(m + det)
        ph = rng.uniform(0, 2 * np.pi)
        for k, a in ((1, 1.0), (2, 0.12), (3, 0.035)):
            out += a * np.sin(2 * np.pi * f * k * t + ph * k)
    trem = 1 + 0.08 * np.sin(2 * np.pi * 0.17 * t + rng.uniform(0, 6))
    return out * trem * env_adsr(n, 2.2, 2.8) / 3


def bell(m, rng, dur=2.6):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.9) + 0.22 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.35)
         + 0.06 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / 0.18))
    att = np.minimum(1, t / 0.006)
    return (s * att).astype(np.float32)


def bass_note(m, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.18 * np.sin(2 * np.pi * 2 * f * t)
    return (s * env_adsr(n, 0.4, 1.8)).astype(np.float32)


def reverb_ir(seconds=3.2, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / 0.75)
    # soften the tail: simple low-pass by cumulative averaging
    k = 24
    ir = np.convolve(ir, np.ones(k) / k, mode="same")
    ir[: int(0.012 * SR)] = 0                              # pre-delay
    return (ir / np.sqrt(np.sum(ir ** 2))).astype(np.float32)


def compose(seconds, seed=4):
    rng = np.random.default_rng(seed)
    n = int((seconds + 6) * SR)
    L = np.zeros(n, np.float32)
    R = np.zeros(n, np.float32)

    def add(sig, t0, gain, pan):
        i = int(t0 * SR)
        j = min(n, i + len(sig))
        if i >= n:
            return
        L[i:j] += sig[: j - i] * gain * np.cos((pan + 1) * np.pi / 4)
        R[i:j] += sig[: j - i] * gain * np.sin((pan + 1) * np.pi / 4)

    cl = CHORD_BEATS * BEAT
    t, idx = 0.0, 0
    while t < seconds:
        prog = PROG_A if (idx // 4) % 2 == 0 else PROG_B
        chord = prog[idx % 4]
        # pads (overlap into the next chord for smooth changes)
        for k, m in enumerate(chord[1:]):
            add(pad_note(m - (12 if k >= 2 else 0), cl + 2.5, rng), t, 0.045, -0.5 + 0.33 * k)
        add(bass_note(chord[0] - 12, cl + 1.0), t, 0.085, 0.0)
        add(pad_note(chord[0], cl + 2.5, rng), t, 0.03, 0.0)
        # arpeggio: sparse eighth notes from the chord, two octaves up; quieter at start and end
        density = 0.0 if t < 6 else (0.35 if t > seconds - 35 else 0.55)
        tones = sorted(set([m + 12 for m in chord[1:]] + [chord[1] + 24, chord[2] + 24]))
        for e in range(CHORD_BEATS * 2):
            if rng.random() < density:
                m = tones[(e * 3 + idx) % len(tones)] if rng.random() < 0.7 else rng.choice(tones)
                add(bell(m, rng), t + e * BEAT / 2 + rng.normal(0, 0.006), 0.06 * rng.uniform(0.6, 1.0),
                    rng.uniform(-0.6, 0.6))
        t += cl
        idx += 1

    irL, irR = reverb_ir(seed=1), reverb_ir(seed=2)
    wetL = oaconvolve(L, irL)[:n]
    wetR = oaconvolve(R, irR)[:n]
    L = 0.72 * L + 0.42 * wetL
    R = 0.72 * R + 0.42 * wetR
    out = np.stack([L, R], 1)[: int(seconds * SR)]
    # fades
    fi, fo = int(3.0 * SR), int(7.0 * SR)
    out[:fi] *= np.linspace(0, 1, fi)[:, None]
    out[-fo:] *= (np.linspace(1, 0, fo) ** 1.6)[:, None]
    out /= np.abs(out).max() / 0.7
    return out


def write_wav(path, x):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    write_wav(sys.argv[1], compose(float(sys.argv[2])))
