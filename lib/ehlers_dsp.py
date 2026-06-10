"""
Ehlers DSP Indicators — Custom implementations not available in pandas-ta.

References:
    - John Ehlers, "Cybernetic Analysis for Stocks and Futures" (2004)
    - John Ehlers, "Cycle Analytics for Traders" (2013)
    - John Ehlers, "Rocket Science for Traders" (2001)
    - MESA (Maximum Entropy Spectral Analysis)

Implemented:
    - super_smoother()          — 2-pole Butterworth low-pass filter
    - roofing_filter()          — high-pass + super smoother (bandpass)
    - hilbert_transform_dominant_cycle() — instantaneous period via Hilbert Transform
    - mama_fama()               — MESA Adaptive Moving Average + Following AMA
    - instantaneous_trendline() — Ehlers iTrend using dominant cycle
    - cyber_cycle()             — Cycle indicator using smooth + Hilbert
    - adaptive_rsi()            — RSI with cycle-adaptive lookback
"""

import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════════

def super_smoother(series: pd.Series, period: int = 10) -> pd.Series:
    """
    Ehlers Super Smoother Filter — 2-pole Butterworth with minimal lag.

    Unlike SMA/EMA, it eliminates Nyquist frequency noise without
    the lag penalty of higher-order moving averages. This is the
    fundamental building block for most Ehlers indicators.

    Args:
        series: Input price or indicator series
        period: Cutoff period (bars). Frequencies shorter than this are attenuated.

    Returns:
        Smoothed pd.Series
    """
    a = np.exp(-np.sqrt(2) * np.pi / period)
    b = 2 * a * np.cos(np.sqrt(2) * np.pi / period)

    c2 = b
    c3 = -(a ** 2)
    c1 = 1 - c2 - c3

    n = len(series)
    result = np.zeros(n)
    vals = series.values.astype(float)

    for i in range(2, n):
        result[i] = (
            c1 * (vals[i] + vals[i - 1]) / 2
            + c2 * result[i - 1]
            + c3 * result[i - 2]
        )

    return pd.Series(result, index=series.index, name=f"super_smoother_{period}")


def roofing_filter(
    close: pd.Series, hp_period: int = 48, lp_period: int = 10
) -> pd.Series:
    """
    Ehlers Roofing Filter — high-pass + super smoother.

    Removes both low-frequency trend and high-frequency noise,
    isolating the tradeable cycle component. The "roof" clips
    frequencies above and below the band of interest.

    Args:
        close: Price series
        hp_period: High-pass cutoff (removes cycles longer than this)
        lp_period: Low-pass cutoff via super smoother

    Returns:
        Bandpass-filtered series (cycle component)
    """
    alpha1 = (
        np.cos(0.707 * 2 * np.pi / hp_period)
        + np.sin(0.707 * 2 * np.pi / hp_period)
        - 1
    ) / np.cos(0.707 * 2 * np.pi / hp_period)

    n = len(close)
    hp = np.zeros(n)
    vals = close.values.astype(float)

    for i in range(2, n):
        hp[i] = (
            (1 - alpha1 / 2) ** 2 * (vals[i] - 2 * vals[i - 1] + vals[i - 2])
            + 2 * (1 - alpha1) * hp[i - 1]
            - (1 - alpha1) ** 2 * hp[i - 2]
        )

    hp_series = pd.Series(hp, index=close.index)
    return super_smoother(hp_series, period=lp_period)


# ═══════════════════════════════════════════════════════════════════════
# Hilbert Transform — Dominant Cycle Detection
# ═══════════════════════════════════════════════════════════════════════

def _hilbert_transform_internals(close: pd.Series):
    """
    Core Hilbert Transform computation shared by dominant cycle and MAMA.

    Returns arrays: (smooth, detrender, period, phase, I1, Q1)
    all of length len(close).

    Implementation follows Ehlers' "Cycle Analytics for Traders" Ch. 9.
    """
    n = len(close)
    vals = close.values.astype(float)

    # Working arrays
    smooth = np.zeros(n)
    detrender = np.zeros(n)
    I1 = np.zeros(n)
    Q1 = np.zeros(n)
    jI = np.zeros(n)
    jQ = np.zeros(n)
    I2 = np.zeros(n)
    Q2 = np.zeros(n)
    Re = np.zeros(n)
    Im = np.zeros(n)
    period = np.full(n, 6.0)  # Initialize to minimum period
    smooth_period = np.full(n, 6.0)
    phase = np.zeros(n)

    # Ehlers uses a 4-bar weighted MA as the smoother
    for i in range(6, n):
        # 4-bar WMA
        smooth[i] = (
            4 * vals[i]
            + 3 * vals[i - 1]
            + 2 * vals[i - 2]
            + vals[i - 3]
        ) / 10.0

        # Detrend with a 1-period lag Hilbert Transform
        detrender[i] = (
            0.0962 * smooth[i]
            + 0.5769 * smooth[i - 2]
            - 0.5769 * smooth[i - 4]
            - 0.0962 * smooth[i - 6]
        ) * (0.075 * period[i - 1] + 0.54)

        # Compute InPhase and Quadrature components
        Q1[i] = (
            0.0962 * detrender[i]
            + 0.5769 * detrender[i - 2]
            - 0.5769 * detrender[i - 4]
            - 0.0962 * detrender[i - 6]
        ) * (0.075 * period[i - 1] + 0.54)

        I1[i] = detrender[i - 3]

        # Advance the phase of I1 and Q1 by 90 degrees
        jI[i] = (
            0.0962 * I1[i]
            + 0.5769 * I1[i - 2]
            - 0.5769 * I1[i - 4]
            - 0.0962 * I1[i - 6]
        ) * (0.075 * period[i - 1] + 0.54)

        jQ[i] = (
            0.0962 * Q1[i]
            + 0.5769 * Q1[i - 2]
            - 0.5769 * Q1[i - 4]
            - 0.0962 * Q1[i - 6]
        ) * (0.075 * period[i - 1] + 0.54)

        # Phasor addition for 3-bar averaging
        I2[i] = I1[i] - jQ[i]
        Q2[i] = Q1[i] + jI[i]

        # Smooth the I and Q components
        I2[i] = 0.2 * I2[i] + 0.8 * I2[i - 1]
        Q2[i] = 0.2 * Q2[i] + 0.8 * Q2[i - 1]

        # Homodyne discriminator
        Re[i] = I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1]
        Im[i] = I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1]

        Re[i] = 0.2 * Re[i] + 0.8 * Re[i - 1]
        Im[i] = 0.2 * Im[i] + 0.8 * Im[i - 1]

        if Im[i] != 0 and Re[i] != 0:
            period[i] = 2 * np.pi / np.arctan(Im[i] / Re[i])

        # Clamp period
        if period[i] > 1.5 * period[i - 1]:
            period[i] = 1.5 * period[i - 1]
        if period[i] < 0.67 * period[i - 1]:
            period[i] = 0.67 * period[i - 1]
        if period[i] < 6:
            period[i] = 6
        if period[i] > 50:
            period[i] = 50

        # Smooth the period
        period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
        smooth_period[i] = 0.33 * period[i] + 0.67 * smooth_period[i - 1]

        # Compute phase
        if I1[i] != 0:
            phase[i] = np.degrees(np.arctan(Q1[i] / I1[i]))

    return smooth, detrender, smooth_period, phase, I1, Q1


def hilbert_transform_dominant_cycle(
    close: pd.Series, min_period: int = 6, max_period: int = 50
) -> pd.Series:
    """
    Compute the dominant cycle period using the Hilbert Transform.

    This is the core cycle measurement tool in MESA. It detects
    the instantaneous frequency of the dominant cycle in the data,
    adapting in real-time as market conditions change.

    Args:
        close: Price series (typically Close or HL2)
        min_period: Minimum allowed cycle period (default 6)
        max_period: Maximum allowed cycle period (default 50)

    Returns:
        pd.Series of dominant cycle period (in bars)
    """
    _, _, smooth_period, _, _, _ = _hilbert_transform_internals(close)

    # Apply final clamping
    smooth_period = np.clip(smooth_period, min_period, max_period)

    return pd.Series(
        smooth_period, index=close.index, name="dominant_cycle"
    )


# ═══════════════════════════════════════════════════════════════════════
# MAMA / FAMA — MESA Adaptive Moving Average
# ═══════════════════════════════════════════════════════════════════════

def mama_fama(
    close: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05
) -> tuple[pd.Series, pd.Series]:
    """
    MESA Adaptive Moving Average (MAMA) and Following Adaptive Moving Average (FAMA).

    MAMA adapts its smoothing factor based on the rate of change of phase
    as measured by the Hilbert Transform. When price is cycling rapidly,
    MAMA tracks closely (fast_limit). When price trends, MAMA smooths
    aggressively (slow_limit). FAMA is a further-smoothed version that
    creates a natural crossover signal.

    Trading logic:
        - MAMA crosses above FAMA → Buy (cycle turning up)
        - MAMA crosses below FAMA → Sell (cycle turning down)

    Args:
        close: Price series
        fast_limit: Maximum alpha (fast tracking). Default 0.5.
        slow_limit: Minimum alpha (slow tracking). Default 0.05.

    Returns:
        (mama, fama) — tuple of pd.Series
    """
    smooth, _, smooth_period, phase, I1, Q1 = _hilbert_transform_internals(close)

    n = len(close)
    mama = np.zeros(n)
    fama = np.zeros(n)

    # Initialize with price
    vals = close.values.astype(float)
    mama[0] = vals[0]
    fama[0] = vals[0]

    for i in range(1, n):
        # Compute delta phase
        delta_phase = phase[i] - phase[i - 1]
        if delta_phase < 1:
            delta_phase = 1

        # Compute alpha from delta phase
        alpha = fast_limit / delta_phase
        if alpha < slow_limit:
            alpha = slow_limit
        if alpha > fast_limit:
            alpha = fast_limit

        # MAMA
        mama[i] = alpha * vals[i] + (1 - alpha) * mama[i - 1]

        # FAMA (half the rate of MAMA)
        fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * fama[i - 1]

    mama_series = pd.Series(mama, index=close.index, name="MAMA")
    fama_series = pd.Series(fama, index=close.index, name="FAMA")

    return mama_series, fama_series


# ═══════════════════════════════════════════════════════════════════════
# Instantaneous Trendline
# ═══════════════════════════════════════════════════════════════════════

def instantaneous_trendline(close: pd.Series) -> pd.Series:
    """
    Ehlers Instantaneous Trendline (iTrend).

    Uses the dominant cycle period to create an adaptive trendline
    that responds to the market's own rhythm rather than a fixed lookback.

    The iTrend removes the dominant cycle component, leaving the trend.
    When price is above iTrend and rising → uptrend.
    When price is below iTrend and falling → downtrend.
    """
    dc = hilbert_transform_dominant_cycle(close)
    n = len(close)
    vals = close.values.astype(float)
    itrend = np.zeros(n)

    for i in range(2, n):
        # Use dominant cycle as adaptive period for the trendline
        p = max(int(dc.iloc[i]), 2)
        alpha = 2.0 / (p + 1)
        itrend[i] = (
            (alpha - alpha**2 / 4) * vals[i]
            + 0.5 * alpha**2 * vals[i - 1]
            - (alpha - 0.75 * alpha**2) * vals[i - 2]
            + 2 * (1 - alpha) * itrend[i - 1]
            - (1 - alpha) ** 2 * itrend[i - 2]
        )

    return pd.Series(itrend, index=close.index, name="iTrend")


# ═══════════════════════════════════════════════════════════════════════
# Cyber Cycle
# ═══════════════════════════════════════════════════════════════════════

def cyber_cycle(close: pd.Series, alpha: float = 0.07) -> pd.Series:
    """
    Ehlers Cyber Cycle indicator.

    A bandpass filter tuned to the dominant cycle frequency.
    Oscillates between +1 and -1 (approximately).

    Use as a cycle oscillator: positive → cycle upswing, negative → downswing.
    Zero-crossings are potential entry/exit points.

    Args:
        close: Price series
        alpha: Smoothing factor (default 0.07, from Ehlers)

    Returns:
        pd.Series of cyber cycle values
    """
    n = len(close)
    vals = close.values.astype(float)
    smooth = np.zeros(n)
    cycle = np.zeros(n)

    for i in range(3, n):
        smooth[i] = (vals[i] + 2 * vals[i - 1] + 2 * vals[i - 2] + vals[i - 3]) / 6.0

    for i in range(6, n):
        cycle[i] = (
            (1 - 0.5 * alpha) ** 2 * (smooth[i] - 2 * smooth[i - 1] + smooth[i - 2])
            + 2 * (1 - alpha) * cycle[i - 1]
            - (1 - alpha) ** 2 * cycle[i - 2]
        )

    return pd.Series(cycle, index=close.index, name="cyber_cycle")


# ═══════════════════════════════════════════════════════════════════════
# Adaptive RSI
# ═══════════════════════════════════════════════════════════════════════

def adaptive_rsi(close: pd.Series, min_period: int = 6, max_period: int = 50) -> pd.Series:
    """
    RSI with cycle-adaptive lookback period.

    Instead of a fixed 14-bar RSI, the lookback adapts to half the
    dominant cycle period. This means the RSI naturally adjusts to
    whatever rhythm the market is currently exhibiting.

    Args:
        close: Price series
        min_period: Minimum RSI lookback
        max_period: Maximum RSI lookback

    Returns:
        pd.Series of adaptive RSI values (0-100)
    """
    dc = hilbert_transform_dominant_cycle(close, min_period=min_period, max_period=max_period)
    n = len(close)
    vals = close.values.astype(float)
    rsi = np.full(n, 50.0)

    for i in range(1, n):
        # Use half the dominant cycle as RSI period (Ehlers recommendation)
        lookback = max(int(dc.iloc[i] / 2), 2)
        start = max(0, i - lookback)

        ups = 0.0
        downs = 0.0
        for j in range(start + 1, i + 1):
            change = vals[j] - vals[j - 1]
            if change > 0:
                ups += change
            else:
                downs += abs(change)

        if ups + downs > 0:
            rsi[i] = 100 * ups / (ups + downs)
        else:
            rsi[i] = 50.0

    # Clamp to [0, 100] to avoid floating-point edge cases
    rsi = np.clip(rsi, 0.0, 100.0)
    return pd.Series(rsi, index=close.index, name="adaptive_rsi")


# ═══════════════════════════════════════════════════════════════════════
# Convenience: compute all Ehlers indicators at once
# ═══════════════════════════════════════════════════════════════════════

def compute_ehlers_suite(
    close: pd.Series,
    fast_limit: float = 0.5,
    slow_limit: float = 0.05,
) -> pd.DataFrame:
    """
    Compute the full suite of Ehlers indicators for a price series.

    Returns a DataFrame with columns:
        dominant_cycle, MAMA, FAMA, iTrend, cyber_cycle,
        adaptive_rsi, roofing, super_smooth_10, super_smooth_20

    Useful for rapid exploration in notebooks.
    """
    dc = hilbert_transform_dominant_cycle(close)
    mama_s, fama_s = mama_fama(close, fast_limit, slow_limit)
    itrend = instantaneous_trendline(close)
    cc = cyber_cycle(close)
    arsi = adaptive_rsi(close)
    roof = roofing_filter(close)
    ss10 = super_smoother(close, 10)
    ss20 = super_smoother(close, 20)

    return pd.DataFrame({
        "dominant_cycle": dc,
        "MAMA": mama_s,
        "FAMA": fama_s,
        "iTrend": itrend,
        "cyber_cycle": cc,
        "adaptive_rsi": arsi,
        "roofing": roof,
        "super_smooth_10": ss10,
        "super_smooth_20": ss20,
    }, index=close.index)
