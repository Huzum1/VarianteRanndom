"""
Polymarket BTC 5m Analyzer — Single-File Streamlit Edition.

Totul într-un singur fișier: fetchere, analiză, predictor, dashboard live.
Rulare:  streamlit run app.py

V2.1: Error diagnostics, Demo Mode, robust cloud deployment.
"""

# ═══════════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════════
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel, Field, field_validator

# Streamlit importat doar în __main__ pentru a preveni warning-uri la import/testare

# ═══════════════════════════════════════════════════════════
# 2. CONFIG
# ═══════════════════════════════════════════════════════════
class CFG:
    WINDOW_SEC: int = 300
    ASSET: str = "BTC"
    GAMMA_URL: str = "https://gamma-api.polymarket.com"
    CLOB_URL: str = "https://clob.polymarket.com"
    BINANCE_REST: str = "https://api.binance.com"
    BINANCE_FAPI: str = "https://fapi.binance.com"
    SYMBOL: str = "BTCUSDT"
    EDGE_THRESHOLD: float = 0.05
    VOL: float = 0.005
    N_PATHS: int = 2000
    POLL_INTERVAL: float = 2.5
    HIGH_CONF: float = 0.75
    MED_CONF: float = 0.55


# ═══════════════════════════════════════════════════════════
# 3. PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════
class MarketWindow(BaseModel):
    window_start_ts: int
    window_end_ts: int
    slug: str
    asset: str = CFG.ASSET
    interval_minutes: int = 5

    @field_validator("window_start_ts")
    @classmethod
    def _div300(cls, v: int) -> int:
        if v % CFG.WINDOW_SEC != 0:
            raise ValueError("window_start_ts must be divisible by 300")
        return v

    @field_validator("window_end_ts")
    @classmethod
    def _end(cls, v: int, info: Any) -> int:
        start = info.data.get("window_start_ts")
        if start is not None and v != start + CFG.WINDOW_SEC:
            raise ValueError("window_end_ts must be window_start_ts + 300")
        return v

    @property
    def seconds_remaining(self) -> int:
        rem = self.window_end_ts - int(time.time())
        return max(0, rem)

    @property
    def progress(self) -> float:
        elapsed = CFG.WINDOW_SEC - self.seconds_remaining
        return min(1.0, max(0.0, elapsed / CFG.WINDOW_SEC))


class Signal(BaseModel):
    side: str = Field(..., pattern="^(UP|DOWN)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_level: str = "UNCERTAIN"
    edge_pct: float = 0.0
    estimated_prob: float = Field(..., ge=0.0, le=1.0)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    seconds_to_close: int = 0


class PolymarketMarket(BaseModel):
    slug: str
    market_id: str = ""
    condition_id: str = ""
    question: str = ""
    up_token: str = ""
    down_token: str = ""
    up_buy: float = 0.5
    up_sell: float = 0.5
    up_mid: float = 0.5
    down_buy: float = 0.5
    down_sell: float = 0.5
    down_mid: float = 0.5
    volume: float = 0.0
    liquidity: float = 0.0
    accepting: bool = True
    closed: bool = False

    @property
    def implied_up(self) -> float:
        return self.up_mid

    @property
    def implied_down(self) -> float:
        return self.down_mid

    @property
    def spread(self) -> float:
        return abs(self.up_mid - self.down_mid)


class AnalysisResult(BaseModel):
    window: MarketWindow
    market: Optional[PolymarketMarket] = None
    btc_price: Optional[float] = None
    open_price: Optional[float] = None
    delta_pct: float = 0.0
    funding_rate: Optional[float] = None
    signal: Optional[Signal] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ═══════════════════════════════════════════════════════════
# 4. HELPERS
# ═══════════════════════════════════════════════════════════
def current_window() -> MarketWindow:
    now = int(time.time())
    ws = now - (now % CFG.WINDOW_SEC)
    slug = f"{CFG.ASSET.lower()}-updown-5m-{ws}"
    return MarketWindow(
        window_start_ts=ws,
        window_end_ts=ws + CFG.WINDOW_SEC,
        slug=slug,
    )


def safe_json(resp: requests.Response) -> Optional[Any]:
    try:
        return resp.json()
    except Exception:
        return None


# Simple cache
__cache: Dict[str, Tuple[float, Any]] = {}


def _cache_get(key: str, ttl: int = 60) -> Optional[Any]:
    now = time.time()
    if key in __cache:
        ts, val = __cache[key]
        if now - ts < ttl:
            return val
    return None


def _cache_set(key: str, val: Any) -> None:
    __cache[key] = (time.time(), val)


# ═══════════════════════════════════════════════════════════
# 5. DATA FETCHERS (cu diagnoze robuste)
# ═══════════════════════════════════════════════════════════
def fetch_polymarket_market(slug: str, status: Dict[str, str]) -> Optional[PolymarketMarket]:
    """Gamma API — descoperă piața determinist după slug."""
    cached = _cache_get(slug)
    if cached is not None:
        status["gamma"] = "OK (cached)"
        return cached
    try:
        r = requests.get(
            f"{CFG.GAMMA_URL}/events",
            params={"slug": slug},
            timeout=10,
            headers={"Accept": "application/json", "User-Agent": "PM-BTC-Analyzer/1.0"},
        )
        r.raise_for_status()
        data = safe_json(r)
        if not data:
            status["gamma"] = "FAIL: empty response"
            return None
        if isinstance(data, list):
            data = data[0] if data else None
        if not data or not isinstance(data, dict):
            status["gamma"] = "FAIL: invalid JSON structure"
            return None
        markets = data.get("markets", [])
        if not markets:
            status["gamma"] = "FAIL: no markets in event (market may not exist yet)"
            return None
        m = markets[0]
        token_ids = m.get("clobTokenIds", [])
        outcomes = m.get("outcomes", "[]")
        try:
            outs = json.loads(outcomes) if isinstance(outcomes, str) else outcomes
        except Exception:
            outs = ["Yes", "No"]
        if len(token_ids) < 2 or len(outs) < 2:
            status["gamma"] = "FAIL: missing token IDs or outcomes"
            return None

        up_id, down_id = str(token_ids[0]), str(token_ids[1])

        # CLOB prices
        up_b, up_s, up_m, up_err = fetch_clob_prices(up_id, status)
        dn_b, dn_s, dn_m, dn_err = fetch_clob_prices(down_id, status)

        if up_err or dn_err:
            status["gamma"] = f"OK metadata; CLOB err: {up_err or dn_err}"
        else:
            status["gamma"] = "OK"

        market = PolymarketMarket(
            slug=slug,
            market_id=str(m.get("id", "")),
            condition_id=m.get("conditionId", ""),
            question=m.get("question", ""),
            up_token=up_id,
            down_token=down_id,
            up_buy=up_b,
            up_sell=up_s,
            up_mid=up_m,
            down_buy=dn_b,
            down_sell=dn_s,
            down_mid=dn_m,
            volume=float(m.get("volume", 0) or 0),
            liquidity=float(m.get("liquidity", 0) or 0),
            accepting=bool(m.get("acceptingOrders", True)),
            closed=bool(m.get("closed", False)),
        )
        _cache_set(slug, market)
        return market
    except requests.exceptions.Timeout:
        status["gamma"] = "FAIL: timeout (10s)"
    except requests.exceptions.ConnectionError as e:
        status["gamma"] = f"FAIL: connection error — {e}"
    except requests.exceptions.HTTPError as e:
        status["gamma"] = f"FAIL: HTTP {e.response.status_code}"
    except Exception as e:
        status["gamma"] = f"FAIL: {type(e).__name__}: {e}"
    return None


def fetch_clob_prices(token_id: str, status: Dict[str, str]) -> Tuple[float, float, float, Optional[str]]:
    """CLOB API — best buy / sell / midpoint. Returnează și error string."""
    try:
        rb = requests.get(
            f"{CFG.CLOB_URL}/price",
            params={"token_id": token_id, "side": "BUY"},
            timeout=8,
        )
        rs = requests.get(
            f"{CFG.CLOB_URL}/price",
            params={"token_id": token_id, "side": "SELL"},
            timeout=8,
        )
        b = float(safe_json(rb).get("price", 0.5) or 0.5) if rb.status_code == 200 else 0.5
        s = float(safe_json(rs).get("price", 0.5) or 0.5) if rs.status_code == 200 else 0.5
        status["clob"] = "OK"
        return round(b, 4), round(s, 4), round((b + s) / 2.0, 4), None
    except requests.exceptions.Timeout:
        status["clob"] = "FAIL: timeout (8s)"
        return 0.5, 0.5, 0.5, "timeout"
    except requests.exceptions.ConnectionError:
        status["clob"] = "FAIL: connection error"
        return 0.5, 0.5, 0.5, "connection"
    except Exception as e:
        status["clob"] = f"FAIL: {type(e).__name__}"
        return 0.5, 0.5, 0.5, str(e)


def fetch_binance_price(status: Dict[str, str]) -> Optional[float]:
    try:
        r = requests.get(
            f"{CFG.BINANCE_REST}/api/v3/ticker/price",
            params={"symbol": CFG.SYMBOL},
            timeout=8,
        )
        r.raise_for_status()
        status["binance_price"] = "OK"
        return float(r.json()["price"])
    except requests.exceptions.Timeout:
        status["binance_price"] = "FAIL: timeout (8s)"
    except requests.exceptions.ConnectionError:
        status["binance_price"] = "FAIL: connection error (Binance may block this IP)"
    except requests.exceptions.HTTPError as e:
        status["binance_price"] = f"FAIL: HTTP {e.response.status_code}"
    except Exception as e:
        status["binance_price"] = f"FAIL: {type(e).__name__}: {e}"
    return None


def fetch_binance_klines(limit: int, status: Dict[str, str]) -> pd.DataFrame:
    try:
        r = requests.get(
            f"{CFG.BINANCE_REST}/api/v3/klines",
            params={"symbol": CFG.SYMBOL, "interval": "1m", "limit": limit},
            timeout=10,
        )
        r.raise_for_status()
        raw = r.json()
        rows = []
        for item in raw:
            rows.append(
                {
                    "open_time": pd.to_datetime(int(item[0]), unit="ms", utc=True),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "close_time": pd.to_datetime(int(item[6]), unit="ms", utc=True),
                }
            )
        status["binance_klines"] = "OK"
        return pd.DataFrame(rows)
    except requests.exceptions.Timeout:
        status["binance_klines"] = "FAIL: timeout (10s)"
    except requests.exceptions.ConnectionError:
        status["binance_klines"] = "FAIL: connection error (Binance may block this IP)"
    except Exception as e:
        status["binance_klines"] = f"FAIL: {type(e).__name__}: {e}"
    return pd.DataFrame()


def fetch_binance_funding(status: Dict[str, str]) -> Optional[float]:
    try:
        r = requests.get(
            f"{CFG.BINANCE_FAPI}/fapi/v1/premiumIndex",
            params={"symbol": CFG.SYMBOL},
            timeout=8,
        )
        r.raise_for_status()
        status["binance_funding"] = "OK"
        return float(r.json().get("lastFundingRate", 0) or 0)
    except requests.exceptions.Timeout:
        status["binance_funding"] = "FAIL: timeout"
    except requests.exceptions.ConnectionError:
        status["binance_funding"] = "FAIL: connection error"
    except Exception as e:
        status["binance_funding"] = f"FAIL: {type(e).__name__}: {e}"
    return None


def fetch_binance_depth(status: Dict[str, str]) -> Tuple[Optional[float], Optional[float]]:
    try:
        r = requests.get(
            f"{CFG.BINANCE_REST}/api/v3/depth",
            params={"symbol": CFG.SYMBOL, "limit": 10},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        bb = float(bids[0][0]) if bids else None
        ba = float(asks[0][0]) if asks else None
        status["binance_depth"] = "OK"
        return bb, ba
    except requests.exceptions.Timeout:
        status["binance_depth"] = "FAIL: timeout"
    except requests.exceptions.ConnectionError:
        status["binance_depth"] = "FAIL: connection error"
    except Exception as e:
        status["binance_depth"] = f"FAIL: {type(e).__name__}: {e}"
    return None, None


def get_window_open_price(window_start_ts: int, df: pd.DataFrame) -> Optional[float]:
    if df.empty:
        return None
    target = pd.to_datetime(window_start_ts, unit="s", utc=True)
    for _, row in df.iterrows():
        if row["open_time"] <= target <= row["close_time"]:
            return float(row["open"])
    return float(df.iloc[-1]["close"]) if not df.empty else None


# ═══════════════════════════════════════════════════════════
# 6. ANALYSIS (pure functions)
# ═══════════════════════════════════════════════════════════
def calc_delta(open_price: float, current_price: float) -> float:
    if open_price <= 0:
        return 0.0
    return (current_price - open_price) / open_price


def delta_weight(delta_pct: float) -> float:
    ad = abs(delta_pct)
    if ad > 0.0010:
        return 7.0
    if ad > 0.0005:
        return 5.0
    if ad > 0.0002:
        return 3.0
    if ad > 0.00005:
        return 1.0
    return 0.5


def delta_to_signal(delta_pct: float, seconds_left: int) -> Signal:
    raw_prob = 0.50 + delta_pct * 500
    base = max(0.0, min(1.0, raw_prob))
    if seconds_left > 60:
        base = 0.5 + (base - 0.5) * 0.80
    elif seconds_left > 30:
        base = 0.5 + (base - 0.5) * 0.90
    elif seconds_left <= 10:
        base = 0.5 + (base - 0.5) * 1.05
    base = max(0.0, min(1.0, base))
    side = "UP" if delta_pct >= 0 else "DOWN"
    conf = base if side == "UP" else (1.0 - base)
    level = (
        "HIGH" if conf >= CFG.HIGH_CONF else
        "MEDIUM" if conf >= CFG.MED_CONF else
        "LOW" if conf >= 0.45 else "UNCERTAIN"
    )
    return Signal(
        side=side,
        confidence=round(conf, 4),
        confidence_level=level,
        estimated_prob=round(base, 4) if side == "UP" else round(1.0 - base, 4),
        indicators={"window_delta_pct": round(delta_pct * 100, 4), "delta_weight": delta_weight(delta_pct)},
        seconds_to_close=seconds_left,
    )


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    arr = np.array(closes[-period - 1:], dtype=float)
    deltas = np.diff(arr)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses)) or 1e-10
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 4)


def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    arr = np.array(values, dtype=float)
    alpha = 2.0 / (period + 1)
    ema_val = arr[0]
    for v in arr[1:]:
        ema_val = alpha * v + (1 - alpha) * ema_val
    return round(float(ema_val), 4)


def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return round(float(np.mean(values[-period:])), 4)


def vwap(df: pd.DataFrame, period: int = 10) -> Optional[float]:
    if len(df) < period:
        return None
    recent = df.tail(period).copy()
    recent["typical"] = (recent["high"] + recent["low"] + recent["close"]) / 3.0
    return round(float((recent["typical"] * recent["volume"]).sum() / recent["volume"].sum()), 4)


def momentum(closes: List[float], period: int = 5) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    return round((closes[-1] - closes[-period - 1]) / closes[-period - 1] * 100, 4)


def predict_signal(
    window: MarketWindow,
    df: pd.DataFrame,
    open_price: float,
    current_price: float,
    funding_rate: Optional[float],
) -> Signal:
    delta_pct = calc_delta(open_price, current_price)
    delta_sig = delta_to_signal(delta_pct, window.seconds_remaining)

    closes = df["close"].tolist() if not df.empty and "close" in df.columns else []
    tech_score = 0.0
    tech_ind = {}
    if len(closes) >= 5:
        tech_ind["rsi_14"] = rsi(closes, 14)
        tech_ind["rsi_7"] = rsi(closes, 7)
        tech_ind["ema_9"] = ema(closes, 9)
        tech_ind["sma_5"] = sma(closes, 5)
        tech_ind["vwap_10"] = vwap(df, 10)
        tech_ind["momentum_5"] = momentum(closes, 5)

        scores = []
        if tech_ind["rsi_14"] is not None:
            rsi_v = tech_ind["rsi_14"]
            scores.append((rsi_v - 50) / 50.0)
        if tech_ind["momentum_5"] is not None:
            mom = tech_ind["momentum_5"]
            scores.append(np.sign(mom) * min(abs(mom) / 0.5, 1.0))
        if tech_ind["vwap_10"] is not None and current_price > 0:
            scores.append(0.5 if current_price > tech_ind["vwap_10"] else -0.5)
        if scores:
            tech_score = np.clip(np.mean(scores), -1.0, 1.0)

    seconds_left = window.seconds_remaining
    mc_prob = 0.5
    if seconds_left > 0 and current_price > 0 and open_price > 0:
        dt = seconds_left / 3600.0
        shocks = np.random.standard_normal(CFG.N_PATHS)
        paths = current_price * np.exp((-0.5 * CFG.VOL ** 2) * dt + CFG.VOL * np.sqrt(dt) * shocks)
        mc_prob = float(np.mean(paths >= open_price))

    funding_bias = 0.0
    if funding_rate is not None and abs(funding_rate) > 0.001:
        funding_bias = np.sign(funding_rate) * 0.02

    delta_prob = delta_sig.estimated_prob if delta_sig.side == "UP" else (1.0 - delta_sig.estimated_prob)
    combined = 0.55 * delta_prob + 0.30 * (0.5 + tech_score * 0.5) + 0.15 * mc_prob + funding_bias
    combined = max(0.01, min(0.99, combined))

    side = "UP" if combined >= 0.5 else "DOWN"
    confidence = combined if side == "UP" else (1.0 - combined)
    level = (
        "HIGH" if confidence >= CFG.HIGH_CONF else
        "MEDIUM" if confidence >= CFG.MED_CONF else
        "LOW" if confidence >= 0.45 else "UNCERTAIN"
    )

    return Signal(
        side=side,
        confidence=round(confidence, 4),
        confidence_level=level,
        estimated_prob=round(combined, 4),
        indicators={
            **tech_ind,
            "window_delta_pct": round(delta_pct * 100, 4),
            "mc_prob": round(mc_prob, 4),
            "funding_bias": round(funding_bias, 4),
        },
        seconds_to_close=seconds_left,
    )


def detect_edge(market: Optional[PolymarketMarket], signal: Optional[Signal]) -> Optional[Dict[str, Any]]:
    if not market or not signal:
        return None
    estimated = signal.estimated_prob
    implied = market.implied_up
    edge = estimated - implied
    signal.edge_pct = round(edge, 4)

    if edge >= CFG.EDGE_THRESHOLD:
        return {"type": "EDGE", "priority": 1, "title": f"EDGE {signal.side} (+{edge:.2%})", "body": f"Est:{estimated:.2%} vs Market:{implied:.2%}"}
    if signal.confidence >= CFG.HIGH_CONF and edge > 0.02:
        return {"type": "HIGH_CONF", "priority": 2, "title": f"High Confidence {signal.side}", "body": f"Conf:{signal.confidence:.2%} Est:{estimated:.2%}"}
    if market.implied_up + market.implied_down < 0.98:
        return {"type": "ARBITRAGE", "priority": 1, "title": "Arbitrage hint", "body": f"Sum={market.implied_up+market.implied_down:.4f}"}
    return None


# ═══════════════════════════════════════════════════════════
# 7. DEMO DATA GENERATOR (pentru testare UI când API-urile eșuează)
# ═══════════════════════════════════════════════════════════
def generate_demo_data(window: MarketWindow) -> Tuple[PolymarketMarket, float, pd.DataFrame, float, float, float]:
    """Generează date simulate realistice pentru a testa UI-ul."""
    base_price = 105000.0
    # Simulăm un preț în creștere ușoară
    price = base_price + (window.progress * 80) + np.random.normal(0, 15)
    
    # Simulăm istoric de 20 lumânări
    rows = []
    for i in range(20):
        t = window.window_start_ts - (20 - i) * 60
        rows.append({
            "open_time": pd.to_datetime(t, unit="s", utc=True),
            "open": base_price + i * 4,
            "high": base_price + i * 4 + 20,
            "low": base_price + i * 4 - 20,
            "close": base_price + i * 4 + 10,
            "volume": 100 + np.random.randint(0, 50),
            "close_time": pd.to_datetime(t + 60, unit="s", utc=True),
        })
    df = pd.DataFrame(rows)
    
    # Odds simulate: UP crescător, DOWN descrescător
    up_mid = 0.50 + (window.progress * 0.18) + np.random.normal(0, 0.01)
    up_mid = max(0.02, min(0.98, up_mid))
    down_mid = 1.0 - up_mid + np.random.normal(0, 0.005)
    down_mid = max(0.02, min(0.98, down_mid))
    # Re-normalizăm
    total = up_mid + down_mid
    up_mid = up_mid / total
    down_mid = down_mid / total
    
    spread = 0.02
    market = PolymarketMarket(
        slug=window.slug,
        market_id="DEMO123",
        question="Will BTC be UP at 5m?",
        up_buy=round(up_mid - spread/2, 4),
        up_sell=round(up_mid + spread/2, 4),
        up_mid=round(up_mid, 4),
        down_buy=round(down_mid - spread/2, 4),
        down_sell=round(down_mid + spread/2, 4),
        down_mid=round(down_mid, 4),
        volume=150000.0,
        liquidity=45000.0,
    )
    
    open_price = rows[0]["open"]
    funding = 0.0001
    bb = price - 10
    ba = price + 10
    return market, price, df, open_price, funding, bb, ba


# ═══════════════════════════════════════════════════════════
# 8. STREAMLIT UI
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import streamlit as st

    st.set_page_config(
        page_title="Polymarket BTC 5m Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")
        demo_mode = st.toggle("🎮 Demo Mode (simulate data)", value=False, help="When APIs fail, use this to see the UI and logic working with fake data")
        st.divider()
        st.markdown("**Status Legend:**")
        st.markdown("🟢 OK — API responded")
        st.markdown("🔴 FAIL — API error (hover for details)")
        st.divider()
        st.caption("v2.1 — Single-file Streamlit Edition")

    # ── Session State ────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []
        st.session_state.alerts = []
        st.session_state.last_window_ts = 0
        st.session_state.window_open_price = None

    st.title("🔮 Polymarket BTC 5m Analyzer")
    st.caption("Live dashboard + predictor + edge detector. Toggle Demo Mode in sidebar if APIs are blocked.")

    window = current_window()
    status: Dict[str, str] = {}

    # ── Status Panel ─────────────────────────────────────────
    st.subheader("🌐 API Status")
    status_cols = st.columns(6)
    
    def render_status(key: str, label: str, col):
        s = status.get(key, "PENDING")
        if s.startswith("OK"):
            col.metric(label, "🟢 OK", delta=None)
        elif s.startswith("FAIL"):
            col.metric(label, "🔴 FAIL", delta=s[6:30] if len(s) > 6 else None)
        else:
            col.metric(label, "⚪ ...", delta=None)
    
    # ── Data Fetching ──────────────────────────────────────
    if demo_mode:
        market, btc_price, klines_df, open_price, funding, bb, ba = generate_demo_data(window)
        status = {k: "OK (demo)" for k in ["gamma", "clob", "binance_price", "binance_klines", "binance_funding", "binance_depth"]}
        st.session_state.window_open_price = open_price
    else:
        # New window detection
        if window.window_start_ts != st.session_state.last_window_ts:
            st.session_state.last_window_ts = window.window_start_ts
            klines_df = fetch_binance_klines(5, status)
            st.session_state.window_open_price = get_window_open_price(window.window_start_ts, klines_df)
        
        market = fetch_polymarket_market(window.slug, status)
        btc_price = fetch_binance_price(status)
        klines_df = fetch_binance_klines(20, status)
        funding = fetch_binance_funding(status)
        bb, ba = fetch_binance_depth(status)
        open_price = st.session_state.window_open_price

    # Render status after fetch
    status_labels = [
        ("gamma", "Polymarket γ"),
        ("clob", "Polymarket CLOB"),
        ("binance_price", "Binance Price"),
        ("binance_klines", "Binance Klines"),
        ("binance_funding", "Binance Funding"),
        ("binance_depth", "Binance Depth"),
    ]
    for i, (key, label) in enumerate(status_labels):
        render_status(key, label, status_cols[i])
    
    # Show full status details in expander
    with st.expander("Detailed API diagnostics"):
        for key, label in status_labels:
            st.write(f"**{label}:** {status.get(key, 'not called')}")
    
    # ── Analysis ───────────────────────────────────────────
    signal = None
    result = None
    delta_pct = 0.0
    
    if open_price and btc_price:
        delta_pct = calc_delta(open_price, btc_price)
        signal = predict_signal(window, klines_df, open_price, btc_price, funding)
        result = AnalysisResult(
            window=window,
            market=market,
            btc_price=btc_price,
            open_price=open_price,
            delta_pct=delta_pct,
            funding_rate=funding,
            signal=signal,
        )
        
        alert = detect_edge(market, signal)
        if alert:
            alert["time"] = datetime.now(timezone.utc).strftime("%H:%M:%S")
            st.session_state.alerts.insert(0, alert)
            if len(st.session_state.alerts) > 30:
                st.session_state.alerts = st.session_state.alerts[:30]
            st.toast(f"🚨 {alert['title']}", icon="⚡")
        
        st.session_state.history.append(
            {
                "time": datetime.now(timezone.utc),
                "up_odds": market.implied_up if market else 0.5,
                "down_odds": market.implied_down if market else 0.5,
                "btc_price": btc_price,
                "delta_pct": delta_pct,
                "confidence": signal.confidence,
                "edge": signal.edge_pct,
            }
        )
        if len(st.session_state.history) > 120:
            st.session_state.history = st.session_state.history[-120:]
    
    # ── Progress ───────────────────────────────────────────
    progress_text = st.empty()
    progress_bar = st.progress(0.0)
    progress_bar.progress(window.progress)
    progress_text.markdown(f"**Window:** `{window.slug}` — ⏱️ **{window.seconds_remaining}s** remaining")
    
    # ── Metrics ────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            label="BTC Price",
            value=f"${btc_price:,.2f}" if btc_price else "N/A",
            delta=f"{delta_pct*100:.4f}%" if open_price and btc_price else None,
        )
    with m2:
        st.metric(
            label="Polymarket UP",
            value=f"{market.implied_up:.2%}" if market else "N/A",
            delta=f"buy {market.up_buy:.2f} / sell {market.up_sell:.2f}" if market else None,
        )
    with m3:
        st.metric(
            label="Polymarket DOWN",
            value=f"{market.implied_down:.2%}" if market else "N/A",
            delta=f"buy {market.down_buy:.2f} / sell {market.down_sell:.2f}" if market else None,
        )
    with m4:
        if signal:
            color = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🟠", "UNCERTAIN": "⚪"}.get(signal.confidence_level, "⚪")
            st.metric(
                label=f"Signal {color}",
                value=f"{signal.side} ({signal.confidence_level})",
                delta=f"conf {signal.confidence:.2%} | edge {signal.edge_pct:.2%}",
            )
        else:
            st.metric(label="Signal", value="N/A")
    
    # ── Main Content ───────────────────────────────────────
    col_left, col_right = st.columns([2, 1])
    with col_left:
        if st.session_state.history:
            hist_df = pd.DataFrame(st.session_state.history)
            hist_df.set_index("time", inplace=True)
            st.subheader("📈 Odds & BTC History (last 120 ticks)")
            st.line_chart(hist_df[["up_odds", "down_odds", "btc_price"]], use_container_width=True)
        else:
            st.info("Waiting for first data tick...")
        
        if result:
            st.subheader("📋 Analysis Details")
            st.json(
                {
                    "window": window.model_dump(),
                    "btc_price": result.btc_price,
                    "open_price": result.open_price,
                    "delta_pct": f"{result.delta_pct:.4%}",
                    "funding_rate": result.funding_rate,
                    "signal": signal.model_dump() if signal else None,
                    "market": market.model_dump() if market else None,
                    "best_bid": bb,
                    "best_ask": ba,
                    "spread": f"{ba - bb:.2f}" if bb and ba else None,
                }
            )
    
    with col_right:
        st.subheader("🚨 Alerts")
        if st.session_state.alerts:
            for a in st.session_state.alerts[:10]:
                emoji = {"EDGE": "🎯", "HIGH_CONF": "🔥", "ARBITRAGE": "⚠️"}.get(a["type"], "📢")
                st.markdown(f"**{emoji} {a['time']} — {a['title']}**<br>{a['body']}", unsafe_allow_html=True)
        else:
            st.info("No alerts yet. Waiting for edge...")
        
        st.subheader("📊 Technicals")
        if signal and signal.indicators:
            for k, v in signal.indicators.items():
                if v is not None:
                    st.text(f"{k}: {v}")
        else:
            st.text("No indicators yet.")
    
    # ── Auto-refresh ───────────────────────────────────────
    st.caption(f"Auto-refresh every {CFG.POLL_INTERVAL}s...")
    time.sleep(CFG.POLL_INTERVAL)
    st.rerun()
