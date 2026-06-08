"""
Teste unitare pentru funcțiile pure din app.py.
Rulează: pytest test_app.py -v
"""
import pytest
import pandas as pd
from datetime import datetime, timezone
from app import (
    MarketWindow, Signal, PolymarketMarket,
    calc_delta, delta_weight, delta_to_signal,
    rsi, ema, sma, vwap, momentum,
    predict_signal, detect_edge,
    current_window,
)


class TestWindowDelta:
    def test_positive(self):
        sig = delta_to_signal(0.0015, 10)
        assert sig.side == "UP"
        assert sig.confidence > 0.5

    def test_negative(self):
        sig = delta_to_signal(-0.0012, 10)
        assert sig.side == "DOWN"
        assert sig.confidence > 0.5

    def test_weight_high(self):
        assert delta_weight(0.0020) == 7.0

    def test_weight_low(self):
        assert delta_weight(0.00001) == 0.5


class TestIndicators:
    def test_rsi(self):
        closes = [100 + i * 0.5 for i in range(20)]
        r = rsi(closes, 14)
        assert r is not None
        assert 0 <= r <= 100

    def test_sma(self):
        vals = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        assert sma(vals, 5) == 17.0

    def test_ema(self):
        vals = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        e = ema(vals, 5)
        assert e is not None

    def test_vwap(self):
        rows = []
        for i in range(10):
            rows.append({"high": 102, "low": 98, "close": 100 + i, "volume": 10})
        df = pd.DataFrame(rows)
        v = vwap(df, 10)
        assert v is not None

    def test_momentum(self):
        closes = [100, 101, 102, 103, 104, 105, 106]
        m = momentum(closes, 5)
        assert m is not None


class TestPredictor:
    def test_predict_up(self):
        window = MarketWindow(
            window_start_ts=1771168800,
            window_end_ts=1771169100,
            slug="btc-updown-5m-1771168800",
            asset="BTC",
            interval_minutes=5,
        )
        rows = []
        for i in range(15):
            rows.append({"open": 100, "high": 101, "low": 99, "close": 100 + i * 0.1, "volume": 1})
        df = pd.DataFrame(rows)
        sig = predict_signal(window, df, 100.0, 100.5, 0.0001)
        assert sig.side == "UP"
        assert sig.confidence >= 0.5


class TestEdgeDetector:
    def test_edge_detected(self):
        m = PolymarketMarket(
            slug="x",
            up_mid=0.52,
            down_mid=0.48,
        )
        s = Signal(side="UP", confidence=0.9, estimated_prob=0.62, confidence_level="HIGH")
        alert = detect_edge(m, s)
        assert alert is not None
        assert alert["type"] == "EDGE"

    def test_no_edge(self):
        m = PolymarketMarket(
            slug="x",
            up_mid=0.50,
            down_mid=0.50,
        )
        s = Signal(side="UP", confidence=0.55, estimated_prob=0.53, confidence_level="MEDIUM")
        alert = detect_edge(m, s)
        assert alert is None


class TestMarketWindow:
    def test_divisible(self):
        w = current_window()
        assert w.window_start_ts % 300 == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
