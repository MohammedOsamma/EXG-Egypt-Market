from unittest.mock import patch, MagicMock
from egx.data import live


def _resp(data, status=200):
    r = MagicMock(); r.status_code = status; r.json.return_value = {"data": data}
    r.raise_for_status.return_value = None
    return r


def test_parse_quotes_and_skip_bad_rows():
    cols = live.BASE_COLS + ["update_mode"]
    rows = [{"s": "EGX:COMI", "d": ["COMI", "CIB", 80.5, 80, 81, 79.5, 1000, 0.6, "delayed_streaming_900"]},
            {"s": "EGX:BAD", "d": ["BAD", "x", None, 1, 1, 1, 1, 0, "delayed_streaming_900"]},
            {"s": "OTHER:XYZ", "d": ["XYZ", "x", 5, 5, 5, 5, 5, 0, "streaming"]}]
    with patch("egx.data.live.requests.post", return_value=_resp(rows)):
        df = live.fetch_quotes()
    assert list(df["symbol"]) == ["COMI"] and df["close"].iloc[0] == 80.5
    assert live.delay_label(df["update_mode"].iloc[0]) == "delayed 15 min"


def test_retry_without_update_mode_on_400():
    rows = [{"s": "EGX:HRHO", "d": ["HRHO", "EFG", 20.0, 19, 21, 18, 500, 1.0]}]
    with patch("egx.data.live.requests.post", side_effect=[_resp([], 400), _resp(rows)]) as p:
        df = live.fetch_quotes()
    assert p.call_count == 2 and df["update_mode"].iloc[0] is None
    assert "not reported" in live.delay_label(None)
