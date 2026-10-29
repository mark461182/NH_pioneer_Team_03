"""앱 동작 확인용 실행(strategy_check.json)이 장중에 실제로 거래를 낼지 미리 본다.

1분봉을 5분봉(주간 08:45~15:45)으로 묶고 strategy_check.json 과 같은 규칙을 돌린다.
  진입: 5분봉 마감 RSI(3) < 30 → 다음 봉 시가 + 1틱, 하루 3번까지
  청산: 종가 > MA(10) → 다음 봉 시가 / 손절 100틱 / 15:35 마감 봉에서 정리
수익성 검증이 아니라 「하루에 거래가 몇 번, 어떤 청산으로 끝나는가」만 센다.

실행: python check_frequency.py   (week02 폴더에서)
결과: results/check_frequency.json
"""
import json
from datetime import date, time
from pathlib import Path

import pandas as pd

from backtest_daily import TICK, rsi

RESULTS = Path(__file__).resolve().parent / "results"
SINCE = date(2026, 9, 1)
ENTRY_RSI, STOP_TICKS, MAX_ENTRIES, FLAT_AT = 30, 100, 3, time(15, 35)


def five_minute_bars() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "k200_1m_continuous.csv", dtype={"date": str, "time": str})
    d.index = pd.to_datetime(d["date"] + d["time"], format="%Y%m%d%H%M%S")
    d = d.between_time("08:45", "15:45")
    b = d.resample("5min", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    b["rsi3"] = rsi(b["close"], 3)
    b["ma10"] = b["close"].rolling(10).mean()
    b["day"] = b.index.date
    return b


def main() -> None:
    rows = list(five_minute_bars().itertuples())
    stop = STOP_TICKS * TICK
    trades, pos, entries = [], None, {}
    for bar, nxt in zip(rows, rows[1:]):
        last = nxt.day != bar.day or bar.Index.time() >= FLAT_AT
        if pos:
            if bar.low <= pos["price"] - stop:
                trades.append({**pos, "why": "손절"})
                pos = None
            elif last:
                trades.append({**pos, "why": "세션 끝"})
                pos = None
            elif bar.close > bar.ma10:
                trades.append({**pos, "why": "규칙 청산"})
                pos = None
        elif not last and bar.rsi3 < ENTRY_RSI and entries.get(bar.day, 0) < MAX_ENTRIES:
            pos = {"day": bar.day, "price": nxt.open + TICK}
            entries[bar.day] = entries.get(bar.day, 0) + 1

    t = pd.DataFrame(trades)
    t = t[t["day"] >= SINCE]
    days = sorted({r.day for r in rows if r.day >= SINCE})
    per_day = t.groupby("day").size().reindex(days, fill_value=0)
    out = {
        "구간": f"{days[0]} ~ {days[-1]}", "거래일": len(days), "거래 있는 날": int((per_day > 0).sum()),
        "하루 거래 수 최소/평균/최대": [int(per_day.min()), round(float(per_day.mean()), 1), int(per_day.max())],
        "청산 사유 비율": t["why"].value_counts(normalize=True).round(2).to_dict(),
    }
    (RESULTS / "check_frequency.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
