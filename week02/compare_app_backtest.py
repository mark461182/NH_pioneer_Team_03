"""앱 백테스트(bar-v1)의 S4 거래가 우리 규칙과 같은지 맞대 본다.

앱은 2026년 12월물(A016C000) 한 종목의 일봉으로 셈하고, 우리 백테스트(optimize_daily.py)는
근월물을 이어 붙인 롤오버 수정가로 셈한다. 데이터가 다르면 거래도 달라지므로 두 가지로 맞댄다.
  ① 같은 데이터: 12월물 일봉(results/k200_202612_1d.csv)에 S4 규칙을 Python 으로 돌린 거래
  ② 다른 데이터: 연결 수정가(results/k200_1d_continuous.csv)로 돌린 거래 (6-3절 표와 같음)
체결은 둘 다 「종가 판정 → 다음 거래일 시가」 이다.

실행: python compare_app_backtest.py results/app_backtest_s4_<시각>.json   (week02 폴더에서)
결과: results/app_backtest_compare.json
"""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backtest_daily import load, rsi
from optimize_daily import s4

RESULTS = Path(__file__).resolve().parent / "results"
KST = timezone(timedelta(hours=9))
SINCE = "2025-01-01"


def kst_day(ms: int) -> date:
    return datetime.fromtimestamp(ms / 1000, KST).date()


def s4_on(close: pd.Series) -> pd.Series:
    """S4 튜닝 1위 (RSI(3) < 30, 종가 > SMA100 진입, 종가 > SMA10 청산) — 종가 판정 포지션."""
    trend, exit_line, r = close.rolling(100).mean(), close.rolling(10).mean(), rsi(close, 3)
    pos, out = 0, []
    for i in range(len(close)):
        if pos == 0 and close.iloc[i] > trend.iloc[i] and r.iloc[i] < 30:
            pos = 1
        elif pos == 1 and close.iloc[i] > exit_line.iloc[i]:
            pos = 0
        out.append(pos)
    return pd.Series(out, index=close.index)


def trades_of(pos: pd.Series) -> list[tuple[date, date | None]]:
    """종가 판정 포지션 → (체결 진입일, 체결 청산일). 체결은 다음 거래일 시가."""
    held = pos.shift(1).fillna(0).loc[SINCE:]
    made, start = [], None
    for day, h in held.items():
        if h == 1 and start is None:
            start = day.date()
        elif h == 0 and start is not None:
            made.append((start, day.date()))
            start = None
    if start is not None:
        made.append((start, None))
    return made


def matched(app: list, ours: list) -> dict:
    span = lambda t: f"{t[0]} → {t[1] or '보유 중'}"
    return {"앱 거래": len(app), "Python 거래": len(ours),
            "진입일 같음": len({a[0] for a in app} & {o[0] for o in ours}),
            "진입·청산일 모두 같음": len(set(app) & set(ours)),
            "앱에만": [span(a) for a in sorted(set(app) - set(ours))],
            "Python 에만": [span(o) for o in sorted(set(ours) - set(app))]}


def main() -> None:
    answer = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    app = [(kst_day(t["openedAt"]), kst_day(t["closedAt"])) for t in answer["trades"]]

    one = pd.read_csv(RESULTS / "k200_202612_1d.csv", dtype={"date": str})
    one.index = pd.to_datetime(one.pop("date"))
    same_data = trades_of(s4_on(one["close"]))

    cont = load()
    other_data = trades_of(s4(cont, n=3, entry=30, trend=100, exit_n=10))

    out = {
        "앱 답": {"모형": answer["model"], "봉": {"처음": str(kst_day(answer["bars"]["from"])),
                                              "끝": str(kst_day(answer["bars"]["to"])), "수": answer["bars"]["count"]},
                 "손익(원)": round(answer["pl"]), "수수료(원)": round(answer["fee"]), "승": answer["wins"],
                 "패": answer["losses"], "최대 낙폭(원)": round(answer["maxDrawdown"])},
        "① 같은 데이터 (12월물 일봉)": matched(app, same_data),
        "② 다른 데이터 (연결 수정가)": matched(app, other_data),
    }
    (RESULTS / "app_backtest_compare.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
