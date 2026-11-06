"""CBBC 벽 돌파 되돌리기 — 6년 1분봉 백테스트 (cbbc_fade.py 와 같은 규칙)

  python backtest_fade.py --events <events_grid.csv 경로>

입력 events_grid.csv: 2020-01 ~ 2026-11 장중 50pt 경계 돌파 이벤트 12,193건과 이후 MHI 선물 경로(p0~p30, 돌파 방향 +pt).
  만드는 법: strategy-lab `cbbc/explore.py build` (HSI 지수·MHI 선물 1분봉 = Futu 개인 계정 데이터 → **저장소에 안 올림**,
  팀 내부 zip 으로만 공유).
출력 results/backtest_summary.csv, results/backtest_by_year.csv (집계값만)

규칙
  즉시판   : 이벤트 분 MHI 종가에 반대 방향 진입, 30분 뒤 청산           (실시간 시세가 있을 때)
  15분 지연 : 이벤트 15분 뒤 진입, 이벤트 30분 뒤 청산 (보유 15분)       (Yahoo 지연 시세로 돌릴 때)
  벽 = 뚫은 50pt 구간 물량 ≥ 300 / 대조 = 물량 0 (CBBC 없는 돌파를 같은 규칙으로 되돌리기)
  한 번에 한 포지션, 왕복 비용 3pt, MHI 1pt = HK$10
  t = 날짜 단위 블록 부트스트랩(2,000번)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COST = 3
OUT = Path(__file__).resolve().parent / "results"


def trades(E: pd.DataFrame, entry: int, exit_: int) -> pd.DataFrame:
    rows = []
    for d, g in E.sort_values("time").groupby("date"):
        busy = None
        for r in g.itertuples(index=False):
            t_in = r.time + pd.Timedelta(minutes=entry)
            if busy is not None and t_in < busy:
                continue
            a = 0.0 if entry == 0 else getattr(r, f"p{entry}")
            b = getattr(r, f"p{exit_}")
            if pd.isna(a) or pd.isna(b):
                continue
            rows.append((str(d), r.time, r.dir, r.qty_here, -(b - a) - COST))
            busy = r.time + pd.Timedelta(minutes=exit_)
    return pd.DataFrame(rows, columns=["date", "time", "dir", "qty", "pnl"])


def boot_t(T: pd.DataFrame, n=2000, seed=0) -> float:
    if len(T) < 5:
        return np.nan
    by = T.groupby("date")["pnl"].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    k = len(by)
    s, c = by["sum"].to_numpy(), by["count"].to_numpy()
    means = [s[i].sum() / c[i].sum() for i in (rng.integers(0, k, k) for _ in range(n))]
    sd = np.std(means)
    return T["pnl"].mean() / sd if sd > 0 else np.nan


def summ(T: pd.DataFrame) -> dict:
    if T.empty:
        return {"거래": 0}
    daily = T.groupby("date")["pnl"].sum()
    return {"거래": len(T), "승률": round((T.pnl > 0).mean(), 3), "평균pt": round(T.pnl.mean(), 2),
            "비용전pt": round(T.pnl.mean() + COST, 2), "t": round(boot_t(T), 2),
            "누적HKD": round(T.pnl.sum() * 10), "거래일": len(daily)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    a = ap.parse_args()
    E = pd.read_csv(a.events, parse_dates=["time"])
    E["date"] = E["date"].astype(str)
    OUT.mkdir(exist_ok=True)
    rows, yrows = [], []
    for grp, sel in (("벽(물량≥300)", E.qty_here >= 300), ("대조(물량 0)", E.qty_here == 0)):
        for ver, (i, o) in (("즉시판", (0, 30)), ("15분 지연판", (15, 30))):
            T = trades(E[sel], i, o)
            for per, m in (("탐색 2020~23", T.date <= "20231231"), ("확인 2024~26", T.date > "20231231"), ("전체", T.date > "")):
                rows.append({"그룹": grp, "규칙": ver, "기간": per, **summ(T[m])})
            for y, g in T.groupby(T.date.str[:4]):
                yrows.append({"그룹": grp, "규칙": ver, "연도": y, **summ(g)})
    S, Y = pd.DataFrame(rows), pd.DataFrame(yrows)
    S.to_csv(OUT / "backtest_summary.csv", index=False, encoding="utf-8-sig")
    Y.to_csv(OUT / "backtest_by_year.csv", index=False, encoding="utf-8-sig")
    print(S.to_string(index=False))
    print()
    print(Y[(Y["그룹"] == "벽(물량≥300)")].to_string(index=False))


if __name__ == "__main__":
    main()
