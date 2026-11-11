"""tradingsignallab 의 지표 6종(신호 12개)을 삼성–하이닉스 「스프레드」 봉에 그대로 붙여 본다 (README 6-11)

  python tsl_on_spread.py --tsl ../../../minyoungkang1227/tradingsignallab   → results/tsl_spread.csv

스프레드 봉 만들기
  1분마다 r_s = rA − β·rB (β = 직전 60거래일 일간 회귀, 하루 고정) → 누적해 지수 S = 100·exp(Σ r_s)
  30분봉·일봉 OHLC = 그 구간 안 S 의 시가·고가·저가·종가, 거래량 = 두 다리 거래량 합 (스프레드 자체 거래량은 없어서)
  지표는 tradingsignallab 의 compute() 를 기본값 그대로 호출 (내가 손대지 않음)

체결·손익 (week04 와 같은 방식)
  신호 봉 다음 봉 시가 진입 → h 봉 뒤 종가 청산, 1단위 = 하이닉스 1 + β 로 맞춘 삼성, 두 다리 1틱 + 0.003%
  손익은 실제 두 다리 가격으로 원 단위 계산 (지수 S 는 신호에만 씀)
  방향: 지표의 강세 = 롱 스프레드(삼성 매수·하이닉스 매도)

판정 (tradingsignallab 의 게이트를 그대로)
  학습 2022~2024: 거래 ≥ 10, 비용 전 t ≥ 3.0 (겹침 제거) 이고 비용 뒤 평균 > 0 → 후보
  후보만 검증 2025~2026 에서 한 번: 비용 뒤 평균 > 0 이고 비용 전 t ≥ 2 → 채택
  12 신호 × 2 봉 길이 × 3 보유기간 = 72번 시험 → BH(FDR 10%) 도 같이 표시
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import pair_core as pc

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
MULT = 10
IS_END = "2024-12-31"


def spread_bars(d: pd.DataFrame, beta: pd.Series, freq: str) -> pd.DataFrame:
    day = d.index.normalize()
    bt = pd.Series(day.map(beta).values, index=d.index)
    la, lb = np.log(d["a_close"]), np.log(d["b_close"])
    rs = (la.diff() - bt * lb.diff()).fillna(0.0)
    rs[~np.isfinite(rs)] = 0.0
    S = 100 * np.exp(rs.cumsum())
    S_open = S / np.exp(rs)                                   # 그 분 시작 시점 지수
    key = d.index.normalize() if freq == "D" else d.index.floor("30min")
    g = pd.DataFrame({"o": S_open, "c": S, "va": d["a_volume"], "vb": d["b_volume"],
                      "ao": d["a_open"], "ac": d["a_close"], "bo": d["b_open"], "bc": d["b_close"], "beta": bt}).groupby(key)
    bars = pd.DataFrame({"open": g["o"].first(), "high": g["c"].max(), "low": g["c"].min(), "close": g["c"].last(),
                         "volume": g["va"].sum() + g["vb"].sum(),
                         "a_open": g["ao"].first(), "a_close": g["ac"].last(), "b_open": g["bo"].first(), "b_close": g["bc"].last(),
                         "beta": g["beta"].first()})
    bars["high"] = bars[["high", "open"]].max(axis=1)
    bars["low"] = bars[["low", "open"]].min(axis=1)
    return bars.dropna(subset=["beta"])


def trades(bars: pd.DataFrame, ev: pd.Series, direction: int, h: int) -> pd.DataFrame:
    idx = np.flatnonzero(ev.fillna(False).values)
    rows, last_exit = [], -1
    c = pc.Costs()
    for i in idx:
        j0, j1 = i + 1, i + h
        if j1 >= len(bars) or j0 <= last_exit:                # 겹침 제거
            continue
        r0, r1 = bars.iloc[j0], bars.iloc[j1]
        bt = bars["beta"].iloc[i]
        qa = pc.hedge_qty(bt, r0.a_open, r0.b_open, pc.Params())
        if qa <= 0:
            continue
        gross = direction * (qa * (r1.a_close - r0.a_open) - (r1.b_close - r0.b_open)) * MULT
        cost = pc.round_trip_cost(qa, 1, (r0.a_open + r1.a_close) / 2, (r0.b_open + r1.b_close) / 2, c, MULT)
        rows.append({"date": bars.index[i].strftime("%Y-%m-%d"), "gross": gross, "cost": cost})
        last_exit = j1
    return pd.DataFrame(rows, columns=["date", "gross", "cost"])


def stat(T: pd.DataFrame) -> dict:
    n = len(T)
    if n < 2 or T.gross.std() == 0:
        return {"n": n, "gross": np.nan, "t": np.nan, "cost": np.nan, "net": np.nan, "p": np.nan}
    t = T.gross.mean() / T.gross.std() * math.sqrt(n)
    from math import erf
    p = 1 - 0.5 * (1 + erf(t / math.sqrt(2)))                 # 단측(이익 쪽)
    return {"n": n, "gross": T.gross.mean(), "t": t, "cost": T.cost.mean(), "net": (T.gross - T.cost).mean(), "p": p}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsl", default=str(HERE.parents[2] / "minyoungkang1227" / "tradingsignallab"))
    a = ap.parse_args()
    sys.path.insert(0, a.tsl)
    eng = importlib.import_module("backtest.engine")

    d = pd.read_csv(OUT / "pair_1m_full.csv", dtype={"date": str, "time": str})
    d.index = pd.to_datetime(d["date"] + d["time"].str.zfill(6), format="%Y%m%d%H%M%S")
    d = d[d["time"] < "153000"]
    beta = pd.read_csv(OUT / "signal_daily.csv", index_col=0, parse_dates=True)["beta"]

    rows = []
    for freq, hs in (("30min", (1, 2, 4)), ("D", (1, 5, 10))):
        bars = spread_bars(d, beta, freq)
        computed = {}
        for key, col, direction, label in eng.SIGNAL_SPECS:
            if key not in computed:
                computed[key] = importlib.import_module(f"indicators.{key}").compute(bars[["open", "high", "low", "close", "volume"]].copy())
            ev = computed[key][col].reindex(bars.index)
            for h in hs:
                T = trades(bars, ev, direction, h)
                tr, va = T[T.date <= IS_END], T[T.date > IS_END]
                s1, s2 = stat(tr), stat(va)
                rows.append({"봉": freq, "신호": label, "보유봉": h,
                             "학습_n": s1["n"], "학습_비용전": s1["gross"], "학습_t": s1["t"], "학습_비용": s1["cost"], "학습_순": s1["net"], "학습_p": s1["p"],
                             "검증_n": s2["n"], "검증_비용전": s2["gross"], "검증_t": s2["t"], "검증_비용": s2["cost"], "검증_순": s2["net"]})
    R = pd.DataFrame(rows)
    p = R["학습_p"].fillna(1.0).values
    R["학습_BH"] = eng.benjamini_hochberg(p, 0.10)
    R["후보"] = (R.학습_n >= 10) & (R.학습_t >= 3.0) & (R.학습_순 > 0)
    R["채택"] = R.후보 & (R.검증_순 > 0) & (R.검증_t >= 2)
    for c in [c for c in R.columns if c.endswith(("비용전", "비용", "순"))]:
        R[c] = R[c].round(0)
    for c in ("학습_t", "검증_t", "학습_p"):
        R[c] = R[c].round(3 if c.endswith("p") else 2)
    R.to_csv(OUT / "tsl_spread.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250, "display.max_rows", 100)
    print(R.drop(columns=["학습_p"]).to_string(index=False))
    print("\n학습 t ≥ 3:", int((R.학습_t >= 3).sum()), "/ 학습 비용 뒤 > 0:", int((R.학습_순 > 0).sum()),
          "/ BH 통과:", int(R.학습_BH.sum()), "/ 후보:", int(R.후보.sum()), "/ 채택:", int(R.채택.sum()), "(72개 중)")


if __name__ == "__main__":
    main()
