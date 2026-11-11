"""signal_scan 에서 학습 기준(|t| ≥ 2, 2022·2023·2024 같은 부호)을 통과한 두 후보를 실제 손익으로 (README 6-10)

  python signal_trade.py   → results/signal_trade.json

규칙 (signal_scan 결과를 본 뒤, 검증 구간 손익을 보기 전에 고정)
  공통  1단위 = 하이닉스 1계약 + β 로 맞춘 삼성(pair_core.hedge_qty), β = 직전 60거래일 일간 회귀
        체결 = 신호 시각 **1분 뒤** 봉 시가 (호가 튐을 빼려고 한 봉 건너뜀), 두 다리 시장가 1틱 + 수수료 0.003%
        학습 2022-01 ~ 2024-12 / 검증 2025-01 ~ 2026-11
  S8  오전 잔차 → 오후 (모멘텀)
        12:00 직전 종가 기준 시가→12:00 잔차 부호 쪽으로, 12:01 시가 진입 → 15:29 종가 청산. 매일 1번
  S11 30분 잔차 반전
        30분 봉 끝(xx:29·xx:59 종가)에서 그 30분 잔차 e 가 직전 20일 30분 잔차 σ 의 1배를 넘으면 반대 방향으로,
        다음 봉 첫 분 다음 분(xx:31·xx:01) 시가 진입 → 그 30분 봉 끝 종가 청산.  09:00 첫 봉(밤사이 갭 포함) 신호는 안 씀
  판정  검증 구간에서 학습과 같은 부호(비용 전 평균 > 0) 이고 비용 뒤 순손익 > 0 이면 채택
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import pair_core as pc

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
IS_END, OOS_START = "2024-12-31", "2025-01-01"
MULT = 10


def load():
    d = pd.read_csv(OUT / "pair_1m_full.csv", dtype={"date": str, "time": str})
    d.index = pd.to_datetime(d["date"] + d["time"].str.zfill(6), format="%Y%m%d%H%M%S")
    d = d[d["time"] < "153000"]
    F = pd.read_csv(OUT / "signal_daily.csv", index_col=0, parse_dates=True)
    return d, F["beta"]


def unit_pnl(dir_, beta, a0, b0, a1, b1, c=pc.Costs()):
    """1단위 손익(원). dir +1 = 롱 스프레드(삼성 매수·하이닉스 매도)."""
    qa = pc.hedge_qty(beta, a0, b0, pc.Params())
    gross = dir_ * (qa * (a1 - a0) - (b1 - b0)) * MULT
    cost = pc.round_trip_cost(qa, 1, (a0 + a1) / 2, (b0 + b1) / 2, c, MULT)
    return gross, cost, qa


def summarize(T: pd.DataFrame) -> dict:
    out = {}
    for per, m in (("학습", T["date"] <= IS_END), ("검증", T["date"] >= OOS_START)):
        X = T[m]
        if X.empty:
            out[per] = {"거래": 0}
            continue
        daily = X.groupby("date")["net"].sum()
        out[per] = {"거래": int(len(X)), "승률(비용전)": round(float((X.gross > 0).mean()), 3),
                    "거래당_비용전원": int(X.gross.mean()), "비용전_t": round(float(X.gross.mean() / X.gross.std() * math.sqrt(len(X))), 2),
                    "거래당_비용원": int(X.cost.mean()), "순손익원": int(X.net.sum()),
                    "일간샤프": round(float(daily.mean() / daily.std() * math.sqrt(252)), 2) if daily.std() > 0 else None}
        out[per]["연도별_순손익원"] = {str(k): int(v) for k, v in X.groupby(X["date"].str[:4])["net"].sum().items()}
        out[per]["연도별_비용전원"] = {str(k): int(v) for k, v in X.groupby(X["date"].str[:4])["gross"].sum().items()}
    return out


def run_s8(d: pd.DataFrame, beta: pd.Series) -> pd.DataFrame:
    rows = []
    for day, g in d.groupby("date"):
        dt = pd.Timestamp(day)
        bt = beta.get(dt, np.nan)
        if not math.isfinite(bt):
            continue
        am = g[g["time"] < "120000"]
        pm = g[g["time"] >= "120100"]
        if len(am) < 30 or len(pm) < 30:
            continue
        e_am = math.log(am.a_close.iloc[-1] / g.a_open.iloc[0]) - bt * math.log(am.b_close.iloc[-1] / g.b_open.iloc[0])
        if e_am == 0:
            continue
        dir_ = 1 if e_am > 0 else -1
        a0, b0 = pm.a_open.iloc[0], pm.b_open.iloc[0]
        a1, b1 = pm.a_close.iloc[-1], pm.b_close.iloc[-1]
        gr, co, qa = unit_pnl(dir_, bt, a0, b0, a1, b1)
        rows.append({"date": dt.strftime("%Y-%m-%d"), "dir": dir_, "e_am": e_am, "qa": qa, "gross": gr, "cost": co, "net": gr - co})
    return pd.DataFrame(rows)


def run_s11(d: pd.DataFrame, beta: pd.Series, k: float = 1.0, look_days: int = 20) -> pd.DataFrame:
    key = d.index.floor("30min")
    la, lb = np.log(d["a_close"]), np.log(d["b_close"])
    end_a, end_b = la.groupby(key).last(), lb.groupby(key).last()
    start_a, start_b = np.log(d["a_open"]).groupby(key).first(), np.log(d["b_open"]).groupby(key).first()
    bt = pd.Series(end_a.index.normalize().map(beta).values, index=end_a.index)
    e = (end_a - start_a) - bt * (end_b - start_b)                      # 그 30분 봉 안의 잔차 (봉 시가→종가, 갭 제외)
    first_bar = end_a.index.strftime("%H%M") == "0900"
    e[first_bar] = np.nan
    days = end_a.index.normalize()
    # σ: 직전 look_days 거래일의 30분 잔차 표준편차 (그날 제외)
    dsd = e.groupby(days).apply(lambda s: float((s.dropna() ** 2).sum())).rename("ss")
    dn = e.groupby(days).apply(lambda s: s.notna().sum()).rename("n")
    sig_day = np.sqrt(dsd.rolling(look_days).sum().shift(1) / dn.rolling(look_days).sum().shift(1))
    sig = pd.Series(days.map(sig_day).values, index=e.index)
    # 다음 봉: 시작 1분 뒤 시가 → 봉 끝 종가
    by_bar = {b: g for b, g in d.groupby(key)}
    bars = list(e.index)
    rows = []
    for i in range(len(bars) - 1):
        b, nb = bars[i], bars[i + 1]
        if nb.normalize() != b.normalize() or not math.isfinite(e.iloc[i]) or not math.isfinite(sig.iloc[i]):
            continue
        if abs(e.iloc[i]) < k * sig.iloc[i]:
            continue
        g = by_bar[nb]
        if len(g) < 5:
            continue
        g1 = g.iloc[1:]                                                   # 한 봉(1분) 건너뜀
        dir_ = -1 if e.iloc[i] > 0 else 1                                 # 반전
        gr, co, qa = unit_pnl(dir_, bt.iloc[i], g1.a_open.iloc[0], g1.b_open.iloc[0], g1.a_close.iloc[-1], g1.b_close.iloc[-1])
        rows.append({"date": b.strftime("%Y-%m-%d"), "t": b.strftime("%H:%M"), "dir": dir_, "z": e.iloc[i] / sig.iloc[i],
                     "qa": qa, "gross": gr, "cost": co, "net": gr - co})
    return pd.DataFrame(rows)


def main():
    d, beta = load()
    T8 = run_s8(d, beta)
    T11 = run_s11(d, beta)
    res = {"S8 오전→오후 모멘텀": summarize(T8), "S11 30분 잔차 반전 (1분 건너뜀)": summarize(T11)}
    for k, v in res.items():
        tr, va = v["학습"], v["검증"]
        kept = va.get("비용전_t", 0) >= 2
        ok = kept and va.get("순손익원", -1) > 0
        v["판정"] = "채택" if ok else ("비용 전 신호는 검증에서도 유지, 비용 뒤 손실" if kept else "검증에서 신호 사라짐 (비용 전 t < 2)")
    # 참고: S11 을 1분 건너뛰지 않고(신호 봉 종가 → 다음 봉 종가) 재면 비용 전 학습 +1,038원 t 3.0 / 검증 +5,347원 t 1.8
    #       → 반전은 마지막 체결가의 호가 튐이고, 1분 뒤 실제로 잡을 수 있는 가격에서는 0. 어느 쪽이든 왕복 비용(1.3만~6.6만원)의 1/10 이하
    (OUT / "signal_trade.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    T8.to_csv(OUT / "signal_trade_s8.csv", index=False, encoding="utf-8-sig")
    T11.to_csv(OUT / "signal_trade_s11.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
