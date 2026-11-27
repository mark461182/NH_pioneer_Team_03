"""6주차 — 어림값으로 정했던 기준의 데이터 근거 (2025-10 ~ 2026-11, 09:15~15:15)

  python evidence6.py   → results/evidence6.json

① 하루 손절 횟수 — 설정 C(하루 손절 한도 없음)에서, 그날 앞서 손절이 k번 난 뒤 들어간 거래의 평균 순손익
② 최대 2단위 — 1분봉 거래량 대비 주문 크기
③ 감시 끊김 60초·180초, 한 다리 노출 15만 원·60초 — 1분·3분 가격·스프레드 변화의 꼬리
④ 시세 멈춤 120초 — **빠진 분**을 셈. NH 분봉은 체결이 없는 분은 봉이 아예 안 오므로 거래량 0 인 봉은 셀 수 없음.
   pair_1m_full.csv 는 두 다리를 inner join 한 것이라, 빠진 분 = 「적어도 한 다리에 체결이 없던 분」 (어느 다리인지는 모름)
⑤ OU 모형 확률 vs 실제 빈도 — theory6.json 의 확률과 백테스트 거래에서 센 비율 (11/24 추가)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from backtest6 import CACHE, find_data

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
STOPS = ["가격손절", "고정z손절", "금액손절", "시간손절", "상관붕괴", "이익보호"]


def main():
    out = {}
    d = pd.read_csv(find_data(), dtype={"date": str, "time": str})
    d = d[(d.time >= "091500") & (d.time < "151500")]
    rec = d[d.date >= "20251001"]
    last = d.iloc[-1]
    unit_a = round(last.b_close / last.a_close)
    out["1단위"] = {"삼성계약": int(unit_a), "명목원_두다리": int((last.b_close + unit_a * last.a_close) * 10)}

    # ① 하루 손절 횟수
    T = pd.read_csv(OUT / "trades_C.csv", dtype={"date": str}, parse_dates=["t_in"])
    ep = T.groupby("ep").agg(date=("date", "first"), t_in=("t_in", "min"), net=("net", "sum"),
                             stop=("exit", lambda s: s.isin(STOPS).any())).sort_values("t_in")
    ep["앞선손절"] = ep.groupby("date")["stop"].transform(lambda s: s.shift(fill_value=False).cumsum())
    g = ep.groupby("앞선손절")["net"].agg(["count", "mean"])
    out["하루손절"] = {"설정": "C (하루 손절 한도 없음), 전 기간, 포지션(분할 묶음) 단위",
                    "손절로_센_청산": STOPS + ["— 엔진의 하루 손절 횟수(STOP_REASONS)와 같은 정의, 이익보호 포함"],
                    "거래당_손절비율": round(float(ep.stop.mean()), 3),
                    "그날_앞선손절별": {int(k): {"거래": int(r["count"]), "평균순손익원": int(r["mean"])} for k, r in g.iterrows()}}
    alt = {}
    for name, stops in (("C_이익보호뺌", [x for x in STOPS if x != "이익보호"]), ("D", STOPS)):
        TT = pd.read_csv(OUT / f"trades_{name[0]}.csv", dtype={"date": str}, parse_dates=["t_in"])
        e2 = TT.groupby("ep").agg(date=("date", "first"), t_in=("t_in", "min"), net=("net", "sum"),
                                  stop=("exit", lambda s: s.isin(stops).any())).sort_values("t_in")
        e2["k"] = e2.groupby("date")["stop"].transform(lambda s: s.shift(fill_value=False).cumsum())
        alt[name] = {int(k): {"거래": int(r["count"]), "평균순손익원": int(r["mean"])}
                     for k, r in e2.groupby("k")["net"].agg(["count", "mean"]).iterrows()}
    out["하루손절"]["다른정의"] = alt

    # ② 거래량
    out["거래량_1분"] = {k: {"중앙": int(rec[f"{k}_volume"].median()), "하위10%": int(rec[f"{k}_volume"].quantile(.1))}
                       for k in ("a", "b")}
    r2 = rec.copy()
    r2["t"] = pd.to_datetime(r2.date + r2.time)
    gap = (r2.groupby("date")["t"].diff().dt.total_seconds() / 60 - 1).dropna()
    runs = gap[gap > 0].value_counts().sort_index()
    days = r2.date.nunique()
    out["빠진분"] = {"거래일": int(days), "있는봉": int(len(r2)), "빠진분": int(gap.sum()),
                   "빠진비율": round(float(gap.sum() / (len(r2) + gap.sum())), 4),
                   "연속으로_빠진_분수별_횟수": {int(k): int(v) for k, v in runs.items()},
                   "2분이상_연속_횟수": int((gap >= 2).sum()),
                   "주의": "inner join 이라 어느 다리가 빠졌는지 모름. 29분 연속 9번은 원인 모름(데이터 수집 문제일 수 있음)"}

    # ③ 가격·스프레드 꼬리
    px = {}
    for k in ("a", "b"):
        lg = np.log(rec[f"{k}_close"]).groupby(rec.date)
        px[k] = {"1분_p99": round(float(lg.diff(1).abs().quantile(.99)), 4), "3분_p99": round(float(lg.diff(3).abs().quantile(.99)), 4)}
    out["가격변화_절대로그"] = px
    nota, notb = unit_a * last.a_close * 10, last.b_close * 10
    out["한다리_손실원_p99"] = {f"{u}단위": {"삼성다리_1분": int(u * px["a"]["1분_p99"] * nota),
                                         "삼성다리_3분": int(u * px["a"]["3분_p99"] * nota),
                                         "하이닉스다리_1분": int(u * px["b"]["1분_p99"] * notb),
                                         "하이닉스다리_3분": int(u * px["b"]["3분_p99"] * notb),
                                         "실행기_기준원": 150_000 * u} for u in (1, 2)}
    _, F = pickle.loads(sorted(CACHE.glob("plans_2022*.pkl"))[-1].read_bytes())
    F = F[F.index >= "2025-10-01"]
    day = F.index.normalize()
    q = {}
    for m in (1, 3):
        ch = (F["spread"].groupby(day).diff(m) / F["sd"]).dropna().abs()
        q[f"{m}분"] = {"p50": round(float(ch.median()), 2), "p99": round(float(ch.quantile(.99)), 2), "p999": round(float(ch.quantile(.999)), 2)}
    out["스프레드변화_σeq배"] = q
    # ⑤ OU 모형 확률 vs 실제
    th = json.loads((OUT / "theory6.json").read_text(encoding="utf-8"))
    A1 = pd.read_csv(OUT / "trades_A1.csv")
    A = pd.read_csv(OUT / "trades_A.csv", parse_dates=["t_in"])
    plans, _ = pickle.loads(sorted(CACHE.glob("plans_2022*.pkl"))[-1].read_bytes())
    mh = A["t_in"].dt.normalize().map(lambda d: plans[d].params.max_hold if plans[d].ok else np.nan)
    C = pd.read_csv(OUT / "trades_C.csv")
    has2 = C.groupby("ep")["tag"].apply(lambda s: (s == "2차").any())
    out["OU_모형_vs_실제"] = {
        "가격손절선(a+1.5)에_먼저닿음": {"모형": th["가격손절_도달확률"]["a+1.5"], "실제": round(float((A1.exit == "가격손절").mean()), 3),
                                 "실제_건": f"{int((A1.exit == '가격손절').sum())}/{len(A1)} (A1: 위험관리 없음 + 가격손절)"},
        "반감기3배_넘게_보유": {"모형": th["수렴시간_반감기단위"]["P(>3HL)"], "실제": round(float((A.hold_min > mh).mean()), 3),
                         "실제_건": f"{int((A.hold_min > mh).sum())}/{len(A)} (A: 위험관리 없음)"},
        "2차진입선(a+0.5)에_닿음": {"모형": th["2차진입선_도달확률"], "실제": round(float(has2.mean()), 3),
                               "실제_건": f"{int(has2.sum())}/{len(has2)} (C: 분할 있음)"}}
    (OUT / "evidence6.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
