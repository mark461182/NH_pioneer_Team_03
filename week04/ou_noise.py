"""OU 보정 — 1분 스프레드 = 「진짜 OU」 + 「호가 튐(관측 잡음)」 으로 나눠 다시 추정 (README 6-13)

  python ou_noise.py   → results/ou_noise_days.csv, results/ou_noise_summary.json

왜
  지금 OU(pair_core.fit_ou_days) 는 관측값 y 에 바로 AR(1) 을 씀. 그런데 1분 잔차 자기상관 −0.3 은
  y_t = x_t + ε_t (ε = 매 분 독립인 호가 튐) 일 때 생기는 모양이라, AR(1) 이 ε 의 되돌림까지
  「빠른 평균회귀」 로 읽어 반감기를 짧게, 기대이익을 크게 잡음 (7-2).

모형 (OU + 측정 잡음, 흔히 쓰는 상태공간 모형)
  x_t = φ·x_{t−1} + η_t        (진짜 스프레드, φ = e^{−κ})
  y_t = x_t + ε_t              (관측, ε ~ 독립, 분산 s²)
  → 자기공분산 γ_0 = V + s²,  γ_k = V·φ^k (k ≥ 1),  V = Var(x)
  추정: 날마다 평균 뺀 y 의 γ_1 … γ_5 (날 안 쌍만) → ln γ_k = ln V + k ln φ 회귀 → φ, V
        s² = γ_0 − V.  γ_k 가 양수가 아니면(되돌림 없음) 「평균회귀 안 보임」
  같은 2일 창·같은 비용(1틱 + 0.003%)·같은 ou_plan 으로, 그날 거래 가능한지와 분당 기대이익을 비교

β: 직전 60거래일 일간 회귀(signal_daily.csv). 2022-01 ~ 2026-11 전 기간.
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
MULT = 10


def autocov_days(days: list[np.ndarray], L: int = 5) -> np.ndarray:
    num, den = np.zeros(L + 1), np.zeros(L + 1)
    for x in days:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 30:
            continue
        x = x - x.mean()
        for k in range(L + 1):
            num[k] += float((x[k:] * x[:len(x) - k]).sum())
            den[k] += len(x) - k
    return num / np.where(den > 0, den, np.nan)


def fit_naive(g: np.ndarray) -> pc.OUFit:
    b = g[1] / g[0]
    if not (0 < b < 1):
        return pc.OUFit(False, why=f"b={b:.3f}")
    k = -math.log(b)
    return pc.OUFit(True, k, 0.0, math.sqrt(g[0]), math.log(2) / k, 0)


def fit_noise(g: np.ndarray) -> tuple[pc.OUFit, float]:
    ks = np.arange(1, len(g))
    gk = g[1:]
    if not np.all(gk > 0):
        return pc.OUFit(False, why="γ_k ≤ 0 — 되돌림 없음"), float("nan")
    slope, icpt = np.polyfit(ks, np.log(gk), 1)
    phi, V = math.exp(slope), math.exp(icpt)
    if not (0 < phi < 1):
        return pc.OUFit(False, why=f"φ={phi:.4f}"), float("nan")
    s2 = max(g[0] - V, 0.0)
    k = -math.log(phi)
    return pc.OUFit(True, k, 0.0, math.sqrt(V), math.log(2) / k, 0), math.sqrt(s2)


def main():
    d = pd.read_csv(OUT / "pair_1m_full.csv", dtype={"date": str, "time": str})
    d = d[d["time"] < "153000"]
    beta = pd.read_csv(OUT / "signal_daily.csv", index_col=0, parse_dates=True)["beta"]
    beta.index = beta.index.strftime("%Y%m%d")
    days = sorted(d["date"].unique())
    by = {k: g for k, g in d.groupby("date")}
    c = pc.Costs()
    rows = []
    for i in range(2, len(days)):
        day, prev = days[i], days[i - 2:i]
        bt = beta.get(day, np.nan)
        if not (isinstance(bt, float) and bt > 0):
            continue
        xs = [np.log(by[p].a_close.values) - bt * np.log(by[p].b_close.values) for p in prev]
        g = autocov_days(xs)
        pa, pb = by[prev[-1]].a_close.iloc[-1], by[prev[-1]].b_close.iloc[-1]
        qa = pc.hedge_qty(bt, pa, pb, pc.Params())
        rt = pc.round_trip_cost(qa, 1, pa, pb, c, MULT)
        r = {"date": day, "rho1": g[1] / g[0], "qa": qa, "cost_won": rt}
        f0 = fit_naive(g)
        f1, s = fit_noise(g)
        for tag, f in (("기존", f0), ("보정", f1)):
            if f.ok:
                cs = rt / (qa * pa * MULT * f.sigma_eq)
                pl = pc.ou_plan(f, cs)
                r.update({f"{tag}_반감기분": f.half_life, f"{tag}_σeq": f.sigma_eq, f"{tag}_비용σ": cs,
                          f"{tag}_거래가능": pl.ok, f"{tag}_진입z": pl.entry if pl.ok else np.nan,
                          f"{tag}_분당이익원": pl.rate * qa * pa * MULT * f.sigma_eq if pl.ok else 0.0})
            else:
                r.update({f"{tag}_반감기분": np.nan, f"{tag}_거래가능": False, f"{tag}_분당이익원": 0.0})
        r["잡음σ"] = s
        r["잡음비율"] = s ** 2 / g[0] if math.isfinite(s) else np.nan     # 1분 분산 중 호가 튐 몫
        rows.append(r)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "ou_noise_days.csv", index=False, encoding="utf-8-sig")

    def yr(X):
        return {"날수": int(len(X)),
                "1분자기상관_중앙": round(float(X.rho1.median()), 3),
                "잡음비율_중앙": round(float(X["잡음비율"].median()), 3),
                "반감기분_중앙_기존": round(float(X["기존_반감기분"].median()), 1),
                "반감기분_중앙_보정": round(float(X["보정_반감기분"].median()), 1),
                "비용σ_중앙_기존": round(float(X["기존_비용σ"].median()), 2),
                "비용σ_중앙_보정": round(float(X["보정_비용σ"].median()), 2),
                "거래가능날_기존": round(float(X["기존_거래가능"].mean()), 3),
                "거래가능날_보정": round(float(X["보정_거래가능"].mean()), 3),
                "하루기대이익원_중앙_기존": int(X["기존_분당이익원"].median() * 380),
                "하루기대이익원_중앙_보정": int(X["보정_분당이익원"].median() * 380)}
    S = {"전체": yr(D)}
    for y, X in D.groupby(D.date.str[:4]):
        S[y] = yr(X)
    (OUT / "ou_noise_summary.json").write_text(json.dumps(S, ensure_ascii=False, indent=1), encoding="utf-8")
    print(pd.DataFrame(S).to_string())


if __name__ == "__main__":
    main()
